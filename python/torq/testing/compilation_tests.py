from pathlib import Path
import re
import pytest
import abc

import ml_dtypes # for bf16 support in numpy
import numpy as np

def check_nans(arr1, arr2):
    nan1 = np.isnan(arr1)
    nan2 = np.isnan(arr2)
    assert (nan1 == nan2).all(), "Nans differ."
    arr1 = arr1.copy()
    arr2 = arr2.copy()
    # Replace NaNs with 0 so that we don't break comparison
    # Skip if no NaNs to avoid errors with integer arrays
    if nan1.any():
        arr1[nan1] = 0
    if nan2.any():
        arr2[nan2] = 0
    return arr1, arr2


class CachedTestDataFile:
    """
    Create an attribute of a class that is generated using the generation_func and
    cached to the file file_path inside the cache_dir of the instance.

    The cached file can be forced to be recomputed by passing the --recompute-cache
    to pytest.

    The data generated on disk at most once per test session even when recomputation 
    is forced.

    Usage:

    class MyTest(CompilationTestCase):
        my_data_file = CachedTestDataFile("my_data.npy", "generate_my_data")

        def generate_my_data(self, file_path):
            # generate data and save to file_path
            ...

    """
    def __init__(self, file_path, generation_func):
        self.file_path = file_path
        self.generation_func = generation_func        
        self.attr_name = f"_{id(self)}_generated"
    
    def __get__(self, instance, owner):

        full_file_path = instance.cache_dir / self.file_path

        # check if we already made sure the file exists in this session
        if not hasattr(instance, self.attr_name):

            force_recompute = instance.request.config.getoption("--recompute-cache")

            if full_file_path.exists():
                test_case_mtime = instance.request.node.fspath.stat().mtime
                file_mtime = full_file_path.stat().st_mtime

                if file_mtime < test_case_mtime:
                    print("[outdated cache detected] Test case modified after cached data generation, forcing recompute")
                    force_recompute = True

            if not full_file_path.exists() or force_recompute:
                print("[test data] generating", full_file_path)
                generation_func = getattr(instance, self.generation_func)
                generation_func(full_file_path)

            # mark as done for this session
            setattr(instance, self.attr_name, True)

        else:
            print("[test data] using existing", full_file_path)

        return full_file_path


class CachedTestData:
    """
    Generate and cache test data to a file inside the cache_dir of the instance.

    The cached data can be forced to be recomputed by passing the --recompute-cache
    to pytest.

    The data is loaded from disk only once per test session.

    Usage:

    class MyTest(CompilationTestCase):
        my_data = CachedTestData("my_data.npy", "generate_my_data")

        def generate_my_data(self):
            # generate data and return it    
        
    """

    def __init__(self, file_path, generation_func):
        self.file_path = file_path
        self.generation_func = generation_func        
        self.attr_name = f"_{id(self)}_data"
    
    def __get__(self, instance, owner):

        force_recompute = instance.request.config.getoption("--recompute-cache")

        full_file_path = instance.cache_dir / self.file_path

        if not hasattr(instance, self.attr_name):            

            if full_file_path.exists():
                test_case_mtime = instance.request.node.fspath.stat().mtime
                file_mtime = full_file_path.stat().st_mtime

                if file_mtime < test_case_mtime:
                    print("[outdated cache detected] Test case modified after cached data generation, forcing recompute")
                    force_recompute = True

            if not full_file_path.exists() or force_recompute:
                print("[test data miss] generating", full_file_path)
                generation_func = getattr(instance, self.generation_func)
                value = generation_func()
                setattr(instance, self.attr_name, value)

                serializable_data = np.empty((len(value),), dtype=object)
                serializable_data[:] = value
                np.save(full_file_path, serializable_data, allow_pickle=True)
            else:
                print("[test data hit] loading", full_file_path)
                serialized_data = np.load(full_file_path, allow_pickle=True)                
                data = [x for x in serialized_data]                
                setattr(instance, self.attr_name, data)
        else:
            print("[test data hit] using loaded", full_file_path)

        return getattr(instance, self.attr_name)


class CompilationTestCase(metaclass=abc.ABCMeta):
    """
    Base class for test cases that compile models and check accuracy.

    This class relies on the compile, execute, and check_results methods
    being implemented by subclasses.

    The class automatically makes available the current test request object
    as self.request for use in methods.

    The main test method is test_model which performs the compilation,
    execution, and result checking.

    Subclasses can define CachedTestData and CachedTestDataFile attributes
    to manage test data generation and caching. These are generated before
    execution of tests starts.
    """

    @pytest.fixture(autouse=True)
    def _stash_request(self, request):
        """
        Stash the pytest request object for later use
        """
        self.request = request

    def generate_test_data(self):
        """
        Generates the data required to perform the test
        """

        for attr in dir(self):
            attr_value = getattr(self, attr)
            if isinstance(attr_value, CachedTestData):
                # Access the attribute to trigger data generation
                _ = getattr(self, attr)
            elif isinstance(attr_value, CachedTestDataFile):
                # Access the attribute to trigger data generation
                _ = getattr(self, attr)

    @abc.abstractmethod
    def generate_input_data(self):
        """
        Generates input data for the model
        """
        pass

    input_data = CachedTestData("input_data.npy", "generate_input_data")

    @abc.abstractmethod
    def compile(self):
        """
        Compiles the model and returns the compiled artifact
        """
        pass

    @abc.abstractmethod
    def execute(self, compiled_model, input_data):
        """
        Executes the compiled model with the given input data and returns the results
        """
        pass

    @abc.abstractmethod
    def check_results(self, results):
        """
        Checks the results of the model execution for correctness
        """
        pass

    @property
    def cache_dir(self) -> Path:
        """
        Returns a directory path unique to this test case and its parameters
        """

        if not hasattr(self, "_cache_dir"):            
            
            test_parts = str(self.request.node.nodeid).split("::")

            # remove the .py suffix from the first part which is the file name
            test_parts[0] = test_parts[0][:-len(".py")]

            # remove the last part which is the test name if the test is not parameterized
            if test_parts[-1] == "test_model":
                test_parts = test_parts[:-1]
            else:
                test_parts = test_parts[:-1] + [test_parts[-1][len("test_model["):-1]]            

            test_parts = test_parts[0].split("/") + test_parts[1:]

            root_dir = Path(self.request.config.cache.mkdir("compilation_test_data"))

            self._cache_dir = root_dir / "/".join(test_parts)

            self._cache_dir.mkdir(parents=True, exist_ok=True)

        return self._cache_dir


    def test_model(self, request):
        """
        Main test method that compiles, executes, and checks the model outputs
        """

        self.generate_test_data()

        compiled_model = self.compile()
        results = self.execute(compiled_model, self.input_data)

        self.check_results(results)


class WithParameters:
    """
    Mixin class to provide parameters to a test case, the parameters are visibile
    as self.params inside the test case class
    """

    @pytest.fixture(autouse=True)
    def _stash_params(self, params):
        self.params = params


def parametrize(argvalues):
    """
    Parametrization decorator for test cases using WithParameters mixin.

    :param argvalues: List of parameter values, if the objects have a idfn method
                        it is used to generate the test ids.
    """
    if len(argvalues) > 0 and hasattr(argvalues[0].__class__, 'idfn'):
        return pytest.mark.parametrize("params", argvalues, ids=argvalues[0].__class__.idfn)
    else:
        return pytest.mark.parametrize("params", argvalues)
