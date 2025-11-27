# Testing

## Running tests

Testing of the compiler and runtime uses the ``pytest`` framework. If you have configured your development environment
as describe in [getting started section](getting_started.md), you should be ready for running tests.

To run the tests first ensure the development python virtual environment is active. This is automatic when using the
development docker. When using a native build you need to activate it using the command line:

```
$ source ../venv/bin/activate
```

Once the virtual environment is active you can execute the whole test plan with the following command:

```
$ pytest
```

Testing caches intermediates results and normally detected automatically changes that require cache invalidation. To
force cache invalidation use the command line `--recompute-cache``:

```
$ pytest --recompute-cache
```

The cache is stored in ``.pytest_cache`` and can be safely removed across tests sessions if desired.

The whole test plan is quite long and during development you will often want to execute only some specific tests.

To enumerate all available test cases use the following command:

```
$ pytest --collect-only
```

You can then execute a specific test by specifiying its name:

```
$ pytest tests/test_foo.py::test_bar[baz]
```

Or you can run all tests that match a given pattern:

```
$ pytest -k somepattern
```

To learn more about how to configure the execution of tests see the official [pytest documentation](https://docs.pytest.org/)

## How test code works

A Pytest test invocation works in two phases: in the first phase called collection pytest finds all
 the available tests, and in the second phase it executes them.

Collection of tests starts by scanning all python modules found in the ``tests`` directory and by finding all
modules that start with the prefix ``test_``. Collection then loads each of these modules and looks for any function
that starts with the pattern ``test_``.

In execution phase pytest invokes each of the tests functions. If the function raises an exception the 
test is considered as failed, otherwise it is considered as successful.

For example, if there exists a file ``tests/test_foo.py`` with the following content:

```

from mycode import bar

def test_bar():
    assert bar(2) == 1

```

Pytest in the collection phase will find a test case ``tests/test_foo.py::test_bar`` and in the execution
phase it will invoke the test function and checks if aseert raises an exception.

### Fixtures

Often multiple tests require the same inputs. Pytest allows to factor this code using the fixture concept.

For instance:

```
from mycode import bar, buz
import pytest

@pytest.fixture
def complex_input():
    # some complex code


def test_bar(complex_input):
    assert bar(complex_input) == 1


def test_buz(complex_input):
    assert buz(complex_input) == 1

```

This code defines two tests cases, ``test_bar`` and ``test_buz`` that require the same complex input.
In order to factor the code used to generate the input into a function ``complex_input`` that we
decorate with the ``pytest.fixture`` decorator. This makes the fixture known to pytest.

The two tests request the **output** of the fixture function by adding a parameter with the corresponding
name to their signature. During test execution pytest will check the signature of each test function to 
discover which fixtures were requested and for each requested fixture it invokes the corresponding fixture
function. It then invokes the test fucntion passing as argument the result of the fixture function.

Fixtures can themselves request other fixtures, for instance:

```
from mycode import bar
import pytest


@pytest.fixture
def other_complex_data():
    # compute some other complex data


@pytest.fixture
def complex_input(other_complex_data):
    # some complex code that that creates the complex
    # input using other_complex_data


def test_bar(complex_input):
    assert bar(complex_input) == 1
```

### Parametrization

In many cases test functions need to run the same check on a range of input parameters. To compactly describe
these types of tests pytest allows to parametrize fixtures as follows:

```
from mycode import bar
import pytest


@pytest.fixture(params=[1,2,3])
def bar_input(request):    
    return 2**(request.param)


def test_bar(bar_input):
    assert bar(bar_input) == bar_input * 2
```

During collection pytest will detect the ``test_bar`` test and check what are the fixtures that it requires,
and will see it needs the fixture ``bar_input``. When it looks for this fixture, it detect that it has
been parametrized with a sequence of three values. It will therefore create one test case for each of three 
variants of the fixture. This will results in three tests: ``tests/test_foo.py::test_bar[1]``, 
``tests/test_foo.py::test_bar[2]`` and ``tests/test_foo.py::test_bar[3]``. These tests can be invoked
individually. 

Parameters are evaluated at collection time and therefore the corresponding code will run
even if the corresponding tests will not be selected for exection. It is therefore very important that
their computation is fast otherwise they will slow down all test sessions.

### Multiple parameters

If a test depends on more than one parametrized fixture pytest will create all possible combinations 
of parameters and genenerate one test case per combination. For instance the following test code:

```
from mycode import bar
import pytest


@pytest.fixture(params=[1,2,3])
def bar_input1(request):    
    # some code that uses request.param

@pytest.fixture(params=[1,2,3])
def bar_input2(request):    
    # some other code that uses request.param

def test_bar(bar_input1, bar_input2):
    assert bar(bar_input1, bar_input2) == True
```

will generate 9 test cases of the form ``tests/test_foo.py::test_bar[x-y]``.

In order to test a subset of these combination we use the classes in ``torq.testing.case``. These
clases provide a way to restrict the combinations of tests we want to run. Assume for instance we want
to test only the combinations (0, 1) and (3, 2). We can define the test as follows:

```
from torq.testing.case import Case

from mycode import bar
import pytest

def get_cases():
    return [Case("first", {"bar_input1": 0, "bar_input2": 1}),
            Case("second", {"bar_input1": 3, "bar_input2": 2})]

@pytest.fixture(get_cases())
def case_config(request):
    return request.param

@pytest.fixture
def bar_input1(case_config):    
    # some code that uses case_config.data["bar_input1"]

@pytest.fixture
def bar_input2(case_config):    
    # some code that uses case_config.data["bar_input2"]

def test_bar(bar_input1, bar_input2):
    assert bar(bar_input1, bar_input2) == True
```

This code defines a function that generates two ``Case`` objects, named ``first`` and ``second``. This list
of cases is used to parametrize the fixture ``case_config``. The two fixtures ``bar_input1`` and 
``bar_input2`` depend on this fixture and read the corresponding value from the ``case_config`` fixture. When 
pytest collects the tests it will see that there are 2 versions of the ``case_config`` fixture (one
for each ``Case`` object) and one version of ``bar_input1`` and ``bar_input2``, it will therefore create
2 * 1 * 1 test cases, ``tests/test_foo.py::test_bar[first]`` and ``tests/test_foo.py::test_bar[second]``
which is what we want.

### Caching

By default fixture functions are invoked once for each test case. It is possible to indicate to pytest that
a fixture can be re-used for all test in a module by marking it with the scope ``module`` or for the whole
test session with the scope ``session``:

```
from mycode import bar, baz
import pytest

@pytest.fixture(scope="module")
def complex_input():
    # some complex code


def test_bar(complex_input):
    assert bar(complex_input) == 1

def test_baz(complex_input):
    assert baz(complex_input) == 1


```

In this case ``complex_input`` will be computed only once and re-used for both ``test_bar`` and ``test_baz``.

In compiler development often can even be re-used across test sessions. For instance reference results
of a inference never change. In order to enable caching across sessions we use the decorators defined in the
``torq.testing.versioned_fixture`` module:

```
from torq.testing.versioned_fixtures import versioned_cached_data_fixture

from mycode import bar
import pytest

@versioned_cached_data_fixture
def complex_input():
    # some complex code


def test_bar(complex_input):
    assert bar(complex_input.data) == 1

```

By marking a fixture function with the ``versioned_cached_data_fixture`` the function is executed once
in the first test session and the result is stored in a file in  ``.pytest_cache/versioned_fixtures/complex_input``.

In the next test session the fixture will not be recomputed and instead it will be loaded from disk.

The returned object from a versioned fixture is an object that provides the actual data of the fixture in 
the ``data`` field and the version of this fixture in the ``version`` field. The version of the fixture
depends on the inputs of the fixture and is automatically changed when they change.

It is also possible to cache fixtures that return a path to a generated file. These fixtures return
an object with a ``version`` field and a ``file_path`` field pointing to the generated file.

```
from torq.testing.versioned_fixtures import versioned_generated_file_fixture

from mycode import bar
import pytest
from pathlib import Path

@versioned_generated_file_fixture
def complex_input(versioned_file: Path):
    # some complex code that genrates versioned_file


def test_bar(complex_input):
    assert bar(complex_input.file_path) == 1

```

The files are again generated in ``.pytest_cache/versioned_fixtures/complex_input`` and pytest will
re-use the content of the file if the version of the file has not changed.

Notice that ``versioned_file`` is a special argument that provides the path where the file should 
be generated.

Anologuously it is possible to cache a fixture that creates a set of files in a directory using the
``versioned_generated_directory_fixture``.

Sometimes cached fixtures depend on other fixtures, if the value of the other fixture changes this will
potentially impact the value of the cached fixture. To detect this, the versioned fixture code keeps track
of the version of any input of a given fixture and automatically changes the version of the output object
accordingly. Different versions of the fixture are stored on disk in different files so that multiple 
versions can be cached at the same time.

In practice there are two reasons why a cached fixture may change: either the fixture depends on a 
parametrized fixture or it depends on an external file that changed. It is possible to version fixtures
for both cases as follows:

```

@versioned_static_file_fixture
def versione_file_path():
    # return a posix path of an externally generated file 


@versioned_hashable_object_fixture
def versioned_hashable_object(input1, input2):
    # returns a hashable object


@versioned_unhashable_object_fixture
def versioned_unhashable_object(hashable_input1, hashable_input2):
    # returns an object that may not be hashable

```

The ``versioned_static_file_fixture`` returns a versioned file where the version of the object is the mtime of the
file returned. When the mtime changes the version changes, invalidating any other versioned fixture that depends
on it.

The ``versioned_hashable_object_fixture`` returns versiond data where the version of the data is the hash of the
data itself (the hash is computed by transforming the object to json and computing an hash of the resulting string).
When the code computes a different value, automatically any versioned fixture depending on it will be invalidated.

The ``versioned_unhashable_object_fixture`` returns versiond data where the version of the data is the hash of the
inputs to the fixture, which must be hashable. This can be used when the fixture has to return an object for which
it is not possible to compute an hash or computing it is very expensive.

With these helper fixtures it is possible to version parametrized fixtures:

```
from torq.testing.versioned_fixtures import versioned_cached_data_fixture

from mycode import bar
import pytest


@pytest.fixture(params=[1,2,3])
def myparameter(request):
    return request.param


@versioned_hashable_object_fixture
def versioned_myparameter(myparameter):
    return myparameter


@versioned_cached_data_fixture
def complex_input(versioned_myparameter):
    # some complex code that depent on parent_fixture


def test_bar(complex_input):
    assert bar(complex_input.data) == 1

```

In this case pytest will generate three cached versions of the ``complex_input`` fixture.

It is also possible to auto-invalidate the pytest cache when an external file change:


```
from torq.testing.versioned_fixtures import versioned_cached_data_fixture
from pathlib import Path

from mycode import bar
import pytest


@versioned_static_file_fixture
def versioned_external_file():
    return Path("/some/external/file")


@versioned_cached_data_fixture
def complex_input(versioned_external_file):
    # generate some complex input based on versioned_external_file


def test_bar(complex_input):
    assert bar(complex_input.data) == 1

```


