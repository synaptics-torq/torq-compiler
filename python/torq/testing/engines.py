import pytest
from pathlib import Path

engines = None

# This is imported from extras/python
def get_engines() -> list:
    global engines

    if engines is not None:
        return engines

    try:
        from engines import engines_runner
        engines = engines_runner

    except ImportError:
        print("engines import failed")
        class _DummyEngines:
            @staticmethod
            def compile(*args, **kwargs):
                return {}

            @staticmethod
            def get_compilation_time(*args, **kwargs):
                return None
            
            @staticmethod
            def list_engines(*args, **kwargs):
                return []

        engines = _DummyEngines()

    return engines


@pytest.fixture
def alternative_engine(request):
    return request.param


def pytest_addoption(parser):
    parser.addoption("--torq-test-alternative-engines", action="store", default="", help="Comma-separated list of alternative engines to test.")


def pytest_generate_tests(metafunc):

    if "alternative_engine" in metafunc.fixturenames:
        engines_runner = get_engines()
        all_engines = engines_runner.list_engines()
        print("available engines: ", all_engines)
        selected = metafunc.config.getoption("torq_test_alternative_engines")

        if selected:
            selected = [e.strip() for e in selected.split(",") if e.strip()]

            unknown = [e for e in selected if e not in all_engines]
            if unknown:
                raise pytest.UsageError(
                    f"Unknown engine(s): {', '.join(unknown)}"
                )
            values = selected
        else:
            values = all_engines

        metafunc.parametrize("alternative_engine", values)

    
def compile_with_engine(request, tflite_model_path, alternative_engine):
    engines_runner = get_engines()


    tmp_path = request.getfixturevalue("tmp_path")
    record_property = request.getfixturevalue("record_property")

    model_path = Path(tflite_model_path.data)
    out_dir = tmp_path / model_path.stem

    # Compile with the current engine
    metadata = engines_runner.compile(alternative_engine, str(model_path), str(out_dir))
    
    # Get compilation time
    engine_compilation_time = engines_runner.get_compilation_time(alternative_engine, str(model_path), str(out_dir))

    for key, value in metadata.items():
        record_property(key, value)

    record_property("compile_time_measurements", {'total_duration': engine_compilation_time * 1_000_000})
