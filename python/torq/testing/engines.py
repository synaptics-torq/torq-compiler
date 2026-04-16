# This is imported from extras/python
try:
    from engines import engines_runner as engines

except ImportError:
    class _DummyEngines:
        @staticmethod
        def compile(*args, **kwargs):
            pass

        @staticmethod
        def get_compilation_time(*args, **kwargs):
            return None
        
        @staticmethod
        def list_engines(*args, **kwargs):
            return []

    engines = _DummyEngines()