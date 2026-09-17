import pytest

from torq.lab import LabError
from torq.lab.pipeline.io import ConvertIODTypesPolicy

from .versioned_fixtures import (
    versioned_hashable_object_fixture,
    versioned_unhashable_object_fixture,
)


@versioned_hashable_object_fixture
def convert_io_dtypes_args(request):
    return list(request.config.getoption("--convert-io-dtypes"))


@versioned_unhashable_object_fixture
def convert_io_dtypes_policy(request, torq_compiler_options, convert_io_dtypes_args) -> ConvertIODTypesPolicy:
    torq_convert_dtypes = {"--torq-convert-dtypes", "--torq-convert-io-dtype"}.issubset(torq_compiler_options)
    try:
        return ConvertIODTypesPolicy.parse_from_args(convert_io_dtypes_args, torq_convert_dtypes)
    except LabError as exc:
        raise pytest.UsageError(str(exc)) from exc
