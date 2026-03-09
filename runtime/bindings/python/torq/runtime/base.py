import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter_ns

import numpy.typing as npt

__all__ = [
    "InferenceRunner",
]


class InferenceRunner(ABC):
    """Abstract base for model inference runners."""

    def __init__(
        self,
        model_path: str | os.PathLike,
    ):
        self._model_path: Path = Path(model_path)
        self._infer_time_ms: float = 0.0

    @property
    def model_path(self) -> str | os.PathLike:
        return self._model_path

    @property
    def infer_time_ms(self) -> float:
        return self._infer_time_ms

    @abstractmethod
    def _infer(self, inputs: Iterable[npt.NDArray] | Mapping[str, npt.NDArray]) -> Sequence[npt.NDArray]:
        ...

    def infer(self, inputs: Iterable[npt.NDArray] | Mapping[str, npt.NDArray]) -> Sequence[npt.NDArray]:
        """Run inference and record elapsed time in ``infer_time_ms``.

        Args:
            inputs: Input arrays or a name-to-array mapping.

        Returns:
            Sequence of output arrays.
        """
        st = perf_counter_ns()
        results = self._infer(inputs)
        self._infer_time_ms = (perf_counter_ns() - st) / 1e6
        return results
