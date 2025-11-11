import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter_ns

import numpy.typing as npt

__all__ = [
    "InferenceRunner",
]


class InferenceRunner(ABC):

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
    def _infer(self, inputs: Sequence[npt.NDArray] | Mapping[str, npt.NDArray]) -> Sequence[npt.NDArray]:
        ...

    def infer(self, inputs: Sequence[npt.NDArray] | Mapping[str, npt.NDArray]) -> Sequence[npt.NDArray]:
        st = perf_counter_ns()
        results = self._infer(inputs)
        self._infer_time_ms = (perf_counter_ns() - st) / 1e6
        return results
