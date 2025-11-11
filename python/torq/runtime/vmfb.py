import os
from typing import Literal

import numpy as np
import iree.runtime as iree_rt

from .base import InferenceRunner


class VMFBInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_threads: int | None = None,
        function: str = "main",
        device_uri: str = "local-task",
        load_method: Literal["preload", "mmap"] = "mmap"
    ):
        super().__init__(model_path)

        if isinstance(n_threads, int):
            iree_rt.flags.parse_flags(
                f"--task_topology_group_count={n_threads}",
                f"--task_topology_max_group_count={n_threads}"
            )

        if load_method == "mmap":
            module = iree_rt.load_vm_flatbuffer_file(self._model_path, driver=device_uri)
            vm_module = module.vm_module
        else:
            instance = iree_rt.VmInstance()
            device = iree_rt.get_device(device_uri, cache=False)
            hal_module = iree_rt.create_hal_module(instance, device)
            with open(self._model_path, "rb") as f:
                fb = f.read()
            vm_module = iree_rt.VmModule.from_flatbuffer(instance, fb, warn_if_copy=False)
            _ctx = iree_rt.SystemContext(vm_modules=[hal_module, vm_module])
            module = _ctx.modules[vm_module.name]

        if function not in vm_module.function_names:
            raise ValueError(f"Function '{function}' not found in '{self._model_path}'")
        self._invoker = module[function]

    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        if isinstance(inputs, dict):
            inputs = list(inputs.values())
        result: iree_rt.DeviceArray | tuple[iree_rt.DeviceArray] = self._invoker(*inputs)
        if isinstance(result, tuple):
            return [r.to_host() for r in result]
        return [result.to_host()]
