"""Write small ONNX test graphs op by op with onnx_graphsurgeon.

Example::

    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, [1, 207])
    y = g.op("Sqrt", x, dtype=BF16, shape=[1, 207])
    model = g.export(y)
"""

from collections import Counter

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import LazyValues

BF16 = ml_dtypes.bfloat16
I4 = ml_dtypes.int4
F32 = np.float32
I8 = np.int8
I32 = np.int32
I64 = np.int64


class OnnxGraph:
    def __init__(self, opset, domains=None):
        """domains: {domain: version} of the opsets to import besides the default one."""
        self.opset = opset
        self.domains = domains or {}
        self.inputs = []
        self.nodes = []
        self._names = Counter()

    def _name(self, base):
        self._names[base] += 1
        count = self._names[base]
        return base if count == 1 else f"{base}_{count}"

    def input(self, name, dtype, shape):
        """A graph input."""
        x = gs.Variable(name, dtype=dtype, shape=list(shape))
        self.inputs.append(x)
        return x

    def const(self, value, dtype, name=None):
        """A constant from a scalar, a nested list or an array."""
        values = np.asarray(value, dtype=dtype)
        if values.dtype == I4:
            # GraphSurgeon 0.6.1 does not pack INT4 arrays; ONNX handles their storage.
            values = LazyValues(onnx.numpy_helper.from_array(values))
        return gs.Constant(name or self._name("const"), values)

    def op(self, op_type, *inputs, outputs=1, dtype=None, shape=None, domain=None, **attrs):
        """Adds a node and returns its output, or the list of its outputs if outputs > 1.

        An input of None is an omitted optional input. dtype and shape declare the type of a
        single output (needed for graph outputs).
        """
        inputs = [gs.Variable.empty() if x is None else x for x in inputs]
        results = [gs.Variable(self._name(op_type.lower())) for _ in range(outputs)]
        self.nodes.append(gs.Node(op_type, inputs=inputs, outputs=results, attrs=attrs,
                                  domain=domain))
        if outputs > 1:
            return results
        return typed(results[0], dtype, shape) if dtype is not None else results[0]

    def _tensor(self, value, dtype):
        return value if isinstance(value, gs.Tensor) else self.const(value, dtype)

    def quantize(self, x, scale, zero_point, zp_dtype=I8, **kwargs):
        """QuantizeLinear; scale and zero_point are numbers (f32 / zp_dtype) or tensors."""
        return self.op("QuantizeLinear", x, self._tensor(scale, F32),
                       self._tensor(zero_point, zp_dtype), **kwargs)

    def dequantize(self, x, scale, zero_point, zp_dtype=I8, **kwargs):
        """DequantizeLinear; scale and zero_point are numbers (f32 / zp_dtype) or tensors."""
        return self.op("DequantizeLinear", x, self._tensor(scale, F32),
                       self._tensor(zero_point, zp_dtype), **kwargs)

    def qdq(self, x, scale, zero_point, zp_dtype=I8, **kwargs):
        """QuantizeLinear then DequantizeLinear with the same parameters."""
        scale, zero_point = self._tensor(scale, F32), self._tensor(zero_point, zp_dtype)
        return self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point, **kwargs)

    def export(self, *outputs, ir_version=10) -> onnx.ModelProto:
        """The model with these graph outputs; each must have a declared type."""
        graph = gs.Graph(nodes=self.nodes, inputs=self.inputs, outputs=list(outputs),
                         opset=self.opset)
        if self.domains:
            graph.import_domains = [onnx.helper.make_opsetid("", self.opset)] + [
                onnx.helper.make_opsetid(domain, version)
                for domain, version in self.domains.items()
            ]
        return gs.export_onnx(graph, ir_version=ir_version)


def typed(x, dtype, shape):
    """Declares the type of a variable (a graph output, or an intermediate to annotate)."""
    x.dtype = dtype
    x.shape = list(shape)
    return x
