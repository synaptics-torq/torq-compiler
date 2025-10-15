#!/usr/bin/env python
# coding=utf-8

from mlir.ir import *
from mlir.dialects import func
from mlir.dialects import arith
from mlir.dialects import memref
from mlir.dialects import affine
import mlir.extras.types as T

from iree_compiler.iree.compiler.dialects import torq_hl
from iree_compiler.iree.compiler.dialects import torq_hw


from mlir.ir import Context, Module

with Context() as ctx:
  ctx.allow_unregistered_dialects = True

  with open("tests/testdata/torqhl_ops/segmentation.mlir", "r") as file:
    module = Module.parse(file.read(), ctx)

module.body.operations[0].print()

func = module.body.operations[0]
entry_block = func.regions[0].blocks[0]

# segmentation is the 3rd operation in the entry block
seg = entry_block.operations[3]
print(f"op name : {seg.name}")

# get result
print(f"result type: {seg.result.type}")

# get attributes
print(f"attr input_zp type: {seg.attributes['input_zp'].type}")
print(f"attr input_zp value: {seg.attributes['input_zp'].value}")

# get operands
print(f"seg operands number: {len(seg.operands)}")
for operand in seg.operands:
  print(f"seg operand: {operand} {operand.type}")

print(f"seg input: {seg.operands[3]}")

seg_n = entry_block.operations[3]

print(type(seg_n))
print(type(seg_n.operation))
