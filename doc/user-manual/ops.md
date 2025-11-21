# Supported Operators

This document provides a comprehensive list of operators supported by the Synaptics backend within the IREE compiler stack. It details which operations are implemented and verified across various input/output types, data layouts, and tiling strategies. The table serves as a reference for users to understand backend coverage and integration status for TOSA and Linalg-based models targeting Synaptics NPU.

## Activation related Operators

| Op     | Input  |  TorqHL  | in_t | weight_t | out_t | Tiling  | Notes |
|:-------|:------:|:--------:|:----:|:--------:|:-----:|:-------:|:------|
| Abs    |  Tosa  | Act(ABS) | bf16 |   n/a    | bf16  | generic |       |
|        |        |          | f32  |   n/a    |  f32  | generic |       |
|        |        |          | i16  |   n/a    |  i16  | generic |       |
|        |        |          | i32  |   n/a    |  i32  | generic |       |
|        |        |          |  i8  |   n/a    |  i8   | generic |       |
| Cast   |  Tosa  | Act(ACT) | bf16 |   n/a    |  f32  | generic |       |
|        |        | Act(F2I) | bf16 |   n/a    |  i16  | generic |       |
|        |        | Act(F2I) | bf16 |   n/a    |  i32  | generic |       |
|        |        | Act(F2I) | bf16 |   n/a    |  i8   | generic |       |
|        |        | Act(ACT) | f32  |   n/a    | bf16  | generic |       |
|        |        | Act(F2I) | f32  |   n/a    |  i16  | generic |       |
|        |        | Act(F2I) | f32  |   n/a    |  i32  | generic |       |
|        |        | Act(F2I) | f32  |   n/a    |  i8   | generic |       |
|        |        | Act(I2F) | i16  |   n/a    | bf16  | generic |       |
|        |        | Act(I2F) | i16  |   n/a    |  f32  | generic |       |
|        |        | Act(ACT) | i16  |   n/a    |  i32  | generic |       |
|        |        | Act(ACT) | i16  |   n/a    |  i8   | generic |       |
|        |        | Act(I2F) | i32  |   n/a    | bf16  | generic |       |
|        |        | Act(I2F) | i32  |   n/a    |  f32  | generic |       |
|        |        | Act(I2F) | i32  |   n/a    |  f32  | generic |       |
|        |        | Act(I2I) | i32  |   n/a    |  i16  | generic |       |
|        |        | Act(I2I) | i32  |   n/a    |  i8   | generic |       |
|        |        | Act(I2F) |  i8  |   n/a    | bf16  | generic |       |
|        |        | Act(I2F) |  i8  |   n/a    |  f32  | generic |       |
|        |        | Act(I2I) |  i8  |   n/a    |  i16  | generic |       |
|        |        | Act(I2I) |  i8  |   n/a    |  i32  | generic |       |
|        |        | Act(I2I) |  i1  |   n/a    |  i16  | generic |       |
| Ceil   |  Tosa  | Act(CEL) | bf16 |   n/a    | bf16  | generic |       |
|        |        |          | f32  |   n/a    |  f32  | generic |       |
| Clamp  | Linalg | Act(CLP) | bf16 |   n/a    | bf16  | generic |       |
|        |        |          | f32  |   n/a    |  f32  | generic |       |
|        |        |          | i16  |   n/a    |  i16  | generic |       |
|        |        |          | i32  |   n/a    |  i32  | generic |       |
|        |        |          |  i8  |   n/a    |  i8   | generic |       |
| Clz    |  Clz   | Act(CLZ) | i32  |   n/a    | int32 | generic |       |
| Floor  |  Tosa  | Act(FLR) | bf16 |   n/a    | bf16  | generic |       |
|        |        |          | f32  |   n/a    |  f32  | generic |       |
| Negate |  Tosa  | Act(NEG) | bf16 |   n/a    | bf16  | generic |       |
|        |        |          | f32  |   n/a    |  f32  | generic |       |
|        |        |          | i16  |   n/a    |  i16  | generic |       |
|        |        |          | i32  |   n/a    |  i32  | generic |       |
|        |        |          |  i8  |   n/a    |  i8   | generic |       |


## Element-Wise Operators

| Op                | Input |       TorqHL       | in_t | weight_t | out_t | Tiling  | Notes |
|:------------------|:-----:|:------------------:|:----:|:--------:|:-----:|:-------:|:-----:|
| ElementWiseBinary | Arith |  EltwiseBin(BAND)  |  i8  |   n/a    |  i8   | generic |       |
|                   |       |                    | i16  |   n/a    |  i16  | generic |       |
|                   |       |                    | i32  |   n/a    |  i32  | generic |       |
|                   |       |  EltwiseBin(BOR)   |  i8  |   n/a    |  i8   | generic |       |
|                   |       |                    | i16  |   n/a    |  i16  | generic |       |
|                   |       |                    | i32  |   n/a    |  i32  | generic |       |
|                   |       |  EltwiseBin(BXOR)  |  i8  |   n/a    |  i8   | generic |       |
|                   |       |                    | i16  |   n/a    |  i16  | generic |       |
|                   |       |                    | i32  |   n/a    |  i32  | generic |       |
|                   |       |  EltwiseBin(AND)   | bool |   n/a    | bool  | generic |       |
|                   |       |   EltwiseBin(OR)   | bool |   n/a    | bool  | generic |       |
|                   |       |  EltwiseBin(XOR)   | bool |   n/a    | bool  | generic |       |
|                   |       |   EltwiseBin(EQ)   |  i8  |   n/a    | bool  | generic |       |
|                   |       |                    | i16  |   n/a    | bool  | generic |       |
|                   |       |                    | i32  |   n/a    | bool  | generic |       |
|                   |       |                    | bf16 |   n/a    | bool  | generic |       |
|                   |       |   EltwiseBin(GT)   |  i8  |   n/a    | bool  | generic |       |
|                   |       |                    | i16  |   n/a    | bool  | generic |       |
|                   |       |                    | i32  |   n/a    | bool  | generic |       |
|                   |       |                    | bf16 |   n/a    | bool  | generic |       |
|                   |       |  EltwiseBin(GTEQ)  |  i8  |   n/a    | bool  | generic |       |
|                   |       |                    | i16  |   n/a    | bool  | generic |       |
|                   |       |                    | i32  |   n/a    | bool  | generic |       |
|                   |       |                    | bf16 |   n/a    | bool  | generic |       |
|                   |       |   EltwiseBin(LT)   |  i8  |   n/a    | bool  | generic |       |
|                   |       |                    | i16  |   n/a    | bool  | generic |       |
|                   |       |                    | i32  |   n/a    | bool  | generic |       |
|                   |       |                    | bf16 |   n/a    | bool  | generic |       |
|                   |       |  EltwiseBin(LTEQ)  |  i8  |   n/a    | bool  | generic |       |
|                   |       |                    | i16  |   n/a    | bool  | generic |       |
|                   |       |                    | i32  |   n/a    | bool  | generic |       |
|                   |       |  EltwiseBin(MAX)   |  i8  |   n/a    |  i8   | generic |       |
|                   |       |                    | i16  |   n/a    |  i16  | generic |       |
|                   |       |                    | i32  |   n/a    |  i32  | generic |       |
|                   |       |                    | bf16 |   n/a    | bf16  | generic |       |
|                   |       |   EltwisBin(MIN)   |  i8  |   n/a    |  i8   | generic |       |
|                   |       |                    | i16  |   n/a    |  i16  | generic |       |
|                   |       |                    | i32  |   n/a    |  i32  | generic |       |
|                   |       |                    | bf16 |   n/a    | bf16  | generic |       |
| ElementWiseUnary  | Arith | EltwiseUnary(BNOT) |  i8  |   n/a    |  i8   | generic |       |
|                   |       |                    | i16  |   n/a    |  i16  | generic |       |
|                   |       |                    | i32  |   n/a    |  i32  | generic |       |
|                   |       | EltwiseUnary(NOT)  | bool |   n/a    | bool  | generic |       |

## Composite Operators

| Op                   | Input  | TorqHL | in_t | weight_t | out_t | Tiling  | Notes |
|:---------------------|:------:|:------:|:----:|:--------:|:-----:|:-------:|:-----:|
| Reciprocal           | Linalg |        | bf16 |   n/a    | bf16  | generic |       |
| Div                  | Arith  |        | bf16 |   n/a    | bf16  | generic |       |
| Erf                  |  Math  |        | bf16 |   n/a    | bf16  | generic |       |
| QuantizedBatchMatMul | Linalg |        |  i8  |   n/a    |  i32  | generic |       |

## Other Operators

| Op               | Input  |        TorqHL         | in_t | weight_t | out_t | Tiling  | Notes                            |
|:-----------------|:------:|:---------------------:|:----:|:--------:|:-----:|:-------:|:---------------------------------|
| Add              | Linalg |          Add          |  i8  |   n/a    |  i8   | generic |                                  |
|                  |        |                       | i16  |   n/a    |  i16  | generic |                                  |
|                  |        |                       | i32  |   n/a    |  i32  | generic |                                  |
|                  |        |                       | bf16 |   n/a    | bf16  | generic |                                  |
| Bitcast          | Arith  |       Identity        | i16  |   n/a    | bf16  |   No    |                                  |
|                  |        |                       | bf16 |   n/a    |  i16  |   No    |                                  |
| Sub              | Linalg |          Add          |  i8  |   n/a    |  i8   | generic |                                  |
|                  |        |                       | i16  |   n/a    |  i16  | generic |                                  |
|                  |        |                       | i32  |   n/a    |  i32  | generic |                                  |
|                  |        |                       | bf16 |   n/a    | bf16  | generic |                                  |
| ArgMax           |  TOSA  |        ArgMax         |  i8  |    i8    |  i8   |   No    |                                  |
| AvgPool          |  TOSA  |        AvgPool        |  i8  |    i8    |  i8   |   No    |                                  |
| Conv1D           | Linalg |        Conv1D         | bf16 |   bf16   | bf16  | custom  |                                  |
| Conv2D           | Linalg |        Conv2D         |  i8  |    i8    |  i8   | custom  |                                  |
|                  | Linalg |        Conv2D         | i16  |    i8    |  i32  | custom  | TOSA specifies i48 output        |
|                  | Linalg |        Conv2D         | bf16 |   bf16   | bf16  | custom  |                                  |
| DepthwiseConv2D  |  TOSA  |          DW           |  i8  |    i8    |  i8   | custom  |                                  |
| Fill             | Linalg |         Fill          | any  |   n/a    |  any  | generic |                                  |
| FullyConnected   | Linalg |          FC           |  i8  |    i8    |  i8   | custom  |                                  |
| Gather           |  TOSA  |        Gather         |  i8  |    i8    |  i8   |   No    |                                  |
|                  |        |        Gather         | i16  |   i16    |  i16  |   No    |                                  |
|                  |        |        Gather         | i32  |   i32    |  i32  |   No    |                                  |
|                  | Linalg |        Gather         | i16  |   i16    |  i16  |   No    |                                  |
| Identity         |  TOSA  |       Identity        | any  |   n/a    |  any  |   No    |                                  |
| MatMul           | Linalg |        MatMul         |  i8  |    i8    |  i32  | generic |                                  |
|                  | Linalg |        MatMul         | i16  |   i16    |  i32  | generic | TOSA specifies i48 output        |
|                  | Linalg |        MatMul         | bf16 |   bf16   |  f32  | generic |                                  |
| Dot              | Linalg |        MatMul         |  i8  |    i8    |  i16  | generic |                                  |
|                  | Linalg |        MatMul         | i16  |   i16    |  i16  | generic |                                  |
|                  | Linalg |        MatMul         | bf16 |   bf16   |  f32  | generic |                                  |
| MatVec           | Linalg |        MatMul         |  i8  |    i8    |  i16  | generic |                                  |
|                  | Linalg |        MatMul         | i16  |   i16    |  i16  | generic |                                  |
|                  | Linalg |        MatMul         | bf16 |   bf16   |  f32  | generic |                                  |
| BatchMatMul      | Linalg |        MatMul         |  i8  |    i8    |  i32  | generic |                                  |
| MaxPool2D        | Linalg |       MaxPool2D       |  i8  |    i8    |  i8   | custom  |                                  |
|                  | Linalg |       MaxPool2D       | i16  |    i8    |  i16  | custom  |                                  |
| Mul              | Linalg |          Mul          |  i8  |   n/a    |  i8   | generic |                                  |
|                  | Linalg |          Mul          | i16  |   n/a    |  i16  | generic |                                  |
|                  | Linalg |          Mul          | bf16 |   n/a    | bf16  | generic |                                  |
| Reduce           | Linalg |      Reduce(all)      |  i8  |   n/a    |  i8   | generic |                                  |
|                  |        |                       | i16  |   n/a    |  i16  | generic |                                  |
|                  |        |                       | i32  |   n/a    |  i32  | generic |                                  |
|                  |        |      Reduce(any)      |  i8  |   n/a    |  i8   | generic |                                  |
|                  |        |                       | i16  |   n/a    |  i16  | generic |                                  |
|                  |        |                       | i32  |   n/a    |  i32  | generic |                                  |
|                  |        |      Reduce(Max)      |  i8  |   n/a    |  i8   | generic |                                  |
|                  |        |                       | i16  |   n/a    |  i16  | generic |                                  |
|                  |        |                       | i32  |   n/a    |  i32  | generic |                                  |
|                  |        |                       | bf16 |   n/a    | bf16  | generic |                                  |
|                  |        |      Reduce(Min)      |  i8  |   n/a    |  i8   | generic |                                  |
|                  |        |                       | i16  |   n/a    |  i16  | generic |                                  |
|                  |        |                       | i32  |   n/a    |  i32  | generic |                                  |
|                  |        |                       | bf16 |   n/a    | bf16  | generic |                                  |
|                  |        |      Reduce(Sum)      | i32  |   n/a    |  i32  | generic |                                  |
|                  |        |      Reduce(Mul)      | bf16 |   n/a    | bf16  | generic |                                  |
|                  |        |      Reduce(XOR)      | i16  |   n/a    |  i16  | generic |                                  |
|                  |        |                       | i32  |   n/a    |  i32  | generic |                                  |
| Reshape-collapse | Linalg |       Identity        |  i8  |   n/a    |  i8   |   No    |                                  |
| Reshape-expand   | Linalg |       Identity        |  i8  |   n/a    |  i8   |   No    |                                  |
| Rescale          | Linalg |          FMA          | ui8  |   ui8    |  i8   | generic |                                  |
|                  | Linalg |          FMA          |  i8  |    i8    |  ui8  | generic |                                  |
|                  | Linalg |          FMA          |  i8  |   i16    |  i16  | generic |                                  |
|                  | Linalg |          FMA          |  i8  |   i16    |  i32  | generic |                                  |
|                  | Linalg |          FMA          | i16  |   i16    |  i8   | generic |                                  |
|                  | Linalg |          FMA          | i32  |    i8    |  i8   | generic |                                  |
|                  | Linalg |          FMA          | i32  |    i8    |  i16  | generic |                                  |
|                  | Linalg |          FMA          | i32  |    i8    |  i32  | generic |                                  |
|                  | Linalg |          FMA          | i16  |   i16    |  i32  | generic |                                  |
| Resize           |  TOSA  | ResizeNearestNeighbor |  i8  |   i16    |  i8   |   No    | block size of 2                  |
| Scale-and-Add    | Linalg |          Add          |  i8  |   i16    |  i8   | custom  |                                  |
|                  |        |          Add          | i16  |   i16    |  i16  | custom  |                                  |
| Scale-and-Sub    | Linalg |          Add          |  i8  |   i16    |  i8   | custom  |                                  |
|                  |        |          Add          | i16  |   i16    |  i16  | custom  |                                  |
| Scatter          |  TOSA  |        SynpHL         |  i8  |    i8    |  i8   |   No    |                                  |
| Table            |  TOSA  |         Table         |  i8  |   n/a    |  i8   |   No    | Default is Table based           |
| Table            |  TOSA  |         Table         | i16  |   n/a    |  i32  |   No    |                                  |
| Transpose        | Linalg |       Transpose       |  i8  |    i8    |  i8   |   No    |                                  |
| Broadcast        | Linalg |       Broadcast       |  *   |   n/a    |   *   | generic | *: i8/i16/i32/bf16/f32 supported |
