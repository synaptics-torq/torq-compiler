# Supported Operators

This document provides a comprehensive list of operators supported by the Synaptics backend within the IREE compiler stack. It details which operations are implemented and verified across various input/output types, data layouts, and tiling strategies. The table serves as a reference for users to understand backend coverage and integration status for TOSA and Linalg-based models targeting Synaptics NPU.

## Activation related Operators

| Op                | Input  |        TorqHL        | in_t | weight_t | out_t | Tiling  | Implemented | Verified | Notes                                        |
|:------------------|:------:|:---------------------:|:----:|:--------:|:-----:|:-------:|:-----------:|:--------:|:---------------------------------------------|
| Abs               |  Tosa  |       Act(ABS)        | bf16 |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | f32  |   n/a    |  f32  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |
| Cast              |  Tosa  |       Act(ACT)        | bf16 |   n/a    |  f32  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(F2I)        | bf16 |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(F2I)        | bf16 |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(F2I)        | bf16 |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(ACT)        | f32  |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(F2I)        | f32  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(F2I)        | f32  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(F2I)        | f32  |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2F)        | i16  |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2F)        | i16  |   n/a    |  f32  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(ACT)        | i16  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(ACT)        | i16  |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2F)        | i32  |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2F)        | i32  |   n/a    |  f32  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2F)        | i32  |   n/a    |  f32  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2I)        | i32  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2I)        | i32  |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2F)        |  i8  |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2F)        |  i8  |   n/a    |  f32  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2I)        |  i8  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2I)        |  i8  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |       Act(I2I)        |  i1  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
| Ceil              |  Tosa  |       Act(CEL)        | bf16 |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | f32  |   n/a    |  f32  | generic |     Yes     |   Yes    |                                              |
| Clamp             | Linalg |       Act(CLP)        | bf16 |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | f32  |   n/a    |  f32  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |
| Clz               |  Clz   |       Act(CLZ)        | i32  |   n/a    | int32 | generic |     Yes     |   Yes    |                                              |
| Floor             |  Tosa  |       Act(FLR)        | bf16 |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | f32  |   n/a    |  f32  | generic |     Yes     |   Yes    |                                              |
| Negate            |  Tosa  |       Act(NEG)        | bf16 |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | f32  |   n/a    |  f32  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |


## Element-Wise Operators

| Op                |   Input    |       TorqHL       | in_t | weight_t | out_t | Tiling  | Implemented | Verified | Notes |
|:------------------|:----------:|:------------------:|:----:|:--------:|:-----:|:-------:|:-----------:|:--------:|:------|
| ElementWiseBinary |   Linalg   |  EltwiseBin(BAND)  |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |       |
|                   |            |                    | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |       |
|                   |            |  EltwiseBin(BOR)   |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |       |
|                   |            |                    | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |       |
|                   |            |  EltwiseBin(BXOR)  |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |       |
|                   |            |                    | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |       |
|                   |            |  EltwiseBin(AND)   | bool |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |   EltwiseBin(OR)   | bool |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |  EltwiseBin(XOR)   | bool |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |   EltwiseBin(EQ)   |  i8  |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i16  |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i32  |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |                    | bf16 |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |   EltwiseBin(GT)   |  i8  |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i16  |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i32  |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |                    | bf16 |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   | Linalg ult |   EltwiseBin(GT)   | i16  |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   | Linalg ule |   EltwiseBin(GT)   | i16  |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |  EltwiseBin(GTEQ)  |  i8  |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i16  |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i32  |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |                    | bf16 |   n/a    | bool  | generic |     Yes     |   Yes    |       |
|                   |            |  EltwiseBin(MAX)   |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |       |
|                   |            |                    | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |       |
|                   |            |                    | bf16 |   n/a    | bf16  | generic |     Yes     |   Yes    |       |
|                   |            |   EltwisBin(MIN)   |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |       |
|                   |            |                    | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |       |
|                   |            |                    | bf16 |   n/a    | bf16  | generic |     Yes     |   Yes    |       |
| ElementWiseUnary  |   Linalg   | EltwiseUnary(BNOT) |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |       |
|                   |            |                    | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |       |
|                   |            |                    | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |       |
|                   |            | EltwiseUnary(NOT)  | bool |   n/a    | bool  | generic |     Yes     |   Yes    |       |


## Other Operators

| Op                | Input  |        TorqHL        | in_t | weight_t | out_t | Tiling  | Implemented | Verified | Notes                                        |
|:------------------|:------:|:---------------------:|:----:|:--------:|:-----:|:-------:|:-----------:|:--------:|:---------------------------------------------|
| Add               | Linalg |          Add          |  i8  |   n/a    |  i8   | generic |   Yes       |   Yes    |                                              |
|                   |        |                       |  i16 |   n/a    |  i16  | generic |   Yes       |   Yes    |                                              |
|                   |        |                       |  i32 |   n/a    |  i32  | generic |   Yes       |   Yes    |                                              |
|                   |        |                       | bf16 |   n/a    |  bf16 | generic |   Yes       |   Yes    |                                              |
| Bitcast           | Linalg |       Identity        | i16  |   n/a    |  bf16 |   No    |     Yes     |   Yes    |                                              |
|                   |        |                       | bf16 |   n/a    |  i16  |   No    |     Yes     |   Yes    |                                              |
| Sub               | Linalg |          Add          |  i8  |   n/a    |  i8   | generic |   Yes       |   Yes    |                                              |
|                   |        |                       |  i16 |   n/a    |  i16  | generic |   Yes       |   Yes    |                                              |
|                   |        |                       |  i32 |   n/a    |  i32  | generic |   Yes       |   Yes    |                                              |
|                   |        |                       | bf16 |   n/a    |  bf16 | generic |   Yes       |   Yes    |                                              |
| ArgMax            |  TOSA  |        ArgMax         |  i8  |    i8    |  i8   |   No    |     Yes     |   Yes    |                                              |
| AvgPool           |  TOSA  |        AvgPool        |  i8  |    i8    |  i8   |   No    |     Yes     |   Yes    |                                              |
| Conv2D            |  TOSA  |        Conv2D         |  i8  |    i8    |  i8   | custom  |     Yes     |  Partly  | 7x7 stride-2 not supported                   |
|                   |  TOSA  |        Conv2D         | i16  |    i8    |  i32  | custom  |     Yes     |  Partly  | TOSA specifies i48 output                    |
|                   |  TOSA  |        Conv2D         | bf16 |   bf16   | bf16  | custom  |   ongoing   |  Partly  |                                              |
| DepthwiseConv2D   |  TOSA  |          DW           |  i8  |    i8    |  i8   | custom  |     Yes     |  Partly  |                                              |
| Fill              | Linalg |         Fill          | any  |   n/a    |  any  | generic |     Yes     |   Yes    |                                              |
| FullyConnected    | Linalg |          FC           |  i8  |    i8    |  i8   | custom  |     Yes     |   Yes    |                                              |
| Gather            |  TOSA  |        Gather         |  i8  |    i8    |  i8   |   No    |     Yes     |   Yes    |                                              |
| Identity          |  TOSA  |       Identity        | any  |   n/a    |  any  |   No    |     Yes     |   Yes    |                                              |
| MatMul            | Linalg |        MatMul         |  i8  |    i8    |  i32  | generic |     Yes     |   Yes    | BatchMatmul, DotOp and Matvec also supported |
|                   | Linalg |        MatMul         | i16  |   i16    |  i32  | generic |     Yes     |   Yes    | TOSA specifies i48 output                    |
|                   | Linalg |        MatMul         | bf16 |   bf16   |  f32  | generic |     Yes     |   Yes    |                                              |
| MaxPool2D         | Linalg |       MaxPool2D       |  i8  |    i8    |  i8   | custom  |     Yes     |   Yes    |                                              |
|                   | Linalg |       MaxPool2D       |  i16 |    i8    |  i16  | custom  |     Yes     |   Yes    |                                              |
| Mul               | Linalg |          Mul          |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |
|                   | Linalg |          Mul          | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   | Linalg |          Mul          | bf16 |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
| Reciprocal        | Linalg |      Composite        | bf16 |   n/a    |  bf16 | generic |     Yes     |   Yes    |                                              |
| Reduce            | Linalg |      Reduce(all)      |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |      Reduce(any)      |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |      Reduce(Max)      |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | bf16 |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
|                   |        |      Reduce(Min)      |  i8  |   n/a    |  i8   | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | bf16 |   n/a    | bf16  | generic |     Yes     |   Yes    |                                              |
|                   |        |      Reduce(Sum)      | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   |        |      Reduce(Mul)      | bf16 |   n/a    |  bf16 | generic |     Yes     |   Yes    |                                              |
|                   |        |      Reduce(XOR)      | i16  |   n/a    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   |        |                       | i32  |   n/a    |  i32  | generic |     Yes     |   Yes    |                                              |
| Reshape-collapse  | Linalg |       Identity        |  i8  |   n/a    |  i8   |   No    |     Yes     |   Yes    |                                              |
| Reshape-expand    | Linalg |       Identity        |  i8  |   n/a    |  i8   |   No    |     Yes     |   Yes    |                                              |
| Rescale           | Linalg |          FMA          |  i8  |   i16    |  i16  | generic |     Yes     |   Yes    |                                              |
|                   | Linalg |          FMA          |  i8  |   i16    |  i32  | generic |     Yes     |   Yes    |                                              |
|                   | Linalg |          FMA          | i16  |   i16    |  i8   | generic |     Yes     |   Yes    |                                              |
|                   | Linalg |          FMA          | i32  |   i8     |  i8   | generic |     Yes     |   Yes    |                                              |
|                   | Linalg |          FMA          | i32  |   i8     |  i16  | generic |     Yes     |   Yes    |                                              |
|                   | Linalg |          FMA          | i32  |   i8     |  i32  | generic |     Yes     |   Yes    |                                              |
|                   | Linalg |          FMA          | i16  |   i16    |  i32  | generic |     Yes     |   Yes    |                                              |
| Resize            |  TOSA  | ResizeNearestNeighbor |  i8  |   i16    |  i8   |   No    |     Yes     |  Partly  | block size of 2                              |
| Scale-and-Add     | Linalg |          Add          |  i8  |   i16    |  i8   | custom  |     Yes     |   Yes    |                                              |
|                   |        |          Add          |  i16 |   i16    |  i16  | custom  |     Yes     |   Yes    |                                              |
| Scale-and-Sub     | Linalg |          Add          |  i8  |   i16    |  i8   | custom  |     Yes     |   Yes    |                                              |
|                   |        |          Add          |  i16 |   i16    |  i16  | custom  |     Yes     |   Yes    |                                              |
| Scatter           |  TOSA  |        SynpHL         |  i8  |    i8    |  i8   |   No    |     Yes     |   Yes    |                                              |
| Table             |  TOSA  |         Table         |  i8  |   n/a    |  i8   |   No    |     Yes     |   Yes    | Default is Gather based                      |
| Table             |  TOSA  |         Table         | i16  |   n/a    |  i32  |   No    |     Yes     |   Yes    |                                              |
| Transpose         | Linalg |       Transpose       |  i8  |    i8    |  i8   |   No    |     Yes     |   Yes    |                                              |
| Broadcast         | Linalg |       Broadcast       |  *   |   n/a    |  *    | generic |     Yes     |   Yes    | *: i8/i16/i32 supported, bf16 not yet        |
