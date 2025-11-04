## Kernel Development

Developing a kernel that runs on Torq using low-level HW APIs (e.g. NDLs) is a complicated activity.

1. Programming a vectorized architecture is intrinsically harder then normal SISD programming
   (requires correct data organization and processing to exploit parallelism) 
2. Idiosyncrasies and peculiarities in the HW
3. Working with strides, offset and counts is unnatural
4. Multiple parallel independent data flows are difficult to grasp by SW people


How can we handle this complexity?

1. Carefully design the data organization (e.g.: put data that can be processed together close to
   each other, provide adequate alignement and padding if needed). Design the processing
   to reduce the number of times the same data is copied in/out from local memory.

2. Encapsulate the specific HW behaviour inside higher-level operations
   This allows us to provide a logical "instruction set" for the NPW which abstract
   the HW details (eg. multiplyAccumulate instruction)

3. Organize the data in tensors with a shape and element type.
   Let the internal implementation take care of the corresponding sizes, strides and offset.
   (same idea as numpy).
   
4. In our Torq Slice the data flows (represented by the NDLs) are running in parallel but they
   are not completely independent, actually they must be exactly syncronized in order to
   get the correct results.
   Even if we have about 10 of these data flows they represent one single computation.
   Let's express the overall computation once and derive all the data flows from there.
   

EasyKernel is a library designed to help with points 2. to 4.
It allows to generate the low-level settings and NDLs configuration from a more natural
higher-level description.


### Example: Add kernel

Required kernel behaviour:
receive in input two dense ``int8`` tensors and generate in output a dense ``int16`` tensor
containing the rescaled elementwise sum of the inputs, clamped to a specified min and max values.. 
Each output value is computed as:

```out[i] = clamp(input1[i] * w1 + input2[i] * w2, min, max)``` 

The first thing to do when developing a kernel is to define the format of the input tensor.
In this example each input consists of a dense tensor of ``int8`` data, using EasyKernel syntax
we could represent this as a 1-dimensional tensor in LRAM:

```{code} C
int dataSize = 1024;  // Suppose 1KB input
LData input({dataSize}, DType::I8);
```

:::{note}
``LData`` indicates data residing in LRAM of the NPU. Similarly ``IData``, ``WData`` and ``BData``
indicate data residing in IRAM, WRAM and BRAM respectively.
``QData`` indicates the generated output data.
:::

Since our ALU can only perform 32 16-bits add operations in parallel, it's helpful to represent this
directly in the data organization, as this will make vectorized computation much more simple to express:

```{code} C
int blockSize = 32;
int blockCount = div_ceil(dataSize, blockSize);
LData input({dataSize, blockSize}, DType::I8);
```

:::{note}
From definition above we can see immediately that we can't process data of any size, but only
multiple of *blockSize*.
:::

A matrix containing 2 input can be represented in a similar way:

```{code} C
int blockSize = 32;
int blockCount = div_ceil(dataSize, blockSize);
int inputCount = 2;
LData input({inputCount, blockCount, blockSize}, DType::I8);
```

We don't have to represent explicitly the strides of the elements in this matrix since
these can be inferred automatically from the shape.
In our case the two inputs are not contiguous rows of a matrix, but two independent vectors in memory.
We can still represent this as a matrix with a big stride between the two rows (corresponding
to the distance of the two input vectors in memory.

```{code} C
int blockSize = 32;
int blockCount = div_ceil(dataSize, blockSize);
int inputCount = 2;
int inputStride = input2Address - input1Address;
LData input({{inputCount, inputStride}, dataSize, blockSize}, DType::I8);
```

:::{tip}
We can express non-natural strides in a data tensor when needed by specifying the stride next to the
corresponding dimension
:::

In the same way we can declare all the data tensors involved in the computation (weight, bias and output).
We have to pay attention to the output, the activation unit is not able to generate 64 elements
in one go, but only 16. Also in this case it's convenient to express this explicitly in the
data organization of the output tensor.

```{code} C
int blockSize = 32;
const uint32_t actBlockSize = 8;
int blockCount = div_ceil(dataSize, blockSize);
int inputCount = 2;
int inputStride = input2Address - input1Address;

LData input({{inputCount, inputStride}, dataSize, blockSize}, DType::I8);
LData weights({inputCount}, DType::I16);
LData biasScale({2}, DType::I32);
LData output({blockCount, blockSize / actBlockSize, actBlockSize)}, DType::I8);
```

From the definitions above we can see at a glance which data the kernel expects in input and
which data is generated in output. We can also see that we need one single 16bit weight and one single
32bit {bias,scale} pair for the entire computation.

Now that the data definition is completed we can start to define the algorithm to perform the processing.
This can be done with the help of a ``Slice`` object that will allow us to express the
computation using high-level instruction and translate them to the corresponding HW and NDL configuration.

The SyNAPU Slice is not able to use data from LRAM directly, we have first to bring them to one of the
internal memories (IRAM, WRAM or BRAM). Since in our case the weight and bias tensors are so small
we can simply bring them to internal memory once at the beginning and leave them there.

```{code} C
Slice slice;
WData wdata = slice.wram.load(weights);
BData bdata = slice.bram.load(biasScale);
```

Now we can start specifying the processing. The idea is to perform the same computation for all the
data blocks, one data block at a time. We need a for loop:

```{code} C
Slice slice;
WData wdata = slice.wram.load(weights);
BData bdata = slice.bram.load(biasScale);

For(auto b = slice.iterate(blockCount)) {
    // Body of the processing to be added here
}
```

We can see ``iterate(N)`` as a method that creates an *iterator* representing a loop that will be repeated N times.
This is just a syntactic representation of a loop that will be executed by the Slice, it is *not* 
a real C++ loop. Each iterator internally contains an *iteration variable* that represents a runtime
index incrementing from 0 to N-1. We can use these iteration variables as indexes to extract parts
of the data tensors.

To add the two tensors we need to add the two inputs for each block , so we need anoter loop to
bring the two inputs in memory, one block at a time:

```{code} C
LData input({{inputCount, inputStride}, dataSize, blockSize}, DType::I8);
LData weights({inputCount}, DType::I16);
LData biasScale({2}, DType::I32);
LData output({blockCount, blockSize / actBlockSize, actBlockSize)}, DType::I8);

Slice slice;
WData wdata = slice.wram.load(weights);
BData bdata = slice.bram.load(biasScale);

For(auto b = slice.iterate(blockCount)) {
  For(auto i = slice.iterate(inputCount)) {
    // Load the block 'b' of input 'i' in IRAM
    IData idata = slice.iram.load(input[i][b]);
    // process idata
  }
}
```

To do the actual rescaling and addition we can multiply all 32 elements of each block by the weight
for the corresponding input, and accumulate the result in the PRAM.
The ALU has a ``scalarProductAccumulate()`` instruction that does exactly this.

```{code} C
LData input({{inputCount, inputStride}, dataSize, blockSize}, DType::I8);
LData weights({inputCount}, DType::I16);
LData biasScale({2}, DType::I32);
LData output({blockCount, blockSize / actBlockSize, actBlockSize)}, DType::I8);

Slice slice;
WData wdata = slice.wram.load(weights);
BData bdata = slice.bram.load(biasScale);

For(auto b = slice.iterate(blockCount)) {
  PData pdata;
  For(auto i = slice.iterate(inputCount)) {
    IData idata = slice.iram.load(input[i][b]);
    // Perform a 32x1 vectorized multiply and accumulate the result in PRAM
    pdata = slice.alu.scalarProductAccumulate(idata, wdata[i]);
  }
}
```

We are almost done, we can't store this result directly in the LRAM, we have to send it through
the activation unit which also performs any required clamp and rescaling if needed.
The activation unit is not able to process 32 values in one single go, but only 8, so we need
an additional loop to iterate through the values in the partials.

```{code} C
LData input({{inputCount, inputStride}, dataSize, blockSize}, DType::I8);
LData weights({inputCount}, DType::I16);
LData biasScale({2}, DType::I32);
LData output({blockCount, blockSize / actBlockSize, actBlockSize)}, DType::I8);

Slice slice;
WData wdata = slice.wram.load(weights);
BData bdata = slice.bram.load(biasScale);

For(auto b = slice.iterate(blockCount)) {
  PData pdata;
  For(auto i = slice.iterate(inputCount)) {
    // Perform a 32x1 vectorized multiply and accumulate the result in PRAM
    IData idata = slice.iram.load(input[i][b]);
    pdata = slice.alu.scalarProductAccumulate(idata, wdata[i]);
  }
  // Clamp the partials in the PRAM and store them in LRAM
  For(auto a = slice.iterate(blockSize / actBlockSize)) {
      QData res = slice.act.rescaleClamp(pdata[a], bdata, shift, min, max, zp);
      slice.store(output[b][a], res);
  }
}
```

This completes the definition of the kernel.
We can now ask the `Slice` object to provide the corresponding HW settings and NDLs configuration using
the getCfgAttr() and getNdls() methods.

### Kernel Structure

The structure of the kernel in the example above is quite typical.
Each kernel must contain the following mandatory instructions:

- One single ``alu`` instruction (specify the behaviour of the ALU unit)
- One single ``act`` instruction (specify the behaviour of the activation unit)
- One single ``store`` insruction (to store the result of the computation to LRAM)

In addition a kernel normally contains a combination of the following instructions:
- ``load`` instructions to bring data from LRAM to the internal memories (up to one load instruction
  for each of IRAM, WRAM, BRAM)
- ``for`` loops to repeat the computation over all the data to be processed
- additionat configuration instructions as needed

It's important to point out the role of the ``load`` instructions, even if they are conceptually
very simple they are critical in the design of a kernel since they give the kernel writer full control
on *which* data to bring to an internal memory and *when*. Ideally load instructions should be put
in the most outer loop possible as this reduces the possibility of contention with other units on
LRAM access.