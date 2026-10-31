// A bf16 3x3 stride-2 conv from 1x64x56x56 to 128 channels (a ResNet downsampling conv),
// with a splat weight. On SR200 its 4-row tiles of the 28 output rows spill a buffer
// computed in the tile; counting that spill would cut the tile to 2 rows.
module @module {
  util.global private @__auto.constant_128_64_3_3_torch.bfloat16 = dense<1.000000e-02> : tensor<128x64x3x3xbf16>
  func.func @main(%arg0: !torch.vtensor<[1,64,56,56],bf16>) -> !torch.vtensor<[1,128,28,28],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %__auto.constant_128_64_3_3_torch.bfloat16 = util.global.load @__auto.constant_128_64_3_3_torch.bfloat16 : tensor<128x64x3x3xbf16>
    %0 = torch_c.from_builtin_tensor %__auto.constant_128_64_3_3_torch.bfloat16 : tensor<128x64x3x3xbf16> -> !torch.vtensor<[128,64,3,3],bf16>
    %none = torch.constant.none
    %int2 = torch.constant.int 2
    %int2_0 = torch.constant.int 2
    %1 = torch.prim.ListConstruct %int2, %int2_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1 = torch.constant.int 1
    %int1_1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int1, %int1_1 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_2 = torch.constant.int 1
    %int1_3 = torch.constant.int 1
    %3 = torch.prim.ListConstruct %int1_2, %int1_3 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_4 = torch.constant.int 1
    %4 = torch.aten.conv2d %arg0, %0, %none, %1, %2, %3, %int1_4 : !torch.vtensor<[1,64,56,56],bf16>, !torch.vtensor<[128,64,3,3],bf16>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,128,28,28],bf16>
    return %4 : !torch.vtensor<[1,128,28,28],bf16>
  }
}
