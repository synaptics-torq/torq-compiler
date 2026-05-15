// TORQ_ALLOWED_WRONG: 0.07
module attributes {torch.debug_module_name = "CosModule"} {
  func.func @main(%arg0: !torch.vtensor<[1024,32],bf16>) -> !torch.vtensor<[1024,32],bf16> {
    %0 = torch.aten.cos %arg0 : !torch.vtensor<[1024,32],bf16> -> !torch.vtensor<[1024,32],bf16>
    return %0 : !torch.vtensor<[1024,32],bf16>
  }
}
