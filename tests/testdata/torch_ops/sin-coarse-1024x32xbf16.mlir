// TORQ_ALLOWED_WRONG: 0.06
module attributes {torch.debug_module_name = "SinModule"} {
  func.func @main(%arg0: !torch.vtensor<[1024,32],bf16>) -> !torch.vtensor<[1024,32],bf16> {
    %0 = torch.aten.sin %arg0 : !torch.vtensor<[1024,32],bf16> -> !torch.vtensor<[1024,32],bf16>
    return %0 : !torch.vtensor<[1024,32],bf16>
  }
}
