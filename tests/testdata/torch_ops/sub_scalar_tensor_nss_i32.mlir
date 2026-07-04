// Regression: runtime rank-0 scalar broadcast in an elementwise sub (scalar - tensor).
// The rank-0 scalar is a real tensor operand of the linalg.generic subi. Lowering to NSS
// tripped an AddOp::getKernelEncoding assert (input1 rank 0 != result rank) because
// BroadcastElementwiseBinaryOpPattern skipped rank-0 operands ("no need broadcast"). The
// fix lets the rank-0 operand flow through broadcastInputs, materializing it to the result
// shape so the add path sees uniform full-rank operands.
// Extracted unmodified from the tsuki part_a_best model via scripts/inspect_onnx_node.py.
module {
  func.func @Sub_full_static_mask_samplep60_bool__chunk0__sub_standalone(%arg0: !torch.vtensor<[],si32>, %arg1: !torch.vtensor<[16010],si32>) -> !torch.vtensor<[16010],si32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "inspect_onnx_node.extract", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Sub"(%arg0, %arg1) : (!torch.vtensor<[],si32>, !torch.vtensor<[16010],si32>) -> !torch.vtensor<[16010],si32> 
    return %0 : !torch.vtensor<[16010],si32>
  }
}

