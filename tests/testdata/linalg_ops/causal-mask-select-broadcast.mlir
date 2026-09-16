// Causal-mask chain isolated from SmolLM2 decoder attention:
//   And(causal[1,1,8,8], pad[1,1,1,8]) -> Not -> Where(not, -inf, scores)
// The And is the non-constant i1 broadcast this branch fixes
// ('Input strides must match' pre-fix); the Not is arith.xori against a
// constant true, and the Where broadcasts the [1,1,8,8] mask over the
// [1,15,8,8] scores.
module {
  func.func @main(
      %causal: tensor<1x1x8x8xi1>,
      %pad: tensor<1x1x1x8xi1>,
      %scores: tensor<1x15x8x8xbf16>) -> tensor<1x15x8x8xbf16> {

    // And(causal, pad) -> [1,1,8,8]
    %init0 = tensor.empty() : tensor<1x1x8x8xi1>
    %and = linalg.generic {
      indexing_maps = [
        affine_map<(d0,d1,d2,d3) -> (d0,d1,d2,d3)>,
        affine_map<(d0,d1,d2,d3) -> (d0,d1,0,d3)>,
        affine_map<(d0,d1,d2,d3) -> (d0,d1,d2,d3)>
      ],
      iterator_types = ["parallel","parallel","parallel","parallel"]
    } ins(%causal, %pad : tensor<1x1x8x8xi1>, tensor<1x1x1x8xi1>) outs(%init0 : tensor<1x1x8x8xi1>) {
    ^bb0(%a: i1, %b: i1, %o: i1):
      %r = arith.andi %a, %b : i1
      linalg.yield %r : i1
    } -> tensor<1x1x8x8xi1>

    // Not(And) -> [1,1,8,8]
    %init1 = tensor.empty() : tensor<1x1x8x8xi1>
    %not = linalg.generic {
      indexing_maps = [
        affine_map<(d0,d1,d2,d3) -> (d0,d1,d2,d3)>,
        affine_map<(d0,d1,d2,d3) -> (d0,d1,d2,d3)>
      ],
      iterator_types = ["parallel","parallel","parallel","parallel"]
    } ins(%and : tensor<1x1x8x8xi1>) outs(%init1 : tensor<1x1x8x8xi1>) {
    ^bb0(%a: i1, %o: i1):
      %t = arith.constant true
      %r = arith.xori %a, %t : i1
      linalg.yield %r : i1
    } -> tensor<1x1x8x8xi1>

    // Where(Not, -inf, scores) -> [1,15,8,8]
    %neginf = arith.constant dense<0xFF80> : tensor<1x15x8x8xbf16>
    %init2 = tensor.empty() : tensor<1x15x8x8xbf16>
    %sel = linalg.generic {
      indexing_maps = [
        affine_map<(d0,d1,d2,d3) -> (d0,0,d2,d3)>,
        affine_map<(d0,d1,d2,d3) -> (d0,d1,d2,d3)>,
        affine_map<(d0,d1,d2,d3) -> (d0,d1,d2,d3)>,
        affine_map<(d0,d1,d2,d3) -> (d0,d1,d2,d3)>
      ],
      iterator_types = ["parallel","parallel","parallel","parallel"]
    } ins(%not, %neginf, %scores : tensor<1x1x8x8xi1>, tensor<1x15x8x8xbf16>, tensor<1x15x8x8xbf16>) outs(%init2 : tensor<1x15x8x8xbf16>) {
    ^bb0(%c: i1, %t: bf16, %f: bf16, %o: bf16):
      %r = arith.select %c, %t, %f : bf16
      linalg.yield %r : bf16
    } -> tensor<1x15x8x8xbf16>

    return %sel : tensor<1x15x8x8xbf16>
  }
}
