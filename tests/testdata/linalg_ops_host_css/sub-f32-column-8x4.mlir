// The f32 sub runs in a host program that reads a column of %arg0
// (memref<8xf32, strided<[4], offset: 1>>). The runtime copies a strided
// argument to a dense buffer and back; before b35fdc35 it took the innermost
// stride as the element size, saw the column as contiguous and copied it flat.
#map = affine_map<(d0) -> (d0)>
module {
  func.func @main(%arg0: tensor<8x4xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>) {
    %col = tensor.extract_slice %arg0[0, 1] [8, 1] [1, 1] : tensor<8x4xf32> to tensor<8xf32>
    %0 = tensor.empty() : tensor<8xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
    ins(%col, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %2 = arith.subf %a, %b : f32
      linalg.yield %2 : f32
    } -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}
