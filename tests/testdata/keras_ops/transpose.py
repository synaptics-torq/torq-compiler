"""Keras test models: Permute (TFLite TRANSPOSE)."""

from torq.testing.keras_builder import layers, sequential

CASES = {
    "model110_transpose_last2first": sequential(
        (8, 32, 48),
        layers.Permute((3, 1, 2)),
        batch_size=10,
    ),
    "model111_transpose_first2last": sequential(
        (8, 32, 48),
        layers.Permute((2, 3, 1)),
        batch_size=10,
    ),
    "model112_transpose_2d": sequential((8, 32, 48), layers.Permute((2, 1, 3)), batch_size=10),
    "model160_transpose_ChLastToFirst_1x8x10x4": sequential((8, 10, 4), layers.Permute((3, 1, 2))),
    "model161_transpose_ChFirstToLast_1x8x10x4": sequential((8, 10, 4), layers.Permute((2, 3, 1))),
    "model162_transpose_2d_matrix_1x4x6x2": sequential((4, 6, 2), layers.Permute((2, 1, 3))),
}
