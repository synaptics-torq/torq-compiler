"""Keras test models: AveragePooling2D and MaxPooling2D.

Each case name says what it tests: pool size, strides, padding and input shape (H x W x C),
e.g. avgpool_k3x3_s3x3_valid_10x10x4.
"""

from torq.testing.keras_builder import layers, sequential

_POOLS = {"avgpool": layers.AveragePooling2D, "maxpool": layers.MaxPooling2D}

# pool, pool size (h, w), strides (h, w), padding, input (H, W, C)
_LAYERS = [
    ("avgpool", (3, 3), (3, 3), "valid", (10, 10, 4)),
    ("avgpool", (4, 4), (4, 4), "valid", (10, 10, 4)),
    ("avgpool", (5, 5), (5, 5), "valid", (10, 10, 4)),
    ("avgpool", (1, 3), (1, 3), "valid", (10, 10, 4)),
    ("avgpool", (3, 3), (3, 3), "same", (10, 10, 4)),
    ("avgpool", (3, 3), (1, 1), "same", (4, 58, 11)),
    ("avgpool", (4, 4), (4, 4), "same", (10, 10, 4)),
    ("avgpool", (5, 5), (5, 5), "same", (10, 10, 4)),
    ("avgpool", (1, 3), (1, 3), "same", (10, 10, 4)),
    ("maxpool", (3, 3), (3, 3), "valid", (10, 10, 4)),
    ("maxpool", (4, 4), (4, 4), "valid", (10, 10, 4)),
    ("maxpool", (5, 5), (5, 5), "valid", (10, 10, 4)),
    ("maxpool", (1, 3), (1, 3), "valid", (10, 10, 4)),
    ("maxpool", (3, 3), (3, 3), "same", (10, 10, 4)),
    ("maxpool", (4, 4), (4, 4), "same", (10, 10, 4)),
    ("maxpool", (5, 5), (5, 5), "same", (10, 10, 4)),
    ("maxpool", (1, 3), (1, 3), "same", (10, 10, 4)),
    ("avgpool", (2, 2), (2, 2), "valid", (10, 10, 4)),
    ("avgpool", (2, 2), (2, 2), "valid", (100, 100, 1)),
    ("avgpool", (2, 2), (2, 2), "valid", (100, 100, 4)),
    ("avgpool", (2, 2), (2, 2), "valid", (134, 158, 5)),
    # a pool over the whole frame
    ("avgpool", (25, 5), (25, 5), "valid", (25, 5, 4)),
    ("avgpool", (5, 25), (5, 25), "valid", (5, 25, 4)),
    # EMZA 70 layers
    ("avgpool", (2, 2), (2, 2), "valid", (65, 77, 10)),
    ("avgpool", (2, 2), (2, 2), "valid", (30, 36, 15)),
]

# Sliding average-pool lowering currently supports the i8 path only.
CONFIG = {
    "avgpool_k3x3_s1x1_same_4x58x11": {"quantize": "int8"},
}


def _name(pool, size, strides, padding, shape):
    h, w, c = shape
    return f"{pool}_k{size[0]}x{size[1]}_s{strides[0]}x{strides[1]}_{padding}_{h}x{w}x{c}"


CASES = {}
for _layer in _LAYERS:
    _key = _name(*_layer)
    if _key in CASES:
        raise ValueError(f"pooling: {_key} is listed twice")
    _pool, _size, _strides, _padding, _shape = _layer
    CASES[_key] = sequential(_shape, _POOLS[_pool](pool_size=_size, strides=_strides,
                                                   padding=_padding))
