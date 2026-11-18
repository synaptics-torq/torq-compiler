"""IR checks of the pattern that gives a stride of (1, 1) to a depthwise conv when each strided
axis has one output (synaptics-torq/torq-compiler-dev#1789): which convs it changes, which convs
it keeps, the image without the elements that the conv does not read, and the indexing maps of
the new conv. tests/ops has the numerical tests.
"""

import re

import pytest

from .conv_stride_pattern_ir import conv_ir, pads, run_pass, slices


def depthwise_strides(output):
    """The strides of each linalg.depthwise_conv_2d_nchw_chw in `output`."""
    return re.findall(
        r"linalg\.depthwise_conv_2d_nchw_chw .*strides = dense<([\d, \[\]]+)>", output
    )


@pytest.mark.ci
@pytest.mark.parametrize("size,kernel,strides,pad_h,pad_w", [
    ((32, 1), (1, 3), (1, 2), 0, 1),  # the NNR 301 layers: a W of 1 with a "same" pad
    # The depthwise lowering also takes the next two convs with their strides.
    ((1, 32), (3, 1), (2, 1), 1, 0),
    ((3, 3), (3, 3), (2, 2), 0, 0),
])
def test_stride_of_one_output_axis_becomes_1(torq_compiler, size, kernel, strides, pad_h, pad_w):
    source = conv_ir("depthwise", kernel, pad_h, pad_w, size=size, strides=strides)
    output = run_pass(torq_compiler, source)
    assert depthwise_strides(output) == ["1"]
    assert not slices(output)


@pytest.mark.ci
def test_strides_with_more_than_one_output_are_kept(torq_compiler):
    """The H axis has more than one output, so the conv keeps the stride (2, 2) that its lowering
    takes."""
    source = conv_ir("depthwise", (3, 3), pad_h=1, pad_w=1, size=(32, 1))
    assert depthwise_strides(run_pass(torq_compiler, source)) == ["2"]


@pytest.mark.ci
@pytest.mark.parametrize("strides,pad_w", [
    ((1, 4), (3, 0)),  # the low border is as large as the window
    ((1, 8), (4, 0)),  # the low border is larger than the window
])
def test_conv_that_reads_only_the_low_border_is_kept(torq_compiler, strides, pad_w):
    """Output 0 reads only the low border. The image without its unread elements would be a slice
    of the padded image, which the conv lowering does not fold."""
    source = conv_ir("depthwise", (1, 3), pad_w=pad_w, size=(32, 2), strides=strides)
    assert depthwise_strides(run_pass(torq_compiler, source)) == [f"[{strides[0]}, {strides[1]}]"]


@pytest.mark.ci
@pytest.mark.parametrize("width,strides,pad_w,image_pads,image_slices", [
    # The high border has the unread element: the conv reads the image.
    (3, (1, 2), (0, 1), [], []),
    # The high border has the two unread elements: the low border stays.
    (2, (1, 4), (1, 2), [("%arg0", "0, 0, 0, 1", "0, 0, 0, 0")], []),
    # No border: the image loses the unread element.
    (4, (1, 2), 0, [], [("%arg0", "1, 8, 32, 3", "1, 1, 1, 1")]),
    # The high border has one of the three unread elements: the image loses the other two.
    (4, (1, 4), (1, 1), [("%extracted_slice", "0, 0, 0, 1", "0, 0, 0, 0")],
     [("%arg0", "1, 8, 32, 2", "1, 1, 1, 1")]),
])
def test_unread_end_of_the_image_is_removed(torq_compiler, width, strides, pad_w, image_pads,
                                            image_slices):
    """A stride-1 conv has one output only when the image ends at the last element that this
    output reads. The border loses the unread elements first, and stays a tensor.pad."""
    source = conv_ir("depthwise", (1, 3), pad_w=pad_w, size=(32, width), strides=strides)
    output = run_pass(torq_compiler, source)
    assert depthwise_strides(output) == ["1"]
    assert pads(output) == image_pads
    assert slices(output) == image_slices


@pytest.mark.ci
def test_indexing_maps_have_the_new_strides(torq_compiler):
    """A linalg named op keeps its indexing maps in an attribute after the first read. A change
    of the strides in place leaves the maps of the original strides on the conv."""
    source = conv_ir("depthwise", (1, 3), pad_w=1, size=(32, 1), strides=(1, 2))
    output = run_pass(torq_compiler, source, then=("linalg-generalize-named-ops",))
    image_maps = re.findall(r"affine_map<\(d0, d1, d2, d3, d4, d5\) -> \((d0, .*d5)\)>", output)
    assert image_maps == ["d0, d3, d1 + d4, d2 + d5"]
