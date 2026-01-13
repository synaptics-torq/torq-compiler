import streamlit as st
import sys
import numpy as np


def show_difference_map(tensor_diff):
    # create numpy array with a three channels image from buffer_data_slice with color of a pixel ref if the value in the original data is not zero, otherwise white
    map_width = st.slider("Map width (pixels)", 1, tensor_diff.size, min(64, tensor_diff.size))

    tolerance = st.number_input("Tolerance", min_value=0, value=1)

    map_size = int(np.ceil(tensor_diff.size / map_width) * map_width)

    buffer_map = np.full((map_size, ), np.nan)
    buffer_map[:tensor_diff.size] = tensor_diff.flatten()    
    buffer_map = buffer_map.reshape(-1, map_width)
    buffer_map = np.expand_dims(buffer_map, axis=-1)
    
    def colorize(data):
        # colorize the data based on the value
        if np.isnan(data):
            return np.array([80, 80, 80], dtype=np.uint8)
        elif data == 0:
            return np.array([0, 255, 0], dtype=np.uint8)
        elif data < tolerance:
            return np.array([255, 128, 0], dtype=np.uint8)        
        else:
            return np.array([255, 0, 0], dtype=np.uint8)

    # apply colorize function to each pixel in the buffer_map
    buffer_map = np.apply_along_axis(colorize, -1, buffer_map)

    # upscale the image with nearest neighbor interpolation x8
    buffer_map = np.repeat(np.repeat(buffer_map, 8, axis=0), 8, axis=1)

    st.image(buffer_map, caption="Buffer difference image (red difference, white unchanged, gray outside buffer)")


st.set_page_config(page_title="Torq Buffer diff tool", layout="wide")

st.title("Torq Buffer diff tool")

st.markdown("""
            This tool is used to visualize the differences between two tensors, typically used for debugging and analysis in neural network applications. 
            It allows you to compare a source tensor with a reference tensor, highlighting differences in their values across channels.
            """)


if len(sys.argv) > 2:
    tensor1 = sys.argv[1]
    tensor2 = sys.argv[2]
    st.write(f"Comparing tensors: `{tensor1}` and `{tensor2}`")
else:
    tensor1 = st.file_uploader("Tensor 1")
    tensor2 = st.file_uploader("Tensor 2")

if tensor1 is None or tensor2 is None:
    st.stop()

# parse tensor 1 and tensor 2 as numpy files

try:
    tensor1 = np.load(tensor1)
    tensor2 = np.load(tensor2)
    if tensor1.dtype == "|V2":
        tensor1 = tensor1.view(np.uint16)
    if tensor2.dtype == "|V2":
        tensor2 = tensor2.view(np.uint16)
except Exception as e:
    st.error(f"Error loading tensors: {e}")
    st.stop()

if tensor1.shape != tensor2.shape:
    st.error(f"Tensors have different shapes: {tensor1.shape} vs {tensor2.shape}")
    st.stop()

if tensor1.dtype != tensor2.dtype:
    st.error(f"Tensors have different dtypes: {tensor1.dtype} vs {tensor2.dtype}")
    st.stop()

tshape = "x".join([str(x) for x in tensor1.shape])
st.write(f"Tensors shape: {tshape}")
st.write(f"Tensors dtype: {tensor1.dtype}")

reshape = st.checkbox("View tensors as 1D", value=False)

if reshape:
    tensor1 = tensor1.reshape(-1)
    tensor2 = tensor2.reshape(-1)

st.subheader("Select slice to view")

shape = tensor1.shape

idx = []

for dim in range(len(shape)):        
    if shape[dim] > 1:                
        start, end = st.slider("Dimension " + str(dim), 0, shape[dim] - 1, (0, shape[dim] - 1))
        idx.append(slice(start, end + 1))
    else:
        st.caption("Dimension " + str(dim))
        st.write(1)
        idx.append(slice(0, 1))

tensor1_slice = tensor1[tuple(idx)].squeeze()
tensor2_slice = tensor2[tuple(idx)].squeeze()

st.subheader("Selected slice")

st.write(f"Selected slice: {idx}")
tshape = "x".join([str(x) for x in tensor1_slice.shape])
st.write(f"Selected slice shape: {tshape}")

if len(tensor1_slice.shape) > 2:
    st.error("Please select a slice with at most two dimensions greater than 1.")
    st.stop()

if tensor1_slice.size == 0:
    st.error("Selected slice is empty.")
    st.stop()

view = st.radio("Show", ["Tensor 1", "Tensor 2", "Difference"])

if view == "Tensor 1":
    st.write("Tensor 1")
    st.dataframe(tensor1_slice)
elif view == "Tensor 2":
    st.write("Tensor 2")
    st.dataframe(tensor2_slice)

elif view == "Difference":
    st.write("Difference")    
    
    diff = np.abs(tensor1_slice - tensor2_slice)

    with st.sidebar:    
        show_difference_map(diff)

    st.dataframe(diff)


