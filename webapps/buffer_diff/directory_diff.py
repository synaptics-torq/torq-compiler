from collections import defaultdict
import streamlit as st
import sys
import numpy as np
import os
import pandas as pd

def load_buffer_info(directory):
    results = []

    with open(directory + '/buffers.csv') as f:
        headers = f.readline().strip().split(';')

        for row in f.readlines():
            row = row.strip().split(';')
            results.append({header: value for header, value in zip(headers, row)})

    results = {int(info["id"]): info for info in results}

    return results

def load_buffer(directory, action_id, buffer_id):
    """Load a buffer from the specified directory."""
    file_path = os.path.join(directory, f"action{action_id}", f"buffer_{buffer_id}.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Buffer file {file_path} does not exist.")
    return np.load(file_path)

@st.cache_data()
def compute_status(dir1, dir2):

    files1 = list_files(dir1)
    files2 = list_files(dir2)
    all_files = sorted(set(files1) | set(files2))

    buffer_info1 = load_buffer_info(dir1)

    cols = defaultdict(lambda: [])

    buffers1 = {}
    buffers2 = {}

    for idx in range(len(all_files)):

        key = all_files[idx]
        action_id, buffer_id = key

        if key in files1:
            data1 = load_buffer(dir1, action_id, buffer_id)

            if buffer_id in buffers1:
                if np.array_equal(buffers1[buffer_id], data1):
                    action1_change = "same"
                else:
                    action1_change = "changed"
            else:
                action1_change = "created"

            buffers1[buffer_id] = data1
        else:
            action1_change = "not in dir1"

        if key in files2:
            data2 = load_buffer(dir2, action_id, buffer_id)

            if buffer_id in buffers2:
                if np.array_equal(buffers2[buffer_id], data2):
                    action2_change = "same"
                else:
                    action2_change = "changed"
            else:
                action2_change = "created"

            buffers2[buffer_id] = data2
        else:
            action2_change = "not in dir2"

        if key in files1 and not key in files2:
            status = "Only in Directory 1"
        elif key in files2 and not key in files1:
            status = "Only in Directory 2"
        else:
            if not np.array_equal(buffers1[buffer_id], buffers2[buffer_id]):
                status = "Different content"
            else:
                status = "Identical"

        info = buffer_info1[buffer_id]

        cols["action_id"].append(action_id)
        cols["buffer_id"].append(buffer_id)
        cols["dir1_vs_dir2"].append(status)
        cols["dir1_prev"].append(action1_change)
        cols["dir2_change"].append(action2_change)

        for field in ["type", "shape", "strides", "address", "size"]:
            cols[field].append(info[field])

    return pd.DataFrame(cols)

def list_files(directory):
    """Recursively list all files in a directory."""
    file_list = []

    for action_dir in os.listdir(directory):

        if not action_dir.startswith("action"):
            continue

        action_path = os.path.join(directory, action_dir)

        for buffer_file in os.listdir(action_path):
            if not buffer_file.startswith("buffer_"):
                continue

            buffer_id = int(buffer_file[len("buffer_"):-len(".npy")])
            action_id = int(action_dir[len("action"):])

            file_list.append((action_id, buffer_id))

    return set(file_list)


def show_dir_difference(dir1, dir2, action_id, buffer_id):
    
    buffer_data_1 = load_buffer(dir1, action_id, buffer_id)
    buffer_data_2 = load_buffer(dir2, action_id, buffer_id)

    st.write("File 1: `" + os.path.join(dir1, f"action{action_id}", f"buffer_{buffer_id}.npy`"))
    st.write("File 2: `" + os.path.join(dir2, f"action{action_id}", f"buffer_{buffer_id}.npy`"))             

    show_buffer_difference(buffer_data_1, buffer_data_2, "dir1", "dir2")


def show_arbitrary_buffer_difference(df, dir1, dir2):

    st.write("### Arbitrary Buffer Comparison")

    buffers = []

    for line in df.itertuples():
        buffers.append((dir1, line.action_id, line.buffer_id))
        buffers.append((dir2, line.action_id, line.buffer_id))

    buffer1 = st.selectbox("Select first buffer to compare", buffers)
    buffer2 = st.selectbox("Select second buffer to compare", buffers)

    buffer_data_1 = load_buffer(*buffer1)
    buffer_data_2 = load_buffer(*buffer2)

    show_buffer_difference(buffer_data_1, buffer_data_2, "Buffer 1", "Buffer 2")


def show_buffer_difference(buffer_data_1, buffer_data_2, tag_1, tag_2):
        
    view_mode = st.selectbox("Mode", ["Difference", tag_1, tag_2])

    if view_mode == tag_1:
        buffer_data = buffer_data_1
    elif view_mode == tag_2:
        buffer_data = buffer_data_2        
    else:        
        buffer_data = buffer_data_2 - buffer_data_1

    if len(buffer_data.shape) > 2:
        idx = []
        
        for dim in range(len(buffer_data.shape) - 2):        
            if buffer_data.shape[dim] > 1:                
                idx.append(st.slider("Dimension " + str(dim), 0, buffer_data.shape[dim] - 1, 0))        
            else:
                st.caption("Dimension " + str(dim))
                st.write(1)
                idx.append(0)

        st.caption("Showing slice[" + ", ".join([str(i) for i in idx]) + ",:,:]")

        buffer_data_slice = buffer_data[tuple(idx)].squeeze()
    
    else:
        buffer_data_slice = buffer_data

    if view_mode == "Difference":

        control_col, diff_col, dir1_col, dir2_col = st.columns(4)

        with control_col:
            if len(buffer_data_slice.shape) > 1:
                buffer_size = buffer_data_slice.shape[0] * buffer_data_slice.shape[1]
            else:
                buffer_size = buffer_data_slice.shape[0]

            map_width = st.slider("Map width (pixels)", 1, buffer_size, 64)

            map_size = int(np.ceil(buffer_size / map_width) * map_width)


        def expand_to_map_size(data, fill_value=np.nan):
            buffer_map = np.full((map_size, ), fill_value)
            buffer_map[:data.size] = data.flatten()
            buffer_map = buffer_map.reshape(-1, map_width)            
            return buffer_map
        
        def upscale(image):
            # upscale the image with nearest neighbor interpolation x4
            return np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)

        with diff_col:
            # create numpy array with a three channels image from buffer_data_slice with color of a pixel ref if the value in the original data is not zero, otherwise white

            buffer_map = expand_to_map_size(buffer_data_slice)
            buffer_map = np.expand_dims(buffer_map, axis=-1)

            # set color to red if the value is not zero, otherwise white, nans are set to gray
            buffer_map = np.where(
                np.isnan(buffer_map),
                np.array([0, 0, 255], dtype=np.uint8),
                np.where(
                    buffer_map == 0,
                    np.array([180, 180, 180], dtype=np.uint8),
                    np.array([255, 0, 0], dtype=np.uint8)
                )
            )

            # upscale the image with nearest neighbor interpolation x4
            buffer_map = upscale(buffer_map)

            st.image(buffer_map, caption="Buffer difference image (red difference, gray unchanged, blue outside buffer)")
        
        buffer1_image = buffer_data_1.astype(np.float32)
        buffer2_image = buffer_data_2.astype(np.float32)

        total_min = min(buffer1_image.min(), buffer2_image.min())
        total_max = max(buffer1_image.max(), buffer2_image.max())

        buffer1_image = expand_to_map_size(buffer1_image, total_min)
        buffer2_image = expand_to_map_size(buffer2_image, total_min)

        scale = total_max - total_min
        if scale == 0:
            scale = 1

        buffer1_image = (buffer1_image - total_min) / scale
        buffer2_image = (buffer2_image - total_min) / scale

        # upscale the image by 4 using nearest neighbor interpolation
        with dir1_col:
            buffer1_image = upscale(buffer1_image)
            st.image(buffer1_image, caption=tag_1 + " image")

        with dir2_col:
            buffer2_image = upscale(buffer2_image)
            st.image(buffer2_image, caption=tag_2 + " image")

    st.dataframe(buffer_data_slice, hide_index=False)

def main():
    st.set_page_config(page_title="Torq Buffer dump comparison tool", layout="wide")

    st.title("Torq Buffer dump comparison tool")

    if len(sys.argv) != 3:
        st.write("Please provide exactly two tensor paths to compare.")
        st.stop()

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    if not os.path.isdir(dir1):        
        st.write(f"`{dir1}` must be a directory")
        st.stop()

    if not os.path.isdir(dir2):
        st.write(f"`{dir2}` must be a directory")
        st.stop()

    st.write(f"Directory 1: `{dir1}`")
    st.write(f"Directory 2: `{dir2}`")

    df = compute_status(dir1, dir2)

    st.write("### Comparison Results")

    show_identical = st.checkbox("Show identical files", value=False, key="show_identical")
    show_only_dir1 = st.checkbox("Show only in Directory 1", value=True, key="show_only_dir1")
    show_only_dir2 = st.checkbox("Show only in Directory 2", value=True, key="show_only_dir2")

    filters = []
    if show_identical:
        filters.append("Identical")
    if show_only_dir1:
        filters.append("Only in Directory 1")
    if show_only_dir2:
        filters.append("Only in Directory 2")

    filters += ["Different content"]

    filtered_df = df[df["dir1_vs_dir2"].isin(filters)]

    selection = st.dataframe(filtered_df, selection_mode="single-row", on_select="rerun")
    
    if len(selection['selection']['rows']) != 0:
        
        row_id = selection['selection']['rows'][0]

        action_id = filtered_df.iloc[row_id]["action_id"]
        buffer_id = filtered_df.iloc[row_id]["buffer_id"]

        st.write(f"### Buffer Difference for Action {action_id}, Buffer {buffer_id}")

        show_dir_difference(dir1, dir2, action_id, buffer_id)

    else:
        st.write("Select a row to see the buffer difference between the two directories.")

    show_arbitrary_buffer_difference(df, dir1, dir2)


main()