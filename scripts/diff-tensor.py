#!/usr/bin/env python3

# Compare two tensors and visualize the differences (as text and/or image)
# The tensors are loaded from .npy files and must have NHWC format

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys

def load_data(filename):
    # Load numpy array from file
    tensor = np.load(str(filename))
    if tensor.dtype == np.dtype('V2'):
        try:
            import ml_dtypes
        except ImportError:
            print("ml_dtypes not found, bfloat16 support not available")
            sys.exit(1)
            
        tensor = tensor.view(ml_dtypes.bfloat16)
    return tensor

def get_channel_count(tensor):
    # Assume NCW or NCHW... format
    if len(tensor.shape) >= 3:
        return tensor.shape[1]
    elif len(tensor.shape) == 2:
        return tensor.shape[0]
    return 1

def get_channel(tensor, channel):
    # Assume NCW or NCHW... format
    if len(tensor.shape) == 1:
        return tensor
    elif len(tensor.shape) == 2:
        return tensor[channel]
    return tensor[0, channel, ...]

def print_tensor(tensor):
    items_per_row = 32 if tensor.dtype in [np.int8, np.uint8, np.bool_] else 16
    field_width = 4 if tensor.dtype in [np.int8, np.uint8, np.bool_] else 6 if tensor.dtype in [np.int16, np.uint16] else 12
    colorIx = "\033[90m"
    colorEnd = "\033[0m"
    
    # Iterate over all dimensions except the last
    first_row = True
    for idx in np.ndindex(tensor.shape[:-1]):
        data = tensor[idx]
        if first_row:
            first_row = False
            print("\033[90m", end="")
            print(colorIx + "     ", " ".join(f"{v:{field_width-1}}:" for v in range(min(len(data), items_per_row//2))), end="   ")
            print(" ".join(f"{v:{field_width-1}}:" for v in range(items_per_row//2,min(len(data), items_per_row))), colorEnd)
            print("\033[0m", end="")
        print("[", end="")
        line_count = range(0, len(data), items_per_row)
        for line, i in enumerate(line_count):
            if len(line_count) > 1:
                print(colorIx + f"{line:{2 if line == 0 else 3}}: " + colorEnd, end="")
            else:
                print("    ", end="")
            print(" ".join(f"{v:{field_width}}" for v in data[i:i+items_per_row//2]), end="   ")
            print(" ".join(f"{v:{field_width}}" for v in data[i+items_per_row//2:i+items_per_row]), end="")
            print("]" if line == len(line_count) - 1 else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--dst', help='Destination image file')
    parser.add_argument('-c', '--channel', help='Channel to visualize', type=int, default=None)
    parser.add_argument('-a', '--all', help='Print all elements', action='store_true')
    parser.add_argument('-t', '--transpose', help='Transpose NHWC to NCHW', action='store_true')
    parser.add_argument('-n', '--nowrap', help='Don\'t wrap lines', action='store_true')
    parser.add_argument('-x', '--scale', help='Scale values for visualization', type=float)
    parser.add_argument('-s', '--sort', help='Sort mistmatching channels', action='store_true')
    parser.add_argument('-d', '--delta', help='Max acceptable delta', type=float, default=0)
    parser.add_argument('-1', help='Show first file instead of difference', dest='first', action='store_true')
    parser.add_argument('-2', help='Show second file instead of difference', dest='second', action='store_true')
    parser.add_argument('src', help='Tensor file (.npy)', nargs=1)
    parser.add_argument('ref', help='Reference tensor file (.npy)', nargs='?')
    args = parser.parse_args()

    msg = ""
    if args.first:
        msg = "first"
        args.ref = None
    elif args.second:
        msg = "second"
        args.src[0] = args.ref
        args.ref = None

    # Load numpy array(s) from file
    tensor = load_data(str(args.src[0]))
    print("Data type: ", tensor.dtype)
    if args.ref:
        msg = "diff"
        reference_tensor = load_data(str(args.ref))
        if tensor.dtype != reference_tensor.dtype:
            print("Reference data type: ", reference_tensor.dtype)
        if tensor.shape != reference_tensor.shape:
            print("Tensors have different shapes: ", tensor.shape, reference_tensor.shape)
            return
    else:
        reference_tensor = np.zeros_like(tensor)

    # Compute difference between input and reference tensor
    tensor = tensor - reference_tensor

    # Input data is considered NCHW by default is 4D, NCW if 3D
    # if not the case it can be transposed with -t option
    if args.transpose:
        if len(tensor.shape) == 4:
            tensor = np.transpose(tensor, (0, 3, 1, 2))
        elif len(tensor.shape) == 3:
            tensor = np.transpose(tensor, (0, 2, 1))
        else:
            print("Transpose only supported for 4D tensors")

    print("Tensor shape: ", tensor.shape)

    # Check for non-null channels
    mismatch_channels = {}
    for i in range(get_channel_count(tensor)):
        abs_diff = np.abs(get_channel(tensor, i))
        max_abs_diff = np.max(abs_diff)
        if max_abs_diff > args.delta:
            channel_data = get_channel(tensor, i)
            l2 = np.sqrt(np.sum(np.square(channel_data * 1.0)) / channel_data.size)
            mismatch_channels[i] = (float(l2), float(max_abs_diff))

    if args.ref:
        if mismatch_channels:
            # sort by average diff
            if args.sort:
                mismatch_list = sorted(mismatch_channels.items(), key=lambda item: item[1], reverse=True)
            else:
                mismatch_list = list(mismatch_channels.items())
            print(len(mismatch_list) , "mismatching channels:", end=' ')
            mismatch_list_top = mismatch_list[:50]
            for mc in mismatch_list_top:
                print(f"{mc[0]}: ({mc[1][0]:.2g};{mc[1][1]:.2g}) ", end=' ')
            if len(mismatch_list) > len(mismatch_list_top):
                print("...")
        else:
            print("Match")

    # Show the desired channel if specified, otherwise all not-matching channels
    if args.channel is not None:
        mismatch_channels = {args.channel: 0}

    np.set_printoptions(threshold=np.inf if args.all else 64 * 32, linewidth=(100000 if args.nowrap else 200))
    channel_data = get_channel(tensor, 0)
    for channel in mismatch_channels.keys():
        channel_data = get_channel(tensor, channel)
        print(f"Channel {channel} {msg}:")
        if channel_data.size > 64 * 32 and not args.all:
            print(channel_data)
        else:
            print_tensor(channel_data)

    if args.scale:
        channel_data = channel_data * args.scale

    if args.dst:
        # Plot the tensor as a grayscale image
        matplotlib.use('Agg')

        # Use red for positive and blue for negative numbers
        color_map = plt.cm.seismic # plt.cm.gray if diff is None else plt.cm.seismic
        plt.imshow(channel_data, cmap=color_map, vmin=-128, vmax=127)
        plt.savefig(args.dst)

if __name__ == "__main__":
    main()
