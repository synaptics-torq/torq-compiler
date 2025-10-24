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
    for idx in np.ndindex(tensor.shape[:-1]):
        data = tensor[idx]
        print("\033[90m", end="")
        print(colorIx + "     ", " ".join(f"{v:{field_width-1}}:" for v in range(items_per_row//2)), end="   ")
        print(" ".join(f"{v:{field_width-1}}:" for v in range(items_per_row//2,items_per_row)), colorEnd)
        print("\033[0m", end="")
        print("[", end="")
        line_count = range(0, len(data), items_per_row)
        for line, i in enumerate(line_count):
            if len(line_count) > 1:
                print(colorIx + f"{line:{2 if line == 0 else 3}}: " + colorEnd, end="")
                print(" ".join(f"{v:{field_width}}" for v in data[i:i+items_per_row//2]), end="   ")
                print(" ".join(f"{v:{field_width}}" for v in data[i+items_per_row//2:i+items_per_row]), end="")
            print("]" if line == len(line_count) - 1 else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--dst', help='Destination image file')
    parser.add_argument('-c', '--channel', help='Channel to visualize', type=int, default=None)
    parser.add_argument('-a', '--all', help='Print all elements', action='store_true')
    parser.add_argument('-n', '--nowrap', help='Don\'t wrap lines', action='store_true')
    parser.add_argument('-s', '--scale', help='Scale values for visualization', type=float)
    parser.add_argument('-d', '--delta', help='Max acceptable delta', type=int, default=0)
    parser.add_argument('-f', '--fix', help='Fix input', action='store_true')
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
    tensor = np.load(str(args.src[0]))
    print("Type: ", tensor.dtype)
    if args.ref:
        msg = "diff"
        reference_tensor = np.load(str(args.ref))
        print("Reference type: ", reference_tensor.dtype)
        if tensor.shape != reference_tensor.shape:
            print("Tensors have different shapes: ", tensor.shape, reference_tensor.shape)
            return
    else:
        reference_tensor = np.zeros_like(tensor)

    if args.fix:
        # Remove initial garbage (16 bytes per channel)
        tensor = tensor.flatten()
        shift = 16 * reference_tensor.shape[3]
        tensor = tensor[shift:]
        # Append the final part of the 2nd tensor to avoid mismatch
        tensor = np.concatenate((tensor, reference_tensor.flatten()[-shift:]))
        # Restore shape
        tensor = tensor.reshape(reference_tensor.shape)
        print("Input tensor fixed")

    # Compute difference between input and reference tensor
    tensor = tensor - reference_tensor

    if len(tensor.shape) > 4:
        print("Unsupported tensor shape")
        return
    elif len(tensor.shape) == 4:
        tensor = np.transpose(tensor, (0, 3, 1, 2))
    else:
        new_shape = (1,) * (4 - len(tensor.shape))+ tensor.shape
        tensor = np.reshape(tensor, new_shape)

    # Check for non-null channels
    mismatch_channels = {}
    for i in range(tensor.shape[1]):
        abs_diff = np.abs(tensor[0, i, :, :])
        max_abs_diff = np.max(abs_diff)
        if max_abs_diff > args.delta:
            l2 = np.sum(np.square(tensor[0, i, :, :])) / (tensor.shape[2] * tensor.shape[3])
            mismatch_channels[i] = (max_abs_diff, round(float(l2), 3))

    if args.ref:
        if mismatch_channels:
            print("Mismatching channels:", mismatch_channels)
        else:
            print("Match")

    # Show the desired channel if specified, otherwise all not-matching channels
    if args.channel is not None:
        mismatch_channels = {args.channel: 0}

    np.set_printoptions(threshold=np.inf if args.all else 64 * 32, linewidth=(100000 if args.nowrap else 120))
    channel_data = tensor[0, 0, :, :]
    for channel in mismatch_channels.keys():
        channel_data = tensor[0, channel, :, :]
        print(f"Channel {channel} {msg}:")
        print(channel_data)

    if args.scale:
        channel_data = channel_data * args.scale

    if args.dst:
        # Plot the tensor as a grayscale image
        matplotlib.use('Agg')

        # Use red for positive and blue for negative numbers
        color_map = plt.cm.seismic # plt.cm.gray if diff is None else plt.cm.seismic
        plt.imshow(channel_data, cmap=color_map, vmin=-128, vmax=127)
        plt.savefig(args.dst)


    # # Save tensors to files
    np.save("one.npy", np.ones([1,32,32,8], dtype=np.int8))
    np.save("two.npy", np.ones([1,32,32,8], dtype=np.int8) * 2)
    np.save("ten.npy", np.ones([1,32,32,8], dtype=np.int8) * 10)
    np.save("in_50.npy", np.ones([1,32,32,8], dtype=np.int8) * 50)
    np.save("in_rand.npy", np.random.randint(-20, 20, [1,32,32,8], dtype=np.int8))

    np.save("in_rand_1x4x32x3.npy", np.random.randint(-20, 20, [1,4,32,3], dtype=np.int8))
    np.save("in_0_1x4x32x3.npy", np.ones([1,4,32,3], dtype=np.int8) * 0)
    np.save("in_50_1x4x32x3.npy", np.ones([1,4,32,3], dtype=np.int8) * 50)
    np.save("in_rand_1x224x224x32.npy", np.random.randint(-50, 50, [1,224,224,32], dtype=np.int8))

    # Fill each channel with a different value
    tensor = np.zeros([1,32,32,8], dtype=np.int8)
    for i in range(8):
        tensor[0, :, :, i] = (i - 4) * 10
    np.save("in_pc.npy", tensor)
    tensor[0, 10, 10, 0] = 33
    np.save("in_pc33.npy", tensor)
    
    # Create a tensor 1x96x112x112 with progressive value in each channel
    tensor = np.zeros([1,112,112,3], dtype=np.int8)
    for i in range(3):
        for j in range(112):
            for k in range(112):
                tensor[0, j, k, i] = k % 2 + 2 * (j % 2) + ( i  % 10 ) * 10 + 1

    np.save("in_1x3x112x112.npy", tensor)


if __name__ == "__main__":
    main()
