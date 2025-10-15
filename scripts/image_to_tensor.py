#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

# Extended from https://github.com/synaptics-synap/toolkit/blob/main/image_to_raw.py
# 
# Convert and rescale a .jpg or .png image to raw format
#

import os
import cv2
import numpy as np
import argparse
import sys
import os.path

def image_to_raw(image_file, raw_filename, w, h, type, add_batch_dim=True):
    base = os.path.splitext(image_file)[0]
    if raw_filename.startswith(".") and not "/" in raw_filename:
        raw_filename = base + raw_filename
    if not type:
        type = os.path.splitext(raw_filename)[1][1:]
    type = type.lower()
    fullimg = cv2.imread(image_file, 3)
    image_data_type = fullimg.dtype

    width = w if w != 0 else fullimg.shape[1]
    height = h if h != 0 else fullimg.shape[0]
    y_size = height * width
    scale = min(width/fullimg.shape[1], height/fullimg.shape[0])
    if scale < 1:
        # Apply gaussian blur when downscaling to avoid artifacts
        sigma =  (1 / scale - 1) / 2
        fullimg = cv2.GaussianBlur(fullimg, (0, 0), sigma, sigma)
    img = cv2.resize(fullimg, (width,height), interpolation=cv2.INTER_AREA)
    # Handle npy output for supported types
    if type == 'npy':
        img_rgb = img[:, :, -1::-1]
        if add_batch_dim:
            img_rgb = np.expand_dims(img_rgb, axis=0)
        np.save(raw_filename, img_rgb.astype('uint8'))
        print(f"Saved .npy tensor (RGB): shape={img_rgb.shape}, dtype={img_rgb.dtype}")
        sys.exit(0)
    if type == 'npy_bgr':
        img_bgr = img
        if add_batch_dim:
            img_bgr = np.expand_dims(img_bgr, axis=0)
        np.save(raw_filename, img_bgr.astype('uint8'))
        print(f"Saved .npy tensor (BGR): shape={img_bgr.shape}, dtype={img_bgr.dtype}")
        sys.exit(0)
    if type == 'rgb':
        img_rgb = img[:,:,-1::-1]
        img_rgb.astype('uint8').tofile(raw_filename)
        print(f"Saved .bin tensor (RGB): shape={img_rgb.shape}, dtype={img_rgb.dtype}")

        sys.exit(0)
    if type == 'rgbp':
        # RGB planar
        img_rgb = img[:,:,-1::-1]
        img_rgbp = np.transpose(img_rgb, (2, 0, 1))
        img_rgbp.astype('uint8').tofile(raw_filename)
        print(f"Saved .bin tensor (RGB planar): shape={img_rgbp.shape}, dtype={img_rgbp.dtype}")

        sys.exit(0)
    elif type == 'bgr':
        img.astype('uint8').tofile(raw_filename)
        print(f"Saved .bin tensor (BGR): shape={img.shape}, dtype={img.dtype}")

        sys.exit(0)
    elif type == 'bgra':
        shape = list(img.shape)
        shape[2] += 1
        bgra = np.ones(shape) * 255
        bgra[:,:,:-1] = img
        bgra.astype('uint8').tofile(raw_filename)
        print(f"Saved .bin tensor (BGRA): shape={bgra.shape}, dtype={bgra.dtype}")
        sys.exit(0)

    # to YUV and normalize
    img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_y = img_ycc[:,:,0]

    sigma =  (1 / 0.5 - 1) / 2
    blurred_img = cv2.GaussianBlur(img_ycc, (0, 0), sigma, sigma)
    img_quarter = cv2.resize(blurred_img, (width//2,height//2), interpolation=cv2.INTER_AREA)

    if type == 'nv12' or type == 'nv15':
        img_uv = img_quarter[:,:,1:3] # uvuvuv..
    elif type == 'nv21':
        img_uv = img_quarter[:,:,2:0:-1]  #vuvuvu..
    else:
        print("Unknown type:", type)
        sys.exit(1)

    img_yuv = np.concatenate((np.reshape(img_y, y_size), np.reshape(img_uv, y_size // 2)))

    if type == 'nv15':
        # Align image to 10 bits
        if image_data_type == np.uint8:
            data16 = img_yuv.astype('uint16') * 4
        elif image_data_type == np.uint16:
            data16 = img_yuv.astype('uint16') // 64
        else:
            print("Unsupported image data type:", image_data_type)
            sys.exit(1)
        # Pack image to 10 bits
        data8 = np.zeros(data16.size * 5 // 4, dtype=np.uint8)
        for i in range(data16.size // 4):
            si = i * 4
            di = i * 5
            data8[di] = data16[si] & 0xff
            data8[di + 1] = ((data16[si] >> 8) & 0x3) + ((data16[si + 1] & 0x3f) << 2)
            data8[di + 2] = ((data16[si + 1] >> 6) & 0xf) + ((data16[si + 2] & 0xf) << 4)
            data8[di + 3] = ((data16[si + 2] >> 4) & 0x3f) + ((data16[si + 3] & 0x3) << 6)
            data8[di + 4] = (data16[si + 3] >> 2) & 0xff
        data8.tofile(raw_filename)
        print(f"Saved .bin tensor (NV15): shape={data8.shape}, dtype={data8.dtype}")
    else:
        img_yuv.astype('uint8').tofile(raw_filename)
        print(f"Saved .bin tensor ({type.upper()}): shape={img_yuv.shape}, dtype={img_yuv.dtype}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--src', help='Source image (.png or .jpg)')
    parser.add_argument('-o', '--dst', help='Destination file')
    parser.add_argument('--format', help='Processing format: npy, npy_bgr nv21, nv12, nv15, rgb, rgbp, bgr, bgra')
    parser.add_argument('-s', '--size', help='Raw image pixel size: WxH')
    parser.add_argument('--no-batch', action='store_true', help='Do not add batch dimension to npy/npy_bgr output')
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        print(f"Error: file {args.src} not found.")
        sys.exit(1)

    w, h = 0, 0
    if args.size:
        try:
            w, h = [int(i) for i in args.size.split('x')]
        except:
            print("Invalid size specified, use: -s WxH" , file=sys.stderr)
            sys.exit(1)

    image_to_raw(args.src, args.dst, w, h, args.format, add_batch_dim=not args.no_batch)

if __name__ == "__main__":
    main()
