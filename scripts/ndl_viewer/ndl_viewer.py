#!/usr/bin/env python3

# Show how a NDL operates over a tensor

import numpy as np
import re
import traceback

try:
    from flask import Flask
except ImportError:
    print("Please install flask with: pip3 install Flask")
    sys.exit(1)

from flask import render_template, request


app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():

    try:

        if request.method == 'GET':
            return render_template('index.html.j2', data=None)

        ndl = request.form.get('ndl')
        memref = request.form.get('memref')

        if request.form.get('offset') == '':
            offset = 0
        else:
            offset = int(request.form.get('offset', 0))

        l_dim, l2_dim, h_dim = parse_ndl(ndl)

        tensor_size, tensor_strides = parse_memref(memref)    

        scan_order, overscan = create_scan_order(tensor_size, tensor_strides, l_dim, l2_dim, h_dim, offset)
        
        tensor_footprint = create_tensor_footprint(tensor_size, tensor_strides)

        if request.form.get('memory_width') == '':
            memory_width = tensor_strides[1]
        else:
            memory_width = int(request.form.get('memory_width'))

        scan_order = scan_order.reshape((-1, memory_width))
        tensor_footprint = tensor_footprint.reshape((-1, memory_width))

        data = {'l_dim': l_dim,
                'l2_dim': l2_dim,
                'h_dims': h_dim,
                'offset': offset,
                'tensor_sizes': tensor_size,
                'tensor_strides': tensor_strides,
                'scan_order': scan_order,
                'tensor_footprint': tensor_footprint,
                'memref': memref,
                'ndl': ndl,
                'offset': offset,
                'memory_width': memory_width,
                'overscan': overscan,
                'total_l_iterations': np.prod([x[0] for x in h_dim])}

    except Exception as e:
        data = {'error': str(e) + traceback.format_exc(),
                'ndl': request.form.get('ndl'),
                'memref': request.form.get('memref'),
                'offset': request.form.get('offset'),
                'memory_width': request.form.get('memory_width')}

    return render_template('index.html.j2', data=data)

def is_finished(counter, ndl):
    return counter[-1] == ndl[-1][0]


def generate_address(counter, ndl):
    address = 0
    for i in range(len(counter)):
        address += counter[i] * ndl[i][1]
    return address


def increment_count(counter, ndl):
    for i in range(len(counter) - 1):
        counter[i] += 1
        if counter[i] < ndl[i][0]:
            return
        else:
            counter[i] = 0
    
    counter[-1] += 1


def parse_ndl(ndl):
    ndl = ndl.strip()
    
    # [ B(l) [4, 1],  Y(l) [8, 4],  X(l) [2, 1280],  X(h) [16, 2560],  X(h) [2, 40960],  Y(h) [40, 32]]    

    matches = re.findall(r'[A-Z]\(([l|h])\) \[(\d+), ((-)?\d+)\]', ndl)

    l_dim = None
    l2_dim = None
    h_dims = []

    for match in matches:
        if match[0] == 'l':
            if l_dim is None:
                l_dim = int(match[1])
                if match[2] != '1':
                    raise ValueError('First l dimension stride must be 1')
            else:

                if l2_dim is not None:

                    if l_dim != l2_dim[1]:
                        raise ValueError('Unsupported l dims')

                    l_dim = l2_dim[1] * l2_dim[0]

                l2_dim = [int(match[1]), int(match[2])]

        else:
            h_dims.append([int(match[1]), int(match[2])])

    return l_dim, l2_dim, h_dims


def parse_memref(memref):

    # memref<1x1280x7x7xi8, strided<[81920, 64, 7, 1]>, #torq_hl.memory_space<lram>>

    memref = memref.strip()

    match = re.match(r'memref<(\d+)x(\d+)x(\d+)x(\d+)x(i\d+), strided<\[(\d+), (\d+), (\d+), (\d+)\]>', memref)

    if match is None:

        match = re.match(r'memref<(\d+)x(\d+)x(\d+)x(\d+)x(i\d+)', memref)

        if match is None:
            raise ValueError('Invalid memref format')

        dtype = match.group(5)

        tensor_sizes = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]

        tensor_strides = [tensor_sizes[3] * tensor_sizes[2] * tensor_sizes[1],
                          tensor_sizes[3] * tensor_sizes[2], 
                          tensor_sizes[3], 1]
    
    else:
        dtype = match.group(5)
        tensor_sizes = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
        tensor_strides = [int(match.group(6)), int(match.group(7)), int(match.group(8)), int(match.group(9))]


    if dtype != 'i8':
        raise ValueError('Unsupported data type ' + dtype)
    

    return tensor_sizes, tensor_strides


def create_scan_order(tensor_size, tensor_strides, l_dim, l2_dim, h_dims, offset):

    memory = -1 * np.ones(tensor_strides[0] * tensor_size[0], dtype=int)

    counter = [0] * len(h_dims)
    linear_counter = 0

    overscan = []

    while not is_finished(counter, h_dims):

        address = generate_address(counter, h_dims)

        if l2_dim is not None:
            l2_counts = l2_dim[0]
            l2_stride = l2_dim[1]
        else:
            l2_counts = 1
            l2_stride = 1

        for i in range(l2_counts):
            local_address = address + l2_stride * i + offset

            for j in range(l_dim):

                if local_address + j >= memory.shape[0]:
                    overscan.append((local_address + j, linear_counter, tuple(counter), (i, j)))
                else:
                    memory[local_address + j] = linear_counter

        linear_counter += 1
        increment_count(counter, h_dims)

    return memory, overscan


def create_tensor_footprint(tensor_size, tensor_strides):

    memory = np.full((tensor_strides[0] * tensor_size[0]), '', dtype=object)

    counter = [0] * len(tensor_size)
    linear_counter = 0

    dims = list(zip(tensor_size, tensor_strides))[::-1]

    # use less than the whole range to ensure we don't end up with a white or black color
    r_increment = 128 / dims[0][0]
    g_increment = 128 / dims[1][0]
    b_increment = 128 / dims[2][0]

    while not is_finished(counter, dims):

        address = generate_address(counter, dims)

        r_val = int(100 + r_increment * counter[0])        
        g_val = int(100 + g_increment * counter[1])
        b_val = int(100 + b_increment * counter[2])

        memory[address] = f'#{r_val:02x}{g_val:02x}{b_val:02x}'

        increment_count(counter, dims)

    return memory


def main():

    app.run(debug=True)


if __name__ == "__main__":
    main()
