config_conv = [
     {'name' : 'conv2d_8x8',
       'quant' : 'int8',
       'api' : 'tf.keras.layers.Conv2D',
       'input_shapes': [(1, 8, 8, 1)],
       'params' : {'filters': 1, 'kernel_size': [3, 3], 'strides': [1, 1], 'padding': 'same', 'activation': 'relu'}
     },
     {'name' : 'conv2d_64x64',
       'quant' : 'int8',
       'api' : 'tf.keras.layers.Conv2D',
       'input_shapes': [(1, 64, 64, 1)],
       'params' : {'filters': 1, 'kernel_size': [3, 3], 'strides': [2, 2], 'padding': 'same', 'activation': 'relu'}
     },
]

config_add = [
     {'name' : 'add_8x8',
       'quant' : 'int8',
       'api' : 'tf.keras.layers.Add',
       'input_shapes': [(1, 8, 8, 1), (1, 8, 8, 1)],
       'params' : {}
     }
]

config_maxpool2d = [
     {'name' : 'maxpool2d_8x8',
      'quant' : 'int8',
      'api' : 'tf.keras.layers.MaxPool2D',
      'input_shapes': [(1, 8, 8, 1)],
      'params' : {'pool_size': (3, 3), 'strides': (1, 1), 'padding': 'same'}
     }
]