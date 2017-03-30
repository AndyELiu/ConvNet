import tensorflow as tf
import math

# this keeps the layer number
global_layer_counter = None


def build_network(x, layer, layer_defs, weight_reuse=False):
    global global_layer_counter
    global_layer_counter = {
        'conv': 0,
        'activation': 0,
        'pooling': 0,
        'lrn': 0,
        'full-connected': 0,
        'res': 0,
        'serial': 0,
        'parallel': 0
    }
    return build_layer(x, layer, layer_defs, weight_reuse)


# build a layer
def build_layer(x, layer, layer_defs, weight_reuse=False):
    layer_def = layer_defs[layer]
    assert 'type' in layer_def

    layer_type = layer_def['type']
    layer_name = "{}-{}".format(layer_type, global_layer_counter[layer_type])
    global_layer_counter[layer_type] += 1
    if layer_type == 'conv':
        return build_conv_layer(x, layer_def, layer_name, weight_reuse)
    elif layer_type == 'activation':
        return build_activation_layer(x, layer_def, layer_name)
    elif layer_type == 'pooling':
        return build_pooling_layer(x, layer_def, layer_name)
    elif layer_type == 'lrn':
        return build_lrn_layer(x, layer_def, layer_name)
    elif layer_type == 'full-connected':
        return build_full_connected_layer(x, layer_def, layer_name,
                                          weight_reuse)
    elif layer_type == 'res':
        return build_res_layer(x, layer_def, layer_defs, layer_name,
                               weight_reuse)
    elif layer_type == 'serial':
        return build_serial_layer(x, layer_def, layer_defs, layer_name,
                                  weight_reuse)
    elif layer_type == 'parallel':
        return build_parallel_layer(x, layer_def, layer_defs, layer_name,
                                    weight_reuse)
    else:
        raise Exception('No such type of layers')


def build_lrn_layer(x, layer_def, name):
    radius = layer_def.get('radius', 5)
    bias = layer_def.get('bias', 1)
    alpha = layer_def.get('alpha', 1)
    beta = layer_def.get('beta', 0.5)
    with tf.name_scope(name):
        return tf.nn.local_response_normalization(x, radius, bias, alpha, beta)


def build_activation_layer(x, layer_def, name):
    nonlinear = layer_def.get('nonlinearity', 'relu')
    with tf.name_scope(name):
        if nonlinear == 'relu':
            x = tf.nn.relu(x)
        elif nonlinear == 'relu6':
            x = tf.nn.relu6(x)
        elif nonlinear == 'sigmoid':
            x = tf.nn.sigmoid(x)
        elif nonlinear == 'tanh':
            x = tf.nn.tanh(x)
        elif nonlinear == 'dropout':
            prob = layer_def['prob']
            x = tf.nn.dropout(x, prob)
        else:
            raise Exception('No such nonlinearity!!!')
        # tf.summary.histogram(nonlinear, x)
        return x


def build_conv_layer(x, layer_def, name, weight_reuse=False):
    c_in = x.shape.as_list()[3]
    hr = layer_def.get('filter_h', 3)
    wr = layer_def.get('filter_w', 3)
    hs = layer_def.get('stride_h', 1)
    ws = layer_def.get('stride_w', 1)
    wd = layer_def.get('weight_norm', 0)
    c_out = layer_def.get('num_filters', c_in)

    stddev = layer_def.get('stddev', math.sqrt(2.0 / (hr * wr * c_in)))
    bias = layer_def.get('bias', 0.1)
    with tf.name_scope(name):
        with tf.variable_scope(name, reuse=weight_reuse):
            W = tf.get_variable(
                'W',
                [hr, wr, c_in, c_out],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=stddev,
                                                            dtype=tf.float32))
            b = tf.get_variable('b',
                                [c_out],
                                initializer=tf.constant_initializer(bias))
            if not weight_reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(W),
                                           wd,
                                           name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            return tf.nn.conv2d(x, W, [1, hs, ws, 1], padding='SAME') + b


def build_pooling_layer(x, layer_def, name):
    hr = layer_def.get('filter_h', 2)
    wr = layer_def.get('filter_w', 2)
    hs = layer_def.get('stride_h', 2)
    ws = layer_def.get('stride_w', 2)
    pooling_type = layer_def.get('pooling_type', 'max')
    with tf.name_scope(name):
        if pooling_type == 'max':
            return tf.nn.max_pool(x, [1, hr, wr, 1], [1, hs, ws, 1], 'SAME')
        else:
            return tf.nn.avg_pool(x, [1, hr, wr, 1], [1, hs, ws, 1], 'SAME')


def build_full_connected_layer(x, layer_def, name, weight_reuse=False):
    assert 'dim' in layer_def
    dim = layer_def['dim']
    wd = layer_def.get('weight_norm', 0)
    input_shape = x.shape.as_list()
    rank = len(input_shape)
    assert rank in (2, 4)

    with tf.name_scope(name):
        if rank == 4:
            d_in = input_shape[1] * input_shape[2] * input_shape[3]
            x = tf.reshape(x, [-1, d_in])
        else:
            d_in = input_shape[1]
        stddev = layer_def.get('stddev', math.sqrt(2.0 / d_in))
        bias = layer_def.get('bias', 0.1)
        with tf.variable_scope(name, reuse=weight_reuse):
            W = tf.get_variable(
                'W',
                [d_in, dim],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=stddev,
                                                            dtype=tf.float32))
            b = tf.get_variable('b',
                                [dim],
                                initializer=tf.constant_initializer(bias))
            if not weight_reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(W),
                                           wd,
                                           name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            return tf.matmul(x, W) + b


def build_res_layer(x, layer_def, layer_defs, name, weight_reuse=False):
    assert 'layers' in layer_def
    x_ = x
    layers = layer_def['layers']
    with tf.name_scope(name):
        for layer in layers:
            x_ = build_layer(x_, layer, layer_defs, weight_reuse)
        return x + x_


def build_serial_layer(x, layer_def, layer_defs, name, weight_reuse=False):
    assert 'layers' in layer_def
    layers = layer_def['layers']
    with tf.name_scope(name):
        for layer in layers:
            x = build_layer(x, layer, layer_defs, weight_reuse)
        return x


def build_parallel_layer(x, layer_def, layer_defs, name, weight_reuse=False):
    assert 'layers' in layer_def
    layers = layer_def['layers']
    with tf.name_scope(name):
        outputs = []
        for layer in layers:
            outputs.append(build_layer(x, layer, layer_defs, weight_reuse))
        return tf.concat(outputs, axis=3)
