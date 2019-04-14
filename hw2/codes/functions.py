import numpy as np


def im2col(x, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x.shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    h_out = input.shape[2] + 2 * pad - kernel_size + 1
    w_out = input.shape[3] + 2 * pad - kernel_size + 1
    c_out = W.shape[0]
    c_in = W.shape[1]
    input_col2 = im2col(input, kernel_size, kernel_size, pad, 1)
    input_col2 = input_col2.T.reshape(h_out, w_out, input.shape[0], -1)
    w_col = W.reshape((c_out, c_in * kernel_size * kernel_size))
    output = np.dot(input_col2, w_col.T) + b
    return output.transpose(2, 3, 0, 1)


def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
    _, c_in, h_in, w_in = input.shape
    _, c_out, h_out, w_out = grad_output.shape
    grad_output_col = im2col(grad_output, kernel_size, kernel_size, kernel_size - 1, 1)
    h_padded = grad_output.shape[2] + kernel_size - 1
    w_padded = grad_output.shape[3] + kernel_size - 1
    grad_output_col = grad_output_col.T.reshape(h_padded, w_padded, grad_output.shape[0], -1)
    W_reshaped = np.rot90(W, 2, (2, 3)).transpose(1, 0, 2, 3).reshape(c_in, -1)
    grad_input = np.dot(grad_output_col, W_reshaped.T).transpose(2, 3, 0, 1)
    grad_input = grad_input[:, :, pad: h_in + pad, pad: w_in + pad]
    input_col = im2col(input.transpose(1, 0, 2, 3), h_out, h_out, pad, 1)
    h_padded = input.shape[2] + 2 * pad - h_out + 1
    w_padded = input.shape[3] + 2 * pad - h_out + 1
    input_col = input_col.T.reshape(h_padded, w_padded, input.shape[1], -1)
    grad_W = np.dot(input_col, grad_output.transpose(1, 0, 2, 3).reshape(c_out, -1).T).transpose(3, 2, 0, 1)
    grad_b = np.sum(grad_output, axis=(0, 2, 3))
    return grad_input, grad_W, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    _, c_in, h_in, w_in = input.shape
    h_out = int((h_in + 2 * pad - kernel_size) / kernel_size + 1)
    w_out = int((w_in + 2 * pad - kernel_size) / kernel_size + 1)
    W = 1.0 / (kernel_size * kernel_size) * np.ones((c_in, c_in, kernel_size, kernel_size))
    input_col2 = im2col(input, kernel_size, kernel_size, pad, kernel_size)
    input_col2 = input_col2.T.reshape(h_out, w_out, input.shape[0], -1)
    w_col = W.reshape((c_in, c_in * kernel_size * kernel_size))
    output = np.dot(input_col2, w_col.T)
    return output.transpose(2, 3, 0, 1)


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    grad_input = grad_output.repeat(kernel_size, axis=2).repeat(kernel_size, axis=3) / (kernel_size * kernel_size)
    return grad_input[:, :, pad: grad_output.shape[2] * kernel_size - pad,
                      pad: grad_output.shape[3] * kernel_size - pad]
