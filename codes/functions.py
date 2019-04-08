import numpy as np
from im2col import im2col_indices
from skimage.measure import block_reduce


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
    input_col2 = im2col_indices(input, kernel_size, kernel_size, pad, 1)
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
    _, C_IN, H_IN, W_IN = input.shape
    _, C_OUT, H_OUT, W_OUT = grad_output.shape
    grad_output_col = im2col_indices(grad_output, kernel_size, kernel_size, kernel_size - 1, 1)
    h_padded = grad_output.shape[2] + kernel_size - 1
    w_padded = grad_output.shape[3] + kernel_size - 1
    grad_output_col = grad_output_col.T.reshape(h_padded, w_padded, grad_output.shape[0], -1)
    W_reshaped = np.rot90(W, 2, (2, 3)).transpose(1, 0, 2, 3).reshape(C_IN, -1)
    grad_input = np.dot(grad_output_col, W_reshaped.T).transpose(2, 3, 0, 1)
    grad_input = grad_input[:, :, pad: H_IN + pad, pad: W_IN + pad]

    input_col = im2col_indices(input.transpose(1, 0, 2, 3), H_OUT, H_OUT, pad, 1)
    h_padded = input.shape[2] + 2 * pad - H_OUT + 1
    w_padded = input.shape[3] + 2 * pad - H_OUT + 1
    input_col = input_col.T.reshape(h_padded, w_padded, input.shape[1], -1)
    grad_W = np.dot(input_col, grad_output.transpose(1, 0, 2, 3).reshape(C_OUT, -1).T).transpose(3, 2, 0, 1)
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
    input_padded = np.lib.pad(input, ((0,), (0,), (pad,), (pad,)), 'constant')
    output = block_reduce(input_padded, block_size=(1, 1, kernel_size, kernel_size), func=np.mean)
    return output


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
