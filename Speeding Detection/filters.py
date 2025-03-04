import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    kernel = np.flip(kernel)
    # Loops through Image pixels
    for i in range(Hi):
      for j in range(Wi):
        # initialize the corresponding output pixel to 0
        out[i,j] = 0
        # Loops through Kernel pixels
        for i_ in range(Hk):
          for j_ in range(Wk):
            # Checks if index is out of bound
            if 0<= i - Hk//2 + i_ <Hi and 0<= j - Wk//2 + j_ <Wi:
              # Carries out the covolution
              out[i,j] += kernel[i_,j_] * image[i - Hk//2 + i_,j - Wk//2 + j_ ]
            else:
              # assuming out of bound pixels have 0 value
              out[i,j] += 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Pads already existing rows in the image
    padded_rows = []
    for row in image:
      padded_row = np.concatenate([np.zeros(pad_width), row, np.zeros(pad_width)])
      padded_rows.append(padded_row)
    
    # Adds new zero rows to complete the padding of the image
    padded_image = np.array(padded_rows)
    zero_box = np.zeros((pad_height,W+(2*pad_width)))
    
    out = np.array([])
    out = np.concatenate([zero_box, padded_image, zero_box])
    
    # Alternative Approach: Using Numpy's padding function
    #out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Flip the kernel
    kernel = np.flip(kernel)
    #plt.imshow(kernel)

    # Pads the image to the right dimentions
    padded_image = zero_pad(image, (Hk)//2 , (Wk)//2)

    # Carries out convolution
    for i in range(Hi):
      #print(i, Hk)
      for j in range(Wi):
        out[i,j] = np.sum(kernel * padded_image[i:i+Hk, j:j+Wk])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Since conv_fast() automatically flips the image, 
    # we will need to flip the kernel in addvance too.
    _g = np.flip(g)

    # Carry out convolution on the flipped kernel
    out = (conv_fast(f,_g))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Subtract the mean of the image from each pixel
    g = g - np.mean(g)
    # Carry out the cross_correlation calculations
    _g = np.flip(g)
    out = conv_fast(f,_g)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))

    # Normalize the kernel in advance
    g = (g-np.mean(g)) / np.std(g)

    # Pads the image to the right dimentions
    padded_image = zero_pad(f, (Hk)//2 , (Wk)//2)

    # Carries out convolution
    for i in range(Hi):
      for j in range(Wi):
        
        # Extract the patch from the image
        patch = padded_image[i:i+Hk, j:j+Wk]
        # Carry out convolution on the Normalized Patch
        out[i,j] = np.sum(((patch - np.mean(patch)) / np.std(patch)) * g)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out
