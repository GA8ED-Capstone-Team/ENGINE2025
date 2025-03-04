

from re import T
import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Flip the kernel
    kernel = np.flip(kernel)
    #plt.imshow(kernel)

    # # Pads the image to the right dimentions
    # padded_image = zero_pad(image, (Hk)//2 , (Wk)//2)

    # Carries out convolution
    for i in range(Hi):
      #print(i, Hk)
      for j in range(Wi):
        out[i,j] = np.sum(kernel * padded[i:i+Hk, j:j+Wk])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    k = (size -1) // 2 
    # i, j = np.arange(size)
    scalar = 1/(2*np.pi*(sigma)**2)

    # kernel = scalar * np.exp( - ((i-k)**2 + (j-k)**2) / (2*sigma**2))

    for i in range(0, size):
      for j in range (0, size):
        kernel[i][j] = np.exp( - ( (i-k)**2 + (j-k)**2 ) / (2*sigma**2) )
    
    kernel = scalar * kernel



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    kernel_x = np.array([[1/2,0,-1/2]])
    out = conv(img, kernel_x)
    #print(out)
    



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    kernel_y = np.array([[1/2],[0],[-1/2]])
    out = conv(img, kernel_y)
    #print(out)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    G = ( partial_x(img)**2 + partial_y(img)**2 )**(0.5)
    theta = np.arctan2(partial_y(img), partial_x(img)) + 180

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)

    #print(G)
    ### BEGIN YOUR CODE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    for i in range(H):
      for j in range(W):
        q = G[i,j]
        t = theta[i][j]
        #print(t)
        
        if (t==0) or (t==180) or (t==360):
          p = G[i][j-1] if 0<=j-1<W else 0
          r = G[i][j+1] if 0<=j+1<W else 0
        
        if t==45 or t==225:
          if (not 0<=i-1<H) or (not 0<=j+1<W):
            p = 0
          else: 
            p = G[i-1][j+1]
          if (not 0<=i+1<H) or (not 0<=j-1<W):
            r = 0
          else: 
            r = G[i+1][j-1]
          #print(r)

        if t == 90 or t == 270:
          p = G[i-1][j] if 0<=i-1<H else 0
          #print((i,j),p)
          r = G[i+1][j] if 0<=i+1<H else 0
          #print(r)
          
        if t==135 or t==315:
          if (not 0<=i-1<H) or (not 0<=j-1<W):
            p = 0
          else : 
            p = G[i-1][j-1]
          if (not 0<=i+1<H) or (not 0<=j+1<W):
            r = 0
          else: 
            r = G[i+1][j+1]
          #print(r)

        if q >= p and q >= r:
          out[i][j] = q
        else:
          out[i][j] = 0

    
    
    ############################# Different Algorithm ##########################
    # theta = np.deg2rad(theta)

    # for i in range(H - 1):
    #   for j in range(W - 1):
    #     dx = np.cos(theta[i][j])
    #     dy = np.sin(theta[i][j])
    #     x1_prime = j + dx
    #     y1_prime = i - dy
    #     i1 = int(y1_prime) if int(y1_prime) < H - 1 else H - 2
    #     j1 = int(x1_prime) if int(x1_prime) < W - 1 else W - 2
    #     #print("upper limit", i1, j1)


    #     x2_prime = j - dx
    #     y2_prime = i + dy
    #     i2 = int(y2_prime) if int(y2_prime) >= 0 else 0
    #     i2 = i2 if i2 < H - 1 else H - 2
    #     j2 = int(x2_prime) if int(x2_prime) >= 0 else 0
    #     j2 = j2 if j2 < W - 1 else W - 2
    #     #print("lower limit", i2, j2)
        


    #     p = 0.25 * (G[i1][j1] + G[i1+1][j1] + G[i1][j1+1] + G[i1+1][j1+1])
    #     q = G[i][j]
    #     r = 0.25 * (G[i2][j2] + G[i2+1][j2] + G[i2][j2+1] + G[i2+1][j2+1])

    #     if max(q, p, r) == q:
    #       out[i][j]=q
    #     else:
    #       out[i][j]=0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype= "bool")
    weak_edges = np.zeros(img.shape, dtype= "bool" )

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    strong_edges = img > high
    weak_edges = (img > low) & (img <= high)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype= "bool")

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    indices = list(indices)
    while indices:

      strong_i, strong_j  = indices.pop()
      #print(strong_i, strong_j)

      for weak_i, weak_j in get_neighbors(strong_i, strong_j, H, W):
        #print(weak_i, weak_j)
        if weak_edges[weak_i][weak_j]:
          #print(weak_i, weak_j)
          weak_edges[weak_i][weak_j] = False
          edges[weak_i][weak_j] = True
          indices.append((weak_i, weak_j))


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Step 1: Smooth the image using a Gaussian kernel
    smoothed_img = conv(img, gaussian_kernel(kernel_size, sigma))
    # Step 2: Compute the Gradient magnitude and direction
    G, theta = gradient(smoothed_img)
    # Step 3: Perform Non-Maximum Suppression and Double-Thresholding
    suppressed_img = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(suppressed_img, high, low)
    # Step 4: Apply Edge Tracking to connect strong edges
    edge = link_edges(strong_edges, weak_edges)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    for x, y in zip(xs, ys):
      line = (x * cos_t) + (y * sin_t)
      rhos_indices = np.round(line - rhos[0]).astype(int)

      for i, rho_idx in enumerate(rhos_indices):
            accumulator[rho_idx, i] += 1

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return accumulator, rhos, thetas
