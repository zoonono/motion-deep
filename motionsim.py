import numpy as np
# from matplotlib import pyplot as plt

def perlin_octave(length, min_f, weights = [.5**p for p in range(3)]):
    total = np.zeros(length)
    for w in weights:
        total += w * perlin_noise(length, min_f)
        min_f *= 2
    return total / max(-np.min(total), np.max(total))

def perlin_noise(length, freq):
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack((p, p)).flatten()
    
    def perlin(x):
        xi = int(x)
        n1 = gradient(p[xi % 256], x - xi)
        n2 = gradient(p[(xi + 1) % 256], x - xi - 1)
        return lerp(n1, n2, fade(x - xi))
        
    def lerp(a,b,x):
        return a + x * (b-a)
    
    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3
    
    def gradient(h,dx):
        dx = dx * (1.0 + h % 8)
        if h % 2 == 0:
            return dx
        else:
            return -dx
    
    points = np.linspace(0, freq, num = length)
    return np.array([perlin(p) for p in points])

def simulate_motion(img, dx, dy, dz, axes = 0):
    """
    The order of axes matters: it is the order pts/lines/planes are taken.
    Example: Suppose img is rows x cols and points are taken left to right, 
        then top to bottom (like a book). Then, axes = (1, 0).
    Close to this Matlab library: https://github.com/dgallichan/retroMoCoBox
    """
    if not isinstance(axes, (tuple, list, np.ndarray)):
        axes = (axes,)
    
    # get helper data structures
    shape = np.array(img.shape)
    grid = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    
    # Create list of coordinates sampled wrt time
    # Shape: (T x (1, 2, or 3, the number of fixed axes))
    # Example: if we sample planes in the y direction, time is (T x 1)
    if len(axes) == 1:
        coords = np.mgrid[0:shape[axes[0]]]
    elif len(axes) == 2:
        coords = np.mgrid[0:shape[axes[0]], 0:shape[axes[1]]]
    elif len(axes) == 3:
        coords = np.mgrid[0:shape[axes[0]], 0:shape[axes[1]], 0:shape[axes[2]]]
    else:
        assert False, 'img should be 3D, so axes can maximally be length 3'
    coords = coords.reshape(len(axes), coords.size // len(axes)).transpose(1,0)
    
    img_shift = np.zeros(img.shape, dtype = np.complex64)
    for i, pt in enumerate(coords):
        # get shifts
        dr = np.array([dx[i], dy[i], dz[i]])
        
        # get all points that were sampled at t = i
        if len(axes) == 1:
            idx = np.index_exp[grid[axes[0]] == pt]
        elif len(axes) == 2:
            idx = np.index_exp[(grid[axes[0]] == pt[0]) &
                               (grid[axes[1]] == pt[1])]
        else:
            idx = np.index_exp[(grid[axes[0]] == pt[0]) &
                               (grid[axes[1]] == pt[1]) &
                               (grid[axes[2]] == pt[2])]
        pts = np.vstack((grid[0][idx], grid[1][idx], grid[2][idx]))
        
        # apply translation to image, rotation to k-space sampling points
        img_shift[idx] = img[idx] * np.exp(-2j * np.pi * np.dot(dr, pts))
    
    return img_shift

def motion_PD(img, axes = 0):
    if not isinstance(axes, (tuple, list, np.ndarray)):
        axes = (axes,)
    length = int(np.prod([s for i, s in enumerate(img.shape) if i in axes]))
    
    img = np.fft.fftn(img)
    movement_f = length * 2
    dx = .1 * perlin_octave(length, movement_f) / length
    dy = .1 * perlin_octave(length, movement_f) / length
    dz = .1 * perlin_octave(length, movement_f) / length
    img = simulate_motion(img, dx, dy, dz, axes = axes)
    return np.fft.ifftn(img)