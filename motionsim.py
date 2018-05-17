import numpy as np
import nufft
from matplotlib import pyplot as plt
import time

def perlin_octave(length, min_f, weights = [.6**p for p in range(3)]):
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

def rotation_matrix(dphi, dtheta, dpsi):
    """Returns the rotation matrix from Euler rotation angles."""
    rotphi = np.array([[1, 0, 0],
                       [0, np.cos(dphi), -np.sin(dphi)],
                       [0, np.sin(dphi), np.cos(dphi)]])
    rottheta = np.array([[np.cos(dtheta), 0, np.sin(dtheta)],
                         [0, 1, 0],
                         [-np.sin(dtheta), 0, np.cos(dtheta)]])
    rotpsi = np.array([[np.cos(dpsi), -np.sin(dpsi), 0],
                       [np.sin(dpsi), np.cos(dpsi), 0],
                       [0, 0, 1]])
    return rotphi.dot(rottheta.dot(rotpsi))

def simulate_motion(img, dx, dy, dz, dphi, dtheta, dpsi, axes = 0):
    """
    The order of axes matters: it is the order pts/lines/planes are taken.
    Example: Suppose img is rows x cols and points are taken left to right, 
        then top to bottom (like a book). Then, axes = (1, 0).
    Close to this Matlab library: https://github.com/dgallichan/retroMoCoBox
    """
    start = time.time()
    if not isinstance(axes, (list, np.ndarray)):
        axes = (axes,)
    
    # get helper data structures
    shape = np.array(img.shape)
    grid = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    center = np.tile((shape // 2), (img.size // dx.size, 1)).transpose(1, 0)
    
    # Create list of coordinates sampled wrt time
    # Shape: (T x (1, 2, or 3, the number of fixed axes))
    # Example: if we sample planes in the y direction, time is (T x 1)
    if len(axes) == 1:
        samp_pts = np.mgrid[0:shape[axes[0]]]
    elif len(axes) == 2:
        samp_pts = np.mgrid[0:shape[axes[0]], 0:shape[axes[1]]]
    elif len(axes) == 3:
        samp_pts = np.mgrid[0:shape[axes[0]], 0:shape[axes[1]], 0:shape[axes[2]]]
    else:
        assert False, 'img should be 3D, so axes can maximally be length 3'
    samp_pts = samp_pts.reshape(len(axes), 
                                samp_pts.size // len(axes)).transpose(1, 0)
    
    points = None
    for i, pt in enumerate(samp_pts):
        start2 = time.time()
        # get shifts
        rotation = rotation_matrix(dphi[i], dtheta[i], dpsi[i])
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
        img[idx] *= np.exp(-2j * np.pi * np.dot(dr, pts))
        pts = rotation.dot(pts - center) + center
        if points is None:
            points = pts
        else:
            points = np.hstack((points, pts))
        print(time.time() - start2)
    points = points.astype(int)
    points[0] = np.mod(points[0], shape[0])
    points[1] = np.mod(points[1], shape[1])
    points[2] = np.mod(points[2], shape[2])
    
    vals = np.array([img[p[0], p[1], p[2]] for p in points.transpose(1, 0)])
    print(time.time() - start)
    start = time.time()
    img_k =  nufft.nufft3d1(points[0], points[1], points[2], vals, 
                          shape[0], shape[1], shape[2])
    print(time.time() - start)
    return img_k

def test_simulate_motion(img, axes = 0):
    if not isinstance(axes, (list, np.ndarray)):
        axes = (axes,)
    length = int(np.prod([s for i, s in enumerate(img.shape) if i in axes]))
    dx = 4 * perlin_octave(length, 32)
    dy = 4 * perlin_octave(length, 32)
    dz = 4 * perlin_octave(length, 32)
    dphi = perlin_octave(length, 32)
    dtheta = perlin_octave(length, 32)
    dpsi = perlin_octave(length, 32)
    return simulate_motion(img, dx, dy, dz, dphi, dtheta, dpsi, axes = axes)

# plt.plot(perlin_octave(256, 32))