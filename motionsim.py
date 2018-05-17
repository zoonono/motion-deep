import numpy as np
from matplotlib import pyplot as plt

def perlin_octave(length, min_f, weights = [.6**p for p in range(8)]):
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
    """
    if len(axes) == 1:
        axes = axes[0]
    
    # get helper data structures
    shape = np.array(img.shape)
    grid = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    center = np.tile((shape // 2).T, 
        (1, int(np.prod([s for i, s in enumerate(shape) if i not in axes]))))
    
    # Create list of coordinates sampled wrt time
    # Shape: (T x (1, 2, or 3, the number of fixed axes))
    # Example: if we sample planes in the y direction, time is (T x 1)
    if not isinstance(axes, (list, np.ndarray)):
        time = np.mgrid[0:shape[axes]]
    else:
        if len(axes) == 2:
            time = np.mgrid[0:shape[axes[0]], 0:shape[axes[1]]]
        elif len(axes) == 3:
            time = np.mgrid[0:shape[axes[0]], 0:shape[axes[1]], 
                            0:shape[axes[2]]]
        else:
            assert False, 'img should be 3D, so axes can maximally be length 3'
        time = time.reshape(len(axes), time.size // len(axes)).transpose(1, 0)
    
    points = None
    for i, pt in enumerate(time):
        # get shifts
        rotation = rotation_matrix(dphi[i], dtheta[i], dpsi[i])
        dr = np.array([dx[i], dy[i], dz[i]])
        
        # get all points that were sampled at t = i
        if not isinstance(axes, (list, np.ndarray)):
            idx = np.index_exp[grid[axes] == pt]
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
    
    # 4 neighbors in each direction, 1.5x oversampling in k-space
    return nufft(img, points, shape, (4, 4, 4), 1.5 * shape)

def nudft(points, function, kx, ky, kz, t = lambda x: x):
    total = 0
    for p in points:
        p = t(p)
        total += function(p) * np.exp(-1j * 2 * np.pi * 
                         (p[0] * kx + p[1] * ky + p[2] * kz))
    return total

def test_simulate_motion(img, axes = 0):
    length = int(np.prod([s for i, s in enumerate(img.shape) if i not in axes]))
    dx = 4 * perlin_octave(length, 32)
    dy = 4 * perlin_octave(length, 32)
    dz = 4 * perlin_octave(length, 32)
    dphi = 4 * perlin_octave(length, 32)
    dtheta = 4 * perlin_octave(length, 32)
    dpsi = 4 * perlin_octave(length, 32)
    return simulate_motion(img, dx, dy, dz, dphi, dtheta, dpsi, axes = axes)

# plt.plot(perlin_octave(256, 32))