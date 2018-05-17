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

def rotate(coord, dphi, dtheta, dpsi, center):
    rotphi = np.array([[1, 0, 0],
                       [0, np.cos(dphi), -np.sin(dphi)],
                       [0, np.sin(dphi), np.cos(dphi)]])
    rottheta = np.array([[np.cos(dtheta), 0, np.sin(dtheta)],
                         [0, 1, 0],
                         [-np.sin(dtheta), 0, np.cos(dtheta)]])
    rotpsi = np.array([[np.cos(dpsi), -np.sin(dpsi), 0],
                       [np.sin(dpsi), np.cos(dpsi), 0],
                       [0, 0, 1]])
    return rotphi.dot(rottheta.dot(rotpsi.dot(coord - center))) + center

def translate(coord, dx, dy, dz):
    return coord + np.array([dx, dy, dz])

def planar_sampling(img, dx, dy, dz, dphi, dtheta, dpsi, axis = 0):
    center = img.shape // 2
    img_k = np.zeros(img.shape)
    points = np.ndindex(img.shape[0], img.shape[1], img.shape[2])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if axis == 0:
                    t = i
                elif axis == 1:
                    t = j
                else:
                    t = k
                t1 = lambda p: rotate(p, dphi[t], dtheta[t], dpsi[t], center)
                t2 = lambda p: translate(p, dx[t], dy[t], dz[t])
                t = lambda p: t2(t1(p)).astype(int)
                f = lambda p: img[p[0] % img.shape[0], 
                                  p[1] % img.shape[1], 
                                  p[2] % img.shape[2]]
                img_k[i,j,k] = nudft(points, f, i, j, k, t = t)
    return img_k
    
def nudft(points, function, kx, ky, kz, t = lambda x: x):
    total = 0
    for p in points:
        p = t(p)
        total += function(p) * np.exp(-1j * 2 * np.pi * 
                         (p[0] * kx + p[1] * ky + p[2] * kz))
    return total

# plt.plot(perlin_octave(256, 32))