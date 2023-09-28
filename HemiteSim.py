import math
import matplotlib.pyplot as plt
import numpy as np


def hermite_beam(m, n, x, y, z, wl, w0):
    k = 2 * np.pi / wl
    Nmn = np.sqrt(2 ** (1 - (m + n))) / (np.sqrt(np.pi) * math.factorial(m) * math.factorial(n))
    zr = np.pi * w0 ** 2 / wl
    rz = z * (1 + (zr / z) ** 2)
    wz = w0 * np.sqrt(1 + (z / zr) ** 2)
    phase = (m + n + 1) * np.arctan(z / zr)
    Hm = np.polynomial.hermite.Hermite([0] * m + [1])
    Hn = np.polynomial.hermite.Hermite([0] * n + [1])
    hx = np.sqrt(2) * x / wz
    hy = np.sqrt(2) * y / wz

    return np.abs(Nmn * Hm(hx) * Hn(hy) * np.exp(-(x ** 2 + y ** 2) / wz) * np.exp(
        -1j * (k * (x ** 2 + y ** 2)) / 2 * rz - 1j * phase)) ** 2


# parâmetros
wl = 10
z = 1
w0 = 1
m = 0
n = 0

X = np.linspace(-5, 5, 1000)
Y = np.linspace(-5, 5, 1000)
x, y = np.meshgrid(X, Y)

plt.imshow(hermite_beam(m, n, x, y, z, wl, w0))
plt.show()
