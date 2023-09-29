import math
import matplotlib.pyplot as plt
import numpy as np

#parâmetros
wl = 1
z = 1
w0 = 1
m = 1
n = 1

#parâmetros calculáveis
k = 2 * np.pi / wl
Nmn = np.sqrt(2 ** (1 - (m + n))) / (np.sqrt(np.pi) * math.factorial(m) * math.factorial(n))
zr = np.pi * w0 ** 2 / wl
rz = z * (1 + (zr / z) ** 2)
wz = w0 * np.sqrt(1 + (z ** 2 / zr ** 2))
phase = (m + n + 1) * np.arctan(z / zr)

#polinômios de hermite
Hm = np.polynomial.hermite.Hermite([0] * m + [1])
Hn = np.polynomial.hermite.Hermite([0] * n + [1])


def hermite_beam(x0, y0):
    hx = np.sqrt(2) * x0 / w0
    hy = np.sqrt(2) * y0 / w0
    return np.abs(Nmn * Hm(hx) * Hn(hy) * np.exp(-(x0 ** 2 + y0 ** 2) / wz ** 2) * np.exp(
        -1j * (k * (x0 ** 2 + y0 ** 2)) / 2 * rz - 1j * phase)) ** 2


def hermite_beam_z0(x0, y0):
    hx = np.sqrt(2) * x0 / w0
    hy = np.sqrt(2) * y0 / w0
    return np.abs(Nmn * Hm(hx) * Hn(hy) * np.exp(-(x0 ** 2 + y0 ** 2) / w0 ** 2)) ** 2


X = np.linspace(-wz * 5, wz * 5, 1000)
Y = np.linspace(-wz * 5, wz * 5, 1000)
x, y = np.meshgrid(X, Y)

plt.imshow(hermite_beam(x, y))
plt.show()
