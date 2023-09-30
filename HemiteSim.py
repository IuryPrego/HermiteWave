import math
import matplotlib.pyplot as plt
import numpy as np

# parâmetros
wl = 1
z = 1
w0 = 1
m = 0
n = 0

# parâmetros calculáveis
k = 2 * np.pi / wl
Nmn = np.sqrt(2 ** (1 - (m + n))) / (np.sqrt(np.pi) * math.factorial(m) * math.factorial(n))
zr = np.pi * w0 ** 2 / wl
if z != 0:
    rz = z * (1 + (zr / z) ** 2)
wz = w0 * np.sqrt(1 + (z / zr) ** 2)
phase = (m + n + 1) * np.arctan(z / zr)

# polinômios de hermite
Hm = np.polynomial.hermite.Hermite([0] * m + [1])
Hn = np.polynomial.hermite.Hermite([0] * n + [1])


def hermite_beam_no_phase(x0, y0):
    hx = np.sqrt(2) * x0 / wz
    hy = np.sqrt(2) * y0 / wz
    return np.abs(Nmn / wz * Hm(hx) * Hn(hy) * np.exp(-(x0 ** 2 + y0 ** 2) / wz ** 2)) ** 2


def hermite_beam(x0, y0):
    hx = np.sqrt(2) * x0 / wz
    hy = np.sqrt(2) * y0 / wz
    return np.abs(Nmn / wz * Hm(hx) * Hn(hy) * np.exp(-(x0 ** 2 + y0 ** 2) / wz ** 2) * np.exp(
        -1j * (k * (x0 ** 2 + y0 ** 2)) / 2 * rz - 1j * phase)) ** 2


# testes
print((-1j * (k * (1 ** 2 + 1 ** 2)) / 2 * rz))
print((- 1j * phase))
print((-1j * (k * (1 ** 2 + 1 ** 2)) / 2 * rz - 1j * phase))
print(np.abs(np.exp(-1j * (k * (1 ** 2 + 1 ** 2)) / 2 * rz - 1j * phase)))


def hermite_beam_z0(x0, y0):
    hx = np.sqrt(2) * x0 / w0
    hy = np.sqrt(2) * y0 / w0
    return np.abs(Nmn / w0 * Hm(hx) * Hn(hy) * np.exp(-(x0 ** 2 + y0 ** 2) / w0 ** 2)) ** 2


def hb_general_case(x0, y0):
    if z > 0:
        return hermite_beam(x0, y0)
    else:
        return hermite_beam_z0(x0, y0)


X = np.linspace(-wz * 5, wz * 5, 1000)
Y = np.linspace(-wz * 5, wz * 5, 1000)
x, y = np.meshgrid(X, Y)

plt.imshow(hb_general_case(x, y))
plt.show()

# testes
if np.array_equal(hermite_beam_no_phase(x, y), hermite_beam_no_phase(x, y)):
    print('igual')
else:
    print('diferente')
