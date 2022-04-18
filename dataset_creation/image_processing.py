from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
import numpy as np


def interpolation(x, y, z, shape):
    X = np.linspace(min(x), max(x), num=shape[1])
    Y = np.linspace(min(y), max(y), num=shape[0])
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    z = 1./z
    interp = NearestNDInterpolator(list(zip(x, y)), z)
    Z = interp(X, Y)
    Y = Y[::-1]

    return X, Y, Z

def plot(X, Y, Z):
    plt.pcolormesh(X, Y, Z, shading='auto')
    # plt.plot(x, y, "ok", label="input point")
    # plt.legend()
    plt.colorbar()
    plt.axis("equal")
    plt.show()
    # plt.imshow(Z)
    # plt.show()

def get_depth_gradient(z, axis=0):
    if axis == 0:
        extra_z_row = [z[-1, :]]
        expanded_z = np.r_[z, extra_z_row]
        depth_gradient = np.diff(expanded_z, axis=axis)
    elif axis == 1:
        extra_z_column = z[:, -1]
        expanded_z = np.c_[z, extra_z_column]
        depth_gradient = np.diff(expanded_z, axis=axis)
    else:
        raise Exception()
    return depth_gradient

# rng = np.random.default_rng()
# x = rng.random(10) - 0.5
# y = rng.random(10) - 0.5
# z = np.hypot(x, y)
# X = np.linspace(min(x), max(x))
# Y = np.linspace(min(y), max(y))
# X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
# interp = NearestNDInterpolator(list(zip(x, y)), z)
# Z = interp(X, Y)
# # Z = np.hypot(X, Y)
# plt.pcolormesh(X, Y, Z, shading='auto')
# plt.plot(x, y, "ok", label="input point")
# plt.legend()
# plt.colorbar()
# plt.axis("equal")
# plt.show()
