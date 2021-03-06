import numpy as np
import math
from gPCBasic import *
from scipy.special import gamma


def HermiteCoef(order):
    """
    order: order of the Hermite polynomial

    MARK: double overflow warning !!!
    """

    coef = np.zeros([1, order + 1])
    for index_coef in range(int(np.round((order - 0.1) / 2)) + 1):
        coef[0, order - 2 * index_coef] = np.power((-1), index_coef) / np.power(2, index_coef) * binom_coef(order,
                                                                                                            index_coef) * np.prod(
            lin_as_mat((order - 2 * index_coef + 1), (order - index_coef)))

    return np.fliplr(coef)


def HermiteCoef_t(order, t):
    """
    order: order of the Hermite polynomial

    """

    coef = HermiteCoef(order)
    cnt = 0
    for index in range(1, order + 1 + 2, 2):
        coef[0, index - 1] = coef[0, index - 1] * np.power(t, cnt)
        cnt = cnt + 1
    return coef


def HermiteF(x, degree):
    """
    Compute the Hermite polynomial H^n(x)
    """
    if degree == 0:
        return np.ones(np.size(x))
    elif degree == 1:
        return x
    else:
        return x * HermiteF(x, degree - 1) - (degree - 1) * HermiteF(x, degree - 2)


def HermiteF_nd(x: np.ndarray, ndim, norder):
    ntmp = x.shape[1]

    if ntmp != ndim:
        print('Incompatible dimension in HermiteF_nd! Quit.\n')
    pmatrix = chaos_sequence(ndim, norder)
    ptmp = np.zeros([norder + 1, ndim])

    for index in range(0, norder + 1, 1):
        ptmp[index, :] = HermiteF(x, index)
    nterm = pmatrix.shape[0]
    poly = np.ones([1, nterm])

    for index_m in range(2, nterm + 1, 1):
        for index_n in range(1, ndim + 1, 1):
            poly[0, index_m - 1] = poly[0, index_m - 1] * ptmp[int(pmatrix[index_m - 1, index_n - 1]), index_n - 1]

    return poly


def HermiteD(x, degree):
    if degree == 0:
        return np.zeros(np.size(x))
    elif degree == 1:
        return np.ones(np.size(x))
    else:
        return degree * HermiteF(x, degree - 1)


def HermiteD_nd(x, ndim, norder, nder):
    ntmp = x.shape[1]

    if ntmp != ndim:
        print('Incompatible dimension in HermiteD_nd! Quit.\n')
    pmatrix = chaos_sequence(ndim, norder)
    ptmp = np.zeros([norder + 1, ndim])
    pder = np.zeros([norder + 1, ndim])
    for index in range(0, norder + 1, 1):
        ptmp[index, :] = HermiteF(x, index)
        pder[index, :] = HermiteD(x, index)

    nterm = pmatrix.shape[0]
    poly = np.ones([1, nterm])
    poly[0, 0] = 1

    for index_m in range(2, nterm + 1, 1):
        for index_n in range(1, ndim + 1, 1):
            if index_n == nder:
                poly[0, index_m - 1] = poly[0, index_m - 1] * pder[int(pmatrix[index_m - 1, index_n - 1]), index_n - 1]
            else:
                poly[0, index_m - 1] = poly[0, index_m - 1] * ptmp[int(pmatrix[index_m - 1, index_n - 1]), index_n - 1]

    return poly


def HermiteRing_1d(s, l):
    c = np.zeros([s + l + 1, 1])
    for index in range(np.abs(s - l), (s + l) + 1, 1):
        g = (l + s + r) / 2
        if np.abs(g - np.round(g)) < 0.0001:
            c[index + 1, 1] = gamma(s + 1) * gamma(l + 1) / (gamma(g - l + 1) * gamma(g - s + 1) * gamma(g - index + 1))
    return c


def HermiteZeros(degree):
    z = np.roots(HermiteCoef(degree))
    return z


def HermiteZeros_t(degree, t):
    z = np.roots(HermiteCoef_t(degree, t))
    return z

def HermiteZeros_direct(degree):
    """
    same as HermiteZeros
    :param degree: degree
    :return: zeros as array
    """
    return HermiteZeros(degree)

def HermiteZeros_iter(degree):
    """
    recommend use HermiteZeros.
    :param degree:
    :return:
    """
    maxit = 50
    EPS = 1.0e-14
    dth = np.pi / degree

    rlast = 0

    if degree <= 0:
        return
    else:
        z = np.zeros((degree, 1))

    for k in range(1, degree + 1, 1):
        r = -np.cos((2 * (k - 1) + 1) * dth)
        if k != 1:
            r = (r + rlast) / 2
        for j in range(1, maxit + 1, 1):
            poly = HermiteF(r, degree)
            pder = HermiteD(r, degree)

            delr = - poly / (pder - sum(1. / (r - z[1:k-1])) * poly) # this need check later

            r = r + delr

            if np.abs(delr) < EPS:
                break

            if j == maxit:
                print('Maximum number of iteration reached in function: HermiteZeros\n')
        z[k] = r
        rlast = r
    return z

if __name__ == "__main__":
    print(HermiteCoef(3))
