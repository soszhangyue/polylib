import numpy as np
import math


def binom_coef(n, k):
    """
    :param n: order
    :param k: k
    :return: binomial coefficient
    """
    result = 1
    for i in range(n - k + 1, n + 1):
        result *= i
    for i in range(1, k + 1):
        result /= i
    return result


def lin_as_mat(start: int, end: int, gap=1, int_type=False):
    """
    return array as matlab
    """
    if end < start:
        if int_type:
            return np.zeros([1, 0], dtype=int)
        else:
            return np.zeros([1, 0])
    elif end < start + gap:
        if int_type:
            return (start * np.ones([1, 1], dtype=int)).reshape([1, 1])
        else:
            return (start * np.ones([1, 1])).reshape([1, 1])
    elif end >= start + gap:
        array_length = int((end - start) / gap) + 1
        if int_type:
            return (np.linspace(start, start + gap * (array_length - 1), num=array_length, endpoint=True,
                                dtype=int)).reshape([1, array_length])
        else:
            return (np.linspace(start, start + gap * (array_length - 1), num=array_length, endpoint=True)).reshape(
                [1, array_length])


def index_p1(p, k):
    """
    This routine takes a row vector p and generates p1 whose entries
    are all the combination of those of rows of p plus one.
    """
    p = p.reshape(1, p.size)
    d = p.shape[1]
    row = d - k + 1
    p1 = np.zeros((row, d))

    for index in range(k, d + 1, 1):
        row = index - k + 1
        p1[row - 1, :] = p
        p1[row - 1, index - 1] = p[0, index - 1] + 1

    return p1


def index_step1(p: np.ndarray):
    """
    This routine takes a matrix p and generates matrix p1 whose entries
    are all the combination of those of rows of p plus one.
    """
    [row, col] = p.shape
    p1 = index_p1(p[0, :], 1)
    for index in range(2, row + 1, 1):
        dp = index_p1(p[index - 1, :], 1)

        rr = dp.shape[0]
        r0 = p1.shape[0]
        for index_k in range(1, rr + 1, 1):
            noadd = 0
            for index_m in range(1, r0 + 1, 1):
                if (dp[index_k - 1, :] == p1[index_m - 1, :]).all():
                    noadd = 1
                    break
            if noadd == 0:
                p1 = np.vstack((p1, dp[index_k - 1, :]))
    return p1


def index_step(p, n):
    p1 = index_step1(p)
    pn = np.vstack((p, p1))
    for index in range(2, n + 1, 1):
        ptmp = p1
        p1 = index_step1(ptmp)
        pn = np.vstack((pn, p1))

    return pn


def chaos_sequence(ndim, p):
    if ndim <= 0:
        print('Invalid dimensionality! (need n>0)\n')
    else:
        q = index_step(np.zeros([1, ndim]), p)

    return q


if __name__ == "__main__":
    p = lin_as_mat(1, 10, 1)
    p1 = index_p1(p, 3)
    chaos = chaos_sequence(2, 2)
    print(chaos)
    # print(p,p.shape)
