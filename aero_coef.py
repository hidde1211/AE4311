import numpy as np

def get_true_aero_coef(alpha,M):
    bz = 1.62 * M - 6.72
    bm = 10.81 * M + 51.27
    Cz = -288.7 * alpha ** 3 + 50.32 * alpha * abs(alpha) - 23.89 * alpha + (
                -13.53 * alpha * abs(alpha) + 4.185 * alpha) * M
    Cm = 202.1 * alpha ** 3 - 246.3 * alpha * abs(alpha) - 37.56 * alpha + (
                71.51 * alpha * abs(alpha) + 10.01 * alpha) * M

    return bz,bm,Cz,Cm