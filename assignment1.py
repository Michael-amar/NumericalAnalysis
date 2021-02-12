"""
In this assignment you should interpolate the given function.
"""
import torch
import numpy as np
import numpy.polynomial.polynomial as poly
import time
import random

class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        if n == 1:
            const = f(a+((b-a)/2))
            return lambda x : const

        def search_spline(x,splinesRange,nsplines):
            for i in range(0,nsplines):
                if splinesRange[i][0] <= x and splinesRange[i][1] >= x:
                    return i
            return -1

        nsplines = (n-2)//2
        P = (list(map(lambda x : [x,f(x)] , np.linspace(a,b,(nsplines*2+2)))))
        C = torch.Tensor(P)
        splinesRange = np.zeros(shape = (nsplines,2))
        for i in range(0,nsplines): 
            splinesRange[i] = [C[i*2+1][0],C[i*2+3][0]]

        #Bezier curve is a function from [0,1] to (x,y)
        #we need to return functoin from x to y
        #so given x we need to calculate its t and then find its y
        # x(t) = (-X0+3X1-3X2+X3)t^3 + (3X0-6X1+3X2)t^2 + (-3X0+3X1)t + X0
        # all points are given so we can find t using bisection
        #then plug t in the y function and get y coordinate
        # y(t) = (-Y0+3Y1-3Y2+Y3)t^3 + (3Y0-6Y1+3Y2)t^2 + (-3Y0+3Y1)t + Y0

        def find_y(x):
            spline_index = search_spline(x,splinesRange,nsplines)
            if spline_index == -1:
                return "x value out of curve"
            x0,y0 = C[spline_index*2+1]
            x1,y1 = 2*C[spline_index*2+1] - C[spline_index*2]
            x2,y2 = C[spline_index*2+2]
            x3,y3 = C[spline_index*2+3]
            calc_y = lambda t: (-y0 + 3*y1 -3*y2 + y3)*(t**3) + (3*y0 - 6*y1 + 3*y2) *(t**2) + (-3*y0 + 3*y1)*t + y0 
            p = [x0 -x,(-3*x0 + 3*x1),(3*x0 - 6*x1 + 3*x2),(-x0 + 3*x1 -3*x2 + x3)]
            roots = np.polynomial.polynomial.polyroots(p)
            roots = roots[np.isreal(roots)]
            if len(roots) == 0:
                return "failed to find point"
            else:
                desired_t = np.real(roots[0])
                return float(calc_y(desired_t))
        return find_y


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 300
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 300 + 1)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
