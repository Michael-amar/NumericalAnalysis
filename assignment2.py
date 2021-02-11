"""
In this assignment you should find the intersection points for two functions.
"""
import numpy as np
import time
import random
from collections.abc import Iterable

def derivative(f,x):    
    h = 1e-8
    return (f(x+h)-f(x))/h

def includes_point(roots,root,epsilon):
    for i in roots:
        if abs(root-i) < epsilon:
            return True
    return False

def NewtonRaphson(f,Xprev,err,timeout):
    initial_time = time.time()
    while (time.time() - initial_time) < timeout :
        der = derivative(f,Xprev)
        if abs(der) > 0.000001 :
            Xnext = Xprev - ((f(Xprev))/(der))
        else :
            return "divide by zero error"
        if abs(f(Xnext)) < err:
            return Xnext
        Xprev = Xnext
    return "timeout"


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        roots = []
        if a <= b:
            min = a
            max = b
        else:
            min = b
            max = a
        f = lambda x : (f1(x)-f2(x))
        n = 25 + (((max-min)//10)*15 )              #longer range ==> more points
        initial_guess = np.linspace(min,max,n)
        for i in range (0,n):
            try:
                root = NewtonRaphson(f,initial_guess[i],maxerr,timeout=1.5)
                if type(root) != str :
                    if min <= root <= max:
                        if not includes_point(roots,root,0.05):
                            roots.append(root)
            except Exception:
                pass
        return roots
##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

if __name__ == "__main__":
    # f = lambda x : (x**3) + (2*(x**2)) -0.5
    # print(NewtonRaphson(f,0.5,0.001,timeout=10))

    # f1 = lambda x: 1-(2*(x**2))+(x**3)
    # f2 = lambda x: x
    # ass2=Assignment2()
    # print(ass2.intersections(f1,f2,-5,100,0.001))

    unittest.main()

