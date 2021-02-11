"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random

def midpoint(f,a,b):
    # Note: cant take more than 1 sample
    h = b - a
    x0 = (a+b)/2
    return np.float32(h*f(x0))

def trapezoidal(f,a,b):
    # Note: cant take more than 2 samples
    h = b - a
    return np.float32((h/2)*(f(a)+f(b)))

def simpsons(f,a,b):
    # Note: cant take more than 3 samples
    h = (b - a) / 2
    x0 = a
    x1 = a + h
    x2 = b
    return np.float32(
        (h/3)*(f(x0) + (4*f(x1)) + f(x2))
    )

def simpsonsX(f,a,b):
    # Note: cant take more than 5 samples
    h = (b-a)/4
    x0 = a
    x1 = x0 + h
    x2 = x0 + (2*h)
    x3 = x0 + (3*h)
    x4 = x0 + (4*h)
    return np.float32(
        ((2*h)/45)* ( (7*f(x0)) + (32*f(x1)) + (12*f(x2)) + (32*f(x3)) + (7*f(x4)) )
    )

def composite_simpsonX(f,a,b,n):
    #Note: cant take more than n samples
    h = (b-a)/(n-1)
    if n%4 == 0 : return np.float32(trapezoidal(f,a,a+h) + simpsons(f,a+h,a+(3*h)) + composite_simpsonX(f,a+(3*h),b,n-3))
    elif n%4 == 1:
        F0=f(a)
        F1=0
        F2=0
        F3=0
        F4=f(b)
        for i in range(1,n-1):
            x = a + (i*h)
            if i%4 ==0:
                F0 += f(x)
                F4 += f(x)
            elif i%4 ==1: F1+= f(x)
            elif i%4 ==2: F2+= f(x)
            elif i%4 ==3: F3+= f(x)
        return np.float32(
            ((2*h)/45)*((7*F0) + (32*F1) + (12*F2) + (32*F3) + (7*F4))
        )
    elif n%4 == 2: return np.float32(trapezoidal(f,a,a+h)+composite_simpsonX(f,a+h,b,n-1))
    elif n%4 == 3: return np.float32(simpsons(f,a,a+(2*h)) + composite_simpsonX(f,a+(2*h),b,n-2))


def composite_simpson(f,a,b,n):
    #Note: cant take more than n samples
    h = (b-a)/(n-1)
    if n%2 == 0:
        return np.float32(trapezoidal(f,a,a+h)+composite_simpson(f,a+h,b,n-1))
    else :
        F0 = f(a)
        F1 = 0
        F2 = f(b)
        for i in range(1,n-1):
            x = a + (i*h)
            if i%2 == 0 :
                fx = f(x)
                F2 += fx
                F0 += fx
            else :
                F1 += f(x)
        return np.float32(
            (h/3)*(F0+(4*F1)+F2)
        )

class FunctionCache:
    def __init__(self,f):
        self.f = f
        self.cache = {}
    
    def get(self,x):
        if x in self.cache:
            return self.cache[x]
        else :  
            fx = self.f(x)
            self.cache[x]=fx
            return fx

class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """
        g = FunctionCache(f)
        f = lambda x : g.get(x)

        # if compositeX(deg=4) is better than composite simpson(deg=2)
        if n == 0 : return np.float32(0)
        elif n == 1 : return midpoint(f,a,b)
        elif n == 2 : return trapezoidal(f,a,b)
        elif n == 3 : return simpsons(f,a,b)
        elif n == 4 : return composite_simpson(f,a,b,4)
        elif n == 5 : return simpsonsX(f,a,b)
        else : return composite_simpsonX(f,a,b,n)

        # #if composite simpson(deg=2) is better than compositeX(deg=4)
        # if n == 0 : return np.float32(0)
        # elif n == 1 : return midpoint(f,a,b)
        # elif n == 2 : return trapezoidal(f,a,b)
        # elif n == 3 : return simpsons(f,a,b)
        # else : return composite_simpson(f,a,b,n) 

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        # replace this line with your solution
        result = np.float32(1.0)

        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 100000)
        print("res:",r)
        true_result = -7.78662 * 10 ** 33
        print("true:",true_result)
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

if __name__ == "__main__":
    # unittest.main()
    # f = lambda x: (x**3) - (42*(x**2)) + (38*x) - 5
    # simpson_x = simpsonsX(f,0.3,47.2)
    # simpsons_x_comp = composite_simpsonX(f,0.3,47.2,5)
    # print ("simpson_x" , simpson_x)
    # print ("simpsons_x_comp" , simpsons_x_comp)
    f = lambda x: np.arctan(x)
    ass3 = Assignment3()
    print(ass3.integrate(h,1,100,16))
    # print(simpsonsX(f,1,100))
