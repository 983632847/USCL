from __future__ import print_function
import random
import numpy as np
import torch
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=1000)
    if random.random() < 0.5:
        # Half chance to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals) # x => nonlinear_x, interpolate to curve [xvals, yvals]
    return nonlinear_x


class NonlinearTrans(object):
    """Randomly do nonlinear transformation on an image.
    Args:
        prob (float): Probability to do the transformation.
    """
    def __init__(self, prob=0.9):
        self.prob = prob

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W), values are between 0 and 1
        Returns:
            img (Tensor): Tensor image of size (C, H, W) after transformation
        """
        out_img = nonlinear_transformation(img, prob=self.prob)
        try:
            # if transform, the dtype would change from Tensor to ndarray
            out_img = torch.from_numpy(out_img)
        except:
            # but the img may also remain the origin, still Tensor
            pass
        # out_img = out_img.double()

        out_img = out_img.type(torch.FloatTensor)

        return out_img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, 100)
    for i in range(5):
        y = nonlinear_transformation(x, prob=1.0)
        plt.plot(x, y)
        plt.show()




    
    
    

    
