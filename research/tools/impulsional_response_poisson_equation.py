import numpy as np
from scipy.special import kv
from scipy.special import iv
import matplotlib.pyplot as plt


"""
author : maxime.dieudonne@univ-amu.fr
date : 24/07/2023

This code is a test to plot the green's function 2D of a screened poisson equation as detailed in
the following article :
Rohan Sawhney∗, Dario Seyb∗, Wojciech Jarosz†, and Keenan Crane†. 2022.
Grid-Free Monte Carlo for PDEs with Spatially Varying Coefficients. ACM
Trans. Graph. 41, 4, Article 53 (July 2022), 17 pages. https://doi.org/10.1145/
3528223.3530134
"""


# functions
def poissonG2D(R, alpha, x):
    G2D = (1 / 2 * np.pi) * ( kv(0, x * np.sqrt(alpha)) -
                              (kv(0, R * np.sqrt(alpha)) / iv(0, R * np.sqrt(alpha))) * iv(0, x * np.sqrt(alpha)))
    return G2D

def poissonG2D_weak(eps,x):
    weak = np.exp(-0.5 * (np.square(x) / eps)) / (eps * np.sqrt(2 * np.pi))
    return weak

# example
if __name__ == '__main__':
    # set x axis
    x = np.linspace(-1, 1, 1000)
    # set parameter of the green function
    R = 1
    alpha = 10
    # set parameter of the weak expression
    eps = 0.00122 / 2
    # call functions
    G2D =  poissonG2D(R,alpha,x)
    weak = poissonG2D_weak(eps,x)

    # figure 1
    fig, ax = plt.subplots()
    ax.plot(x, G2D)
    ax.set_xlabel('x')
    ax.set_ylabel('Green function')
    ax.set_title('Green function of screened poisson equation')

    # figure 2
    fig2, ax2 = plt.subplots()
    ax2.plot(x, weak)
    ax2.set_xlabel('x')
    ax2.set_ylabel('weak expression')
    ax2.set_title('weak expression of the Green function of screened poisson equation')


    plt.show()