import numpy as np


def newton_force(m1, m2, r):
    G = 6.67430e-11
    return G * m1 * m2 / r**2

def newton_acc(m,r):
    G = 6.67430e-11
    return G * m / r**2

class System():
    def __init__(masses: np.array, p_init: np.array):
        x=0

    def dpdt(self, ic, t):
        dpdt = 0

        distances = self.calc_distances()

        return dpdt
    
    def calc_distances(self):
        return None
    
    def rk4(self):
        return None
    
    def leapfrog(self):
        return None
    
    def plot(self):
        return None
    
    def Animate(self):
        return None
    
