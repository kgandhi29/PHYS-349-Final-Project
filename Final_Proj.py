import numpy as np
import copy
import matplotlib.pyplot as plt
import time



class Mass(object):
    """
    Class storing information about a mass, including position, and velocity. 

    Data Attributes:
        mass
        position
        velocity
   
    """
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)



class System():
    def __init__(self, Masses: list[Mass], Natural_units = True, epsilon=0.01):
        self.N = len(Masses)
        self.sys = Masses
        self.epsilon = epsilon

        self.masses = np.array([m.mass for m in Masses])
        self.positions = np.array([m.position for m in Masses])
        self.velocities = np.array([m.velocity for m in Masses])

        if Natural_units:
            self.G = 4 * np.pi * np.pi
        else:
            self.G = 6.67430e-11

    def acc(self, pi, t):

        atot = np.zeros((self.N,3))

        for i in range(self.N):
            for j in range(self.N):
                if j == i:
                    continue
                r_vec = pi[j] - pi[i]
                r = np.linalg.norm(r_vec)
                if r > 0:
                    a_ij = self.G * self.masses[j] * r_vec / (r**2 + self.epsilon**2)**(3/2)
                    atot[i] += a_ij

        return atot
    
    def acc_2(self, pi, t):
        #Does vectorized operations, should be faster for larger N, but slower for small N due to overhead of vectorization
        pos_i = pi[:, np.newaxis, :]  # (N, 1, 3)
        pos_j = pi[np.newaxis, :, :]  # (1, N, 3)
        r_vec = pos_j - pos_i  # (N, N, 3)

        r = np.linalg.norm(r_vec, axis=2)  # (N, N)
        r = np.where(r == 0, np.inf, r)
        acc_matrix = self.G * self.masses[np.newaxis, :, np.newaxis] * r_vec / (r[:, :, np.newaxis]**2 + self.epsilon**2)**(3/2)
        atot = np.sum(acc_matrix, axis=1)  #shape (N, 3)

        return atot
    
    def rk4(self, t, func = None):
        """This function"""
        dt = t[1] - t[0]   
        dpdt = func

        if dpdt is None:
            dpdt = self.acc_2

        x = self.positions
        v = self.velocities

        k1v  = dt * dpdt(x, t[0])
        k1x = dt * v

        k2v = dt * dpdt(x + k1x/2, t[0] + dt/2)
        k2x = dt * (v + k1v/2) 

        k3v = dt * dpdt(x + k2x/2, t[0] + dt/2)
        k3x = dt * (v + k2v/2)

        k4v = dt * dpdt(x + k3x, t[0] + dt)
        k4x = dt * (v + k3v)

        #Allocating space for trajectory
        xtraj = np.zeros((len(t), self.N, 3))
        vtraj = np.zeros((len(t), self.N, 3))
        
        #Setting initial position and velocity
        xtraj[0] = x
        vtraj[0] = v

        for i in range(1, len(t)):
            term2x = 1/6 * (k1x + 2*k2x + 2*k3x + k4x)
            term2v = 1/6 * (k1v + 2*k2v + 2*k3v + k4v)

            x = x + term2x
            v = v + term2v
            
            k1v  = dt * dpdt(x, t[i])
            k1x = dt * v

            k2v = dt * dpdt(x + k1x/2, t[i] + dt/2)
            k2x = dt * (v + k1v/2) 

            k3v = dt * dpdt(x + k2x/2, t[i] + dt/2)
            k3x = dt * (v + k2v/2)

            k4v = dt * dpdt(x + k3x, t[i] + dt)
            k4x = dt * (v + k3v)

            xtraj[i] = x
            vtraj[i] = v

        ptraj = np.concatenate((xtraj, vtraj), axis = 2)
            
        return ptraj
    
    def leapfrog(self, t, func = None):
        """This function"""
        dt = t[1] - t[0]
        x = self.positions
        v = self.velocities 

        dpdt = func

        if dpdt is None:
            dpdt = self.acc_2

        a = dpdt(x, t[0])

        v_half = v + a * dt / 2

        #Allocating space for trajectory
        xtraj = np.zeros((len(t), self.N, 3))
        vtraj = np.zeros((len(t), self.N, 3))

        #Setting initial position and velocity (initial v_half)
        xtraj[0] = x
        vtraj[0] = v_half

        for i in range(1, len(t)):

            x = x + v_half * dt
            a = dpdt(x, t)
            v_half = v_half + a * dt

            xtraj[i] = x
            vtraj[i] = v_half

        ptraj = np.concatenate((xtraj, vtraj), axis = 2)

        return ptraj
    
    def plot(self, traj, elev = 90, azim = -90):

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.set_xlabel('X(AU)')
        ax.set_ylabel('Y(AU)')
        ax.set_zlabel('Z(AU)')
        ax.view_init(elev=elev, azim=azim)

        positions = traj[:,:,:3]

        for i in range(self.N):
            ax.plot(positions[:,i,0], positions[:,i,1], positions[:,i,2])
        plt.show()
          
    def Animate(self):
        return None
    

#units of AU, Solar Masses and Years

#Keplers law leads to G = 4 * pi * pi
sun = Mass(1, np.zeros(3), np.zeros(3))
earth = Mass(3.00274e-6, np.array([1,0,0]), np.array([0,2*np.pi,0]))

#sun = Mass(1.989*10**30, np.zeros(3), np.zeros(3))
#earth = Mass(5.97*10**24, np.array([1.496*10**11,0,0]), np.array([0,29800,0]))


es = System([sun,earth])

#test timing of dpdt2 and dpdt3

ts = time.time()
ptraj = es.rk4(np.linspace(0,1,1000), es.acc)
te = time.time()
print("Rk4 Time for acc: ", te - ts)

ts = time.time()
ptraj3 = es.rk4(np.linspace(0,1,1000), es.acc_2)
te = time.time()
print("Rk4 Time for acc_2: ", te - ts)

ts = time.time()
ptraj2 = es.leapfrog(np.linspace(0,10,1000), es.acc)
te = time.time()
print("Leapfrog Time for acc: ", te - ts)

ts = time.time()
ptraj4 = es.leapfrog(np.linspace(0,10,1000), es.acc_2)
te = time.time()
print("Leapfrog Time for acc_2: ", te - ts)

es.plot(ptraj)

#The x and v returned from the leapfrog method do not agree on time
#   - x is at full step, v is at half step
#   - Either leave it, or average out the half step v to get time agreement with x
#   - Also, the initial velocity is left out (vtraj only store v_half)

# - Animation with matplotlib FuncAnimation.
#   - Dots at end of lines

# - Make more dynamic with units
#   - Adjust labels on plot


