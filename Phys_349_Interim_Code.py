from matplotlib import animation
import numpy as np
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
    """Class storing information about a system of masses, and methods to calculate trajectories, plot, and animate."""
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

        self.trajectory = None

    def acc(self, X, t):
        """This function calculates acceleration using Newtons law of gravitation, given the positions, X of N masses"""

        atot = np.zeros((self.N,3))

        for i in range(self.N):

            for j in range(self.N):
                if j == i:
                    continue

                r_vec = X[j] - X[i] #Calculate displacement between mass i and j
                r = np.linalg.norm(r_vec)

                if r > 0: #epsilon should stop this from breaking (may be redundant)

                    #acceleration of mass i due to mass j
                    a_ij = self.G * self.masses[j] * r_vec / (r**2 + self.epsilon**2)**(3/2)

                    atot[i] += a_ij

        return atot
    
    def acc_2(self, X, t):
        """This function calculates acceleration using Newtons law of gravitation, given the positions, X of N masses"""
        #Does vectorized operations, should be faster for larger N, but slower for small N due to overhead of vectorization
        
        X_i = X[:, np.newaxis, :]  # (N, 1, 3)
        X_j = X[np.newaxis, :, :]  # (1, N, 3)
        r_vec = X_j - X_i  # (N, N, 3)

        r = np.linalg.norm(r_vec, axis=2)  # (N, N)
        r = np.where(r == 0, np.inf, r)
        acc_matrix = self.G * self.masses[np.newaxis, :, np.newaxis] * r_vec / (r[:, :, np.newaxis]**2 + self.epsilon**2)**(3/2)
        atot = np.sum(acc_matrix, axis=1)  #shape (N, 3)

        return atot
    
    def rk4(self, t, acc_func = None):
        """This function solves the motion of a system of masses using the Runge-Kutta 4th order method"""
        dt = t[1] - t[0]   

        if acc_func is None:
            #Currently have 2 acceleration functions. Can choose or defaults to acc_2
            #may remove later if we go with only one acc function
            acc_func = self.acc_2

        x = self.positions
        v = self.velocities

        k1v  = dt * acc_func(x, t[0])
        k1x = dt * v

        k2v = dt * acc_func(x + k1x/2, t[0] + dt/2)
        k2x = dt * (v + k1v/2) 

        k3v = dt * acc_func(x + k2x/2, t[0] + dt/2)
        k3x = dt * (v + k2v/2)

        k4v = dt * acc_func(x + k3x, t[0] + dt)
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
            
            k1v  = dt * acc_func(x, t[i])
            k1x = dt * v

            k2v = dt * acc_func(x + k1x/2, t[i] + dt/2)
            k2x = dt * (v + k1v/2) 

            k3v = dt * acc_func(x + k2x/2, t[i] + dt/2)
            k3x = dt * (v + k2v/2)

            k4v = dt * acc_func(x + k3x, t[i] + dt)
            k4x = dt * (v + k3v)

            xtraj[i] = x
            vtraj[i] = v

        #Combine into a single array of shape (len(t), N, 6)
        ptraj = np.concatenate((xtraj, vtraj), axis = 2)
        self.trajectory = ptraj

        return ptraj
    
    def leapfrog(self, t, acc_func = None):
        """This function solves the motion of a system of masses using the leapfrog method."""
        dt = t[1] - t[0]

        x = self.positions
        v = self.velocities 

        if acc_func is None:
            #Currently have 2 acceleration functions, can choose, or defaults to acc_2
            #may remove later if we go with only one acc function
            acc_func = self.acc_2

        a = acc_func(x, t[0])

        #first half step for velocity, using acceleration at initial positions
        v_half = v + a * dt / 2

        #Allocating space for trajectory
        xtraj = np.zeros((len(t), self.N, 3))
        vtraj = np.zeros((len(t), self.N, 3))

        #Setting initial position and velocity (initial v_half)
        xtraj[0] = x
        vtraj[0] = v_half

        for i in range(1, len(t)):

            x = x + v_half * dt

            a = acc_func(x, t)
        
            v_half = v_half + a * dt

            xtraj[i] = x
            vtraj[i] = v_half

        #Combine into a single array of shape (len(t), N, 6)
        ptraj = np.concatenate((xtraj, vtraj), axis = 2)

        self.trajectory = ptraj

        return ptraj
    
    def plot(self, elev = 90, azim = -90):
        """This function plots a trajectory using matplotlib in 3d. The view angle can be altered using elev and azim input parameters"""
        if self.trajectory is None:
            #may just run one of them automatically later, instead of raising an error
            raise ValueError("No trajectory found. Run rk4 or leapfrog to generate trajectory before plotting.")
        
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.set_xlabel('X(AU)')
        ax.set_ylabel('Y(AU)')
        ax.set_zlabel('Z(AU)')
        ax.view_init(elev=elev, azim=azim)

        positions = self.trajectory[:,:,:3]

        for i in range(self.N):
            ax.plot(positions[:,i,0], positions[:,i,1], positions[:,i,2])
        plt.show()

    def _init_animate(self):
        """This function sets the initial objects to be animated"""
        for line, pt in zip(self.lines, self.pts):
            line.set_data([], [])
            line.set_3d_properties([])

            pt.set_data([], [])
            pt.set_3d_properties([])

        return self.lines + self.pts
    
    def _Animate(self, itr):
        """This function draws lines and points for the masses in the system up to index itr."""
        #shape (tmax, N, 3)
        positions = self.trajectory[:,:,:3]

        #shape (N, tmax, 3)
        positions = positions.transpose(1, 0, 2)

        for line, pt, pi in zip(self.lines, self.pts, positions):
            
            #Not sure if I need to transpose?
            line.set_data(pi[:itr,0].T, pi[:itr,1].T)
            line.set_3d_properties(pi[:itr,2].T)

            pt.set_data([pi[itr,0].T], [pi[itr,1].T])
            pt.set_3d_properties([pi[itr,2].T])

            #line.set_data(pi[:itr,0], pi[:itr,1])
            #line.set_3d_properties(pi[:itr,2])

            #pt.set_data([pi[itr,0]], [pi[itr,1]])
            #pt.set_3d_properties([pi[itr,2]])

        return self.lines + self.pts
    
    def Simulate(self, title = 'N-body Simulation'):
        """Animates using Matplotlib"""

        if self.trajectory is None:
            #may just run one of them automatically later, instead of raising an error
            raise ValueError("No trajectory found. Run rk4 or leapfrog to generate trajectory before simulating.")

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.set_title(f"{title}")
        ax.set_xlabel('X(AU)')
        ax.set_ylabel('Y(AU)')
        ax.set_zlabel('Z(AU)')
        ax.set_xlim(np.min(self.trajectory[:,:,0]), np.max(self.trajectory[:,:,0]))
        ax.set_ylim(np.min(self.trajectory[:,:,1]), np.max(self.trajectory[:,:,1]))

        #This doesn't like when the system is stuck in a plane
        ax.set_zlim(np.min(self.trajectory[:,:,2]), np.max(self.trajectory[:,:,2]))


        colors = plt.cm.jet(np.linspace(0, 1, self.N))

        self.lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
        
        self.pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

        anim = animation.FuncAnimation(fig, self._Animate, init_func = self._init_animate, frames=len(self.trajectory), interval=50)
        plt.show()

    def energy_cons(self):
        """Compares initial to final energy to check conservation of energy."""
        if self.trajectory is None:
            raise ValueError("No trajectory found. Run rk4 or leapfrog to generate trajectory before checking energy conservation.")
        
        initial_positions = self.trajectory[0,:,:3]
        initial_velocities = self.trajectory[0,:,3:]

        final_positions = self.trajectory[-1,:,:3]
        final_velocities = self.trajectory[-1,:,3:]

        #Kinetic energy: K = 1/2 m v^2
        K_initial = 0.5 * np.sum(self.masses * np.sum(initial_velocities**2, axis=1))
        K_final = 0.5 * np.sum(self.masses * np.sum(final_velocities**2, axis=1))

        #Potential energy: U = -G * sum(m_i * m_j / r_ij)
        U_initial = 0
        U_final = 0

        for i in range(self.N):
            for j in range(i + 1, self.N):
                r_ij = np.linalg.norm(initial_positions[i] - initial_positions[j])
                U_initial -= self.G * self.masses[i] * self.masses[j] / r_ij

                r_ij = np.linalg.norm(final_positions[i] - final_positions[j])
                U_final -= self.G * self.masses[i] * self.masses[j] / r_ij

        # Total energy
        E_initial = K_initial + U_initial
        E_final = K_final + U_final

        print(f"Initial Energy: {E_initial}")
        print(f"Final Energy: {E_final}")
        print(f"Energy Difference: {E_final - E_initial}")
        print(f"Ei / Ef: {E_initial / E_final}")



sun = Mass(1, np.zeros(3), np.zeros(3))
earth = Mass(3.00274e-6, np.array([1,0,0]), np.array([0,2*np.pi,0]))

es = System([sun,earth])

LFtraj = es.leapfrog(np.linspace(0, 1, 100))

es.energy_cons()

RK4traj = es.rk4(np.linspace(0, 1, 100))

es.energy_cons()

es.Simulate('Earth-Sun System')
#es.plot()
