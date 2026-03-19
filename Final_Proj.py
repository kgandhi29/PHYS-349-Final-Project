import numpy as np
import copy
import matplotlib.pyplot as plt



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

    def dpdt(self, p, t):
        positions = p[:,:3]
        velocities = p[:,3:]

        # Vectorized force calculation
        pos_i = positions[:, np.newaxis, :]  # (N, 1, 3)
        pos_j = positions[np.newaxis, :, :]  # (1, N, 3)
        r_vec = pos_j - pos_i  # (N, N, 3)
        r = np.linalg.norm(r_vec, axis=2)  # (N, N)
        
        # Sets r=0 to inf to avoid division by zero in acceleration calculation
        r = np.where(r == 0, np.inf, r)
        
        # Shape (N, N, 3)
        acc_matrix = self.G * self.masses[np.newaxis, :, np.newaxis] * r_vec / (r[:, :, np.newaxis]**2 + self.epsilon**2)**(3/2)
        
        # Sum over j (axis=1) to get total acceleration for each i
        atot = np.sum(acc_matrix, axis=1)  #shape (N, 3)
        
        dpdt = np.concatenate((velocities, atot), axis = 1)

        return dpdt  

    def dpdt2(self, pi, t):

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
    
    def rk4(self, t):
        dt = t[1] - t[0]   
        dpdt = self.dpdt 
        
        p = np.concatenate((self.positions, self.velocities), axis =1)  #should be shape (N,6)
        
        k1 = dt * dpdt(p, t[0])
        k2 = dt * dpdt(p + k1/2, t[0] + dt/2)
        k3 = dt * dpdt(p + k2/2, t[0] + dt/2)
        k4 = dt * dpdt(p + k3, t[0] + dt)

        pr = np.reshape(p,(-1,6))
        traj = []
        traj.append(pr)
        
        for i in range(1, len(t)):
            term2 = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
            p = p + term2
            
            k1 = dt * dpdt(p, t[i])
            k2 = dt * dpdt(p + k1/2, t[i] + dt/2)
            k3 = dt * dpdt(p + k2/2, t[i] + dt/2)
            k4 = dt * dpdt(p + k3, t[i] + dt)
            
            pr = np.reshape(p,(-1,6))
            traj.append(copy.copy(pr))

        traj = np.array(traj)  #shape (len(t), N, 6)
            
        return np.array(traj)
    
    def leapfrog(self, t):
        """This function"""
        ptraj = []
        vtraj = []
        dt = t[1] - t[0]
        p = self.positions
        v = self.velocities     
        a = self.dpdt2(p, t)

        v_half = v + a * dt / 2

        ptraj.append(copy.copy(p))
        vtraj.append(copy.copy(v_half))

        for i in range(1, len(t)):

            p = p + v_half * dt

            ptraj.append(copy.copy(p))

            a = self.dpdt2(p, t)
            v_half = v_half + a * dt

            vtraj.append(copy.copy(v_half))

        ptraj = np.array(ptraj)
        vtraj = np.array(vtraj)

        return ptraj, vtraj
    
    def plot(self, traj, elev = 90, azim = -90):

        #setup 3d axis
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

ptraj = es.rk4(np.linspace(0,1,1000))

xtraj, vtraj = es.leapfrog(np.linspace(0,10,1000))

ptraj2 = np.concatenate((xtraj, vtraj), axis = 2)
es.plot(ptraj2)


#1. Seperate x and v in rk4, adjust, then combine dpdt functions into one (so it is vectorized and returns the same shape (N,3))

#2. Combine xtraj and vtraj at the end of leapfrog, even though they don't agree on time

#3. Average out half step velocites to get full step and then concatenate with positions to get same shape as rk4 traj.


# - Animation with matplotlib FuncAnimation.
#   - Dots at end of line

# - Make more dynamic with units
#   - Adjust labels on plot


