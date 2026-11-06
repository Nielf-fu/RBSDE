import numpy as np
import scipy as sp
import csv
import sys
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# import os



# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TwoDim_periodic:
    """
    Reflected FBSDEs (Example 5.3) decoupled case
    
    dimensions:
    - X_t: 2-dim (X1, X2)
    - Y_t: 1-dim
    - Z_t: 2-dim(Z1, Z2)
    - W_t: 2-dim (W1, W2)
    """
    
    def __init__(self, lb=-np.pi, rb=np.pi, N=10, M=100, T=1.0):
        """
        lb: lower bound of spatial domain for X
        rb: upper bound of spatial domain for X
        N: number of time steps
        M: space partition points in [lb, rb] for each dimension of X
        T: Terminal time
        """
        self.lb = lb
        self.rb = rb
        self.N = N
        self.M = M
        self.T = T
        self.xdim = 2
        self.ydim = 1
        self.zdim = 2
        self.tgrid = np.linspace(0, T, N+1) # [0:N]
        self.dt = T / N
        self.dx = (rb - lb) / M
        self.xgrid = self.generate_grid()  # shape (M^2, 2)
        self.kstep = None  # to be set when using generator approximation method
        self.alpha = None  # to be set when using generator approximation method
        self.beta = None   # to be set when using generator approximation method

        self.runningtime = None

        self.Ysolu = None
        self.Zsolu1 = None
        self.Zsolu2 = None
        self.Lsolu = None



    def _sum_x(self, t, X):
        """compute t + X1 + X2"""
        x1, x2 = X[..., 0], X[..., 1]
        return t + x1 + x2

    def obstacle(self, t, X):

        sum_val = self._sum_x(t, X)
        obs = np.sin(sum_val) - np.sqrt(2)*t/self.T

        return obs

    def drift(self, t, X):
        """
        Drift coefficient of the forward SDE)
        t: time
        X: 2-dim (..., 2)
        return: 2-dim (..., 2)
        """
        sum_val = self._sum_x(t, X)
        b1 = np.sin(sum_val)
        b2 = np.cos(sum_val)
        return np.stack([b1, b2], axis=-1)
    
    def drift1(self, t, X):
        sum_val = self._sum_x(t, X)
        return np.sin(sum_val)
    def drift2(self, t, X):
        sum_val = self._sum_x(t, X)
        return np.cos(sum_val)

    def sigma(self, t, X):
        """
        Diffusion coefficient of the forward SDE)
        t: time
        X: 2-dim (..., 2)
        return: 2x2 matrix (..., 2, 2)
        """
        sum_val = self._sum_x(t, X)
        sig11 = 0.5*np.cos(sum_val) + 1.0
        sig22 = 0.5*np.sin(sum_val) + 1.0
        
        sigma_matrix = np.zeros(X.shape + (2,))
        sigma_matrix[..., 0, 0] = sig11
        sigma_matrix[..., 1, 1] = sig22
        return sigma_matrix
    
    def sigma1(self, t, X):
        sum_val = self._sum_x(t, X)
        return 0.5*np.cos(sum_val) + 1.0
    
    def sigma2(self, t, X):
        sum_val = self._sum_x(t, X)
        return 0.5*np.sin(sum_val) + 1.0

    def generator(self, t, X, Y, Z1, Z2):
        """
        Generator function of the BSDE
        :param t: time
        :param X: 2-dim (..., 2)
        :param Y: 1-dim Y_t
        :param Z: 2-dim (..., 2)
        :return: 1-dim
        """
        sum_val = self._sum_x(t, X)
        z1, z2 = Z1, Z2
        
        # return -3.0*z1 + 2.0*z2 - 3.0/2.0*Y**2 + 9.0/8.0*np.cos(sum_val) + 2.0
        return 2*z1**2 - 3*z1 + 2*z2 + 0.5*Y**2*np.cos(sum_val)**2 - 7.0/8.0*Y + 2.0*np.cos(sum_val)**3

    def terminal(self, t, X):
        """
        param t: time
        param X: 2-dim (..., 2)
        return: 1-dim
        """
        sum_val = self._sum_x(t, X)
        return np.cos(sum_val)

    def y_exact(self, t, X):
        """
        exact solution of Y_t
        t: time
        X: 2-dim (..., 2)
        return: 1-dim
        """
        sum_val = self._sum_x(t, X)
        return np.cos(sum_val)

    def z_exact(self, t, X):
        """
        exact solution of Z_t
        t: time
        X: 2-dim (..., 2)
        return: 2-dim (..., 2)
        """
        sum_val = self._sum_x(t, X)
        z1 = -np.sin(sum_val) * (0.5*np.cos(sum_val) + 1.0)
        z2 = -np.sin(sum_val) * (0.5*np.sin(sum_val) + 1.0)
        return np.stack([z1, z2], axis=-1)
    
    def generate_grid(self):
        """
        Generate spatial grid for X in [-pi, pi]^2
        return: grid of shape (M^2, 2)
        """
        x1 = np.linspace(self.lb, self.rb-self.dx, self.M)
        x2 = np.linspace(self.lb, self.rb-self.dx, self.M)
        X1, X2 = np.meshgrid(x1, x2)
        return np.stack([X1.flatten(), X2.flatten()], axis=-1)
    
    # compute the expectation by sinc method
    # def GaussQuad_sinc(self, n, k, ysinc):
    #     """
    #     Computes the conditional expectation E[u(x + w)], where w is a Brownian motion.
    #     x is a fixed vector of points where the expectation is computed. 
    #     The function uses sinc interpolation to compute the expectation at the points xsinc, which are uniformly spaced.
    #     ysinc is the corresponding values of u(x + kh) at the points xsinc.
    #     k: number of steps for generator approximation method (1, 2, 3, 4, 5, or 6)
    #     ysinc: values of u(t_{n+k}, x + kh) at the points xsinc, shape (m,)
    #     """

    #     x = self.xgrid
    #     x_center = x + self.drift(self.tgrid[n], x) * k * self.dt
    #     x1_center = x_center[:, 0]
    #     x2_center = x_center[:, 1]
    #     x1 = self.xgrid[:, 0]
    #     x2 = self.xgrid[:, 1]
    #     x1_matrix = x1.reshape(-1, 1) * np.ones_like(x_center[:, 0]).reshape((1, -1))
    #     x2_matrix = x2.reshape(-1, 1) * np.ones_like(x_center[:, 1]).reshape((1, -1))
    #     x1_cmatrix = np.ones_like(x1).reshape(-1, 1) * x1_center.reshape(1, -1)
    #     x2_cmatrix = np.ones_like(x2).reshape(-1, 1) * x2_center.reshape(1, -1)

    #     dist1 = self.shift(x1_matrix - x1_cmatrix)
    #     dist2 = self.shift(x2_matrix - x2_cmatrix)

    #     ysinc_matrix = ysinc.reshape(-1, 1) * np.ones_like(x_center[:, 0]).reshape((1, -1))
    #     sigma1 = np.ones_like(x1).reshape(-1, 1) * self.sigma1(self.tgrid[n], x).reshape(1, -1)
    #     sigma2 = np.ones_like(x2).reshape(-1, 1) * self.sigma2(self.tgrid[n], x).reshape(1, -1)

    #     temp = ysinc_matrix * 1.0/(2*np.pi*k*self.dt)/np.abs(sigma1*sigma2)*np.exp(-0.5/(k*self.dt) * (dist1**2/sigma1**2 + dist2**2/sigma2**2))

    #     EY = np.sum(temp, axis=0) * self.dx**2
    #     EYW1 = np.sum(temp * dist1/sigma1, axis=0) * self.dx**2
    #     EYW2 = np.sum(temp * dist2/sigma2, axis=0) * self.dx**2


    #     return EY, EYW1, EYW2

    def GaussQuad_sinc(self, n, k, ysinc):
        """
        E[Y(X)], E[Y(X) dW1], E[Y(X) dW2] with X = x + b(t_n,x) k dt + sigma(t_n,x) dW, dW ~ N(0, k dt I),
        by the trapezoidal (sinc) rule on the periodic grid.
        sigma is diagonal, so the Gaussian kernel is a product of two 1-D kernels on the tensor grid.
        """
        t = self.tgrid[n]
        x = self.xgrid
        v = k * self.dt
        L = self.rb - self.lb
        x_center = x + self.drift(t, x) * v
        nodes = np.linspace(self.lb, self.rb - self.dx, self.M).reshape(-1, 1)   # 1-D grid of generate_grid

        def kernel_1d(center, sig):
            # periodized Gaussian sum_m g(d + m*L) and its dW-weighted version; the images left out are
            # below 1e-17 (needed once sqrt(k*dt)*sigma is not small compared with L, e.g. T = 5)
            d = self.shift(nodes - center.reshape(1, -1))
            sig = sig.reshape(1, -1)
            nimg = max(0, int(np.ceil(8.9*np.sqrt(v)*sig.max()/L - 0.5)))
            g = np.zeros_like(d)
            gw = np.zeros_like(d)
            for m in range(-nimg, nimg + 1):
                dm = d + m*L
                e = np.exp(-0.5*dm**2/(v*sig**2))
                g += e
                gw += e*dm/sig
            # normalize so the discrete weights integrate 1 exactly; otherwise the mass error of the
            # under-resolved Gaussian (sqrt(k*dt)*sigma ~ dx) compounds over the backward steps and blows up
            mass = np.sum(g, axis=0) * self.dx
            return g/mass, gw/mass

        G1, G1w = kernel_1d(x_center[:, 0], self.sigma1(t, x))
        G2, G2w = kernel_1d(x_center[:, 1], self.sigma2(t, x))

        # generate_grid flattens meshgrid(x1, x2): point i2*M + i1 is (x1[i1], x2[i2])
        Ymat = ysinc.reshape(self.M, self.M)
        A = Ymat @ G1
        Aw = Ymat @ G1w

        EY = np.sum(G2 * A, axis=0) * self.dx**2
        EYW1 = np.sum(G2 * Aw, axis=0) * self.dx**2
        EYW2 = np.sum(G2w * A, axis=0) * self.dx**2

        return EY, EYW1, EYW2






# alpha of generator approximation method
    def get_alpha(self, k):
        if k==1:
            alpha = np.array([-1.0, 1.0])
        elif k==2:
            alpha = np.array([-3.0/2.0, 2.0, -1.0/2.0])
        elif k==3:
            alpha = np.array([-11.0/6.0, 3.0, -3.0/2.0, 1.0/3.0])
        elif k==4:
            alpha = np.array([-25.0/12.0, 4.0, -3.0, 4.0/3.0, -1.0/4.0])
        elif k==5:
            alpha = np.array([-137.0/60.0, 5.0, -5.0, 10.0/3.0, -5.0/4.0, 1.0/5.0])
        elif k==6:
            alpha = np.array([-49.0/20.0, 6.0, -15.0/2.0, 20.0/3.0, -15.0/4.0, 6.0/5.0, -1.0/6.0])
        else:
            raise ValueError("k must be in {1, 2, 3, 4, 5, 6} in get_alpha(k).")
        return alpha
    

    # beta for the explicit generator approximation method
    def get_beta(self, k):

        if k==1:
            beta = np.array([1.0])
        elif k==2:
            beta = np.array([2.0, -1.0])
        elif k==3:
            beta = np.array([3.0, -3.0, 1.0])
        elif k==4:
            beta = np.array([4.0, -6.0, 4.0, -1.0])
        elif k==5:
            beta = np.array([5.0, -10.0, 10.0, -5.0, 1.0])
        elif k==6:
            beta = np.array([6.0, -15.0, 20.0, -15.0, 6.0, -1.0])
        else:
            raise ValueError("k must be in {1, 2, 3, 4, 5, 6} in get_beta(k).")
        
        return beta
    
    def shift(self, x):

        x_shifted = (x - self.lb) % (self.rb - self.lb) + self.lb

        return x_shifted


    def Euler_step(self, n, Yknown):
        """
        Perform one step of the Euler method for the backward SDE
        n: current time step
        Yknown: known values of Y at time step n+1, shape (M^2,)
        return: Ysolu, Zsolu1, Zsolu2 at time step n, each of shape (M^2,)
        """

        Ysolu = np.zeros_like(Yknown)
        Zsolu1 = np.zeros_like(Yknown)
        Zsolu2 = np.zeros_like(Yknown)

        EY, EYW1, EYW2 = self.GaussQuad_sinc(n, 1, Yknown)

        Zsolu1 = EYW1 / self.dt
        Zsolu2 = EYW2 / self.dt
        Yex = EY
        Ysolu = EY + self.generator(self.tgrid[n], self.xgrid, Yex, Zsolu1, Zsolu2) * self.dt

        obstacle = self.obstacle(self.tgrid[n], self.xgrid)
        Ysolu = np.maximum(Ysolu, obstacle)
        Lsolu = np.maximum(0, obstacle - Ysolu)/self.dt


        return Ysolu, Zsolu1, Zsolu2, Lsolu


    def exGA_step(self, n, kstep, Yknown):
        
        # """One step of k-step explicit Genorator Approximation Solver for FBSDEs"""
        
        """
        Perform one step of the Euler method for the backward SDE
        n: current time step
        Yknown: known values of Y at time step n+1, shape (M^2,)
        return: Ysolu, Zsolu1, Zsolu2 at time step n, each of shape (M^2,)
        """

        Yknown = Yknown.reshape((Yknown.shape[0], kstep))


        Ysolu = np.zeros_like(Yknown[:,0])
        Zsolu1 = np.zeros_like(Yknown[:,0])
        Zsolu2 = np.zeros_like(Yknown[:,0])
        EY = np.zeros_like(Yknown)
        EYW1 = np.zeros_like(Yknown)
        EYW2 = np.zeros_like(Yknown)

        for i in range(kstep):
            EY[:, i], EYW1[:, i], EYW2[:, i] = self.GaussQuad_sinc(n, i+1, Yknown[:, i])
        
        
        Zsolu1 = np.matmul(EYW1, self.alpha[1:]).reshape(len(Ysolu)) / self.dt
        Zsolu2 = np.matmul(EYW2, self.alpha[1:]).reshape(len(Ysolu)) / self.dt
        Yex = np.matmul(EY, self.beta).reshape(len(Ysolu))
        Ysolu = -1.0/self.alpha[0]*(np.matmul(EY, self.alpha[1:]).reshape(len(Ysolu)) + self.generator(self.tgrid[n], self.xgrid, Yex, Zsolu1, Zsolu2) * self.dt)

        obstacle = self.obstacle(self.tgrid[n], self.xgrid)
        Lsolu = -self.alpha[0]/self.dt * np.maximum(0, obstacle - Ysolu)
        Ysolu = np.maximum(Ysolu, obstacle)

        return Ysolu, Zsolu1, Zsolu2, Lsolu


    def exGA_solver(self):

        self.Ysolu = np.zeros((self.M**2, self.N+1))
        self.Zsolu1 = np.zeros((self.M**2, self.N+1))
        self.Zsolu2 = np.zeros((self.M**2, self.N+1))
        self.Lsolu = np.zeros((self.M**2, self.N+1))

        start = datetime.now()


        # terminal condition
        if self.kstep == 1:
            self.Ysolu[:, self.N] = self.terminal(self.tgrid[self.N], self.xgrid)
        elif self.kstep == 2:
            self.Ysolu[:, self.N] = self.terminal(self.tgrid[self.N], self.xgrid)
            self.Ysolu[:, self.N-1], self.Zsolu1[:, self.N-1], self.Zsolu2[:, self.N-1], self.Lsolu[:, self.N-1] = self.Euler_step(self.N-1, self.Ysolu[:, self.N])
        else:
            print("Only for kstep = 1 or 2, because this is an obstacle problem.")
            sys.exit(0)


        # backward iteration
        for n in range(self.N-self.kstep, -1, -1):
            print("current step:", n)
            self.Ysolu[:, n], self.Zsolu1[:, n], self.Zsolu2[:, n], self.Lsolu[:, n] = self.exGA_step(n, self.kstep, self.Ysolu[:, n+1:n+1+self.kstep])
        
        end = datetime.now()

        self.runningtime = (end - start).total_seconds()


        return self.Ysolu, self.Zsolu1, self.Zsolu2, self.Lsolu


if __name__ == "__main__":
    
    fbsde = TwoDim_periodic(lb=-np.pi, rb=np.pi, N=360, M=128, T=5.0)
    fbsde.kstep = 1
    fbsde.alpha = fbsde.get_alpha(fbsde.kstep)
    fbsde.beta = fbsde.get_beta(fbsde.kstep)

    Ynum, Znum1, Znum2, Lnum = fbsde.exGA_solver()

    print("Running time (seconds):", fbsde.runningtime)
    
    with open('Yresult.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for i in range(fbsde.N+1):
            writer.writerow(fbsde.Ysolu[:, i])

    with open('Xresult.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fbsde.xgrid[:, 0])
        writer.writerow(fbsde.xgrid[:, 1])
    
    with open('tresult.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fbsde.tgrid)

    with open('Lresult.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for i in range(fbsde.N+1):
            writer.writerow(fbsde.Lsolu[:, i])
    
    
    figure = plt.figure(figsize=(20, 10))
    ax1 = figure.add_subplot(121, projection='3d')
    ax1.scatter(fbsde.xgrid[:, 0], fbsde.xgrid[:, 1], Ynum[:, 0], s=1)
    ax1.set_title("Numerical Solution Y at t=0")
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    ax1.set_zlabel("Y")

    ax1.plot_surface(fbsde.xgrid[:, 0].reshape(fbsde.M, fbsde.M), fbsde.xgrid[:, 1].reshape(fbsde.M, fbsde.M), fbsde.obstacle(fbsde.tgrid[0], fbsde.xgrid).reshape(fbsde.M, fbsde.M), color='r', alpha=0.5)
    plt.show()
    
