import numpy as np
import scipy as sp
import csv
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
# import os



# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class AmericanCallOption:
    def __init__(self, S0, K, T, mu, sigma, r, q):
        self.S0 = S0      # Current stock price
        self.K = K        # Strike price
        self.T = T        # Time to expiration (years)
        self.mu = mu      # Expected return 
        self.r = r        # Risk-free interest rate
        self.q = q        # Continuous dividend yield
        self.sigma = sigma  # Volatility
        self.N = None    # Number of time steps
        self.dt = None   # Time step size
        self.dh = None   # Space step size = 0.5*sqrt(dt)
        self.tgrid = None # Time grid
        self.lb = None  # Lower bound of space grid
        self.rb = None  # Upper bound of space grid
        self.xnum = None # Number of space steps
        
        self.yknown = None  # Placeholder for known y values
        self.zknown = None  # Placeholder for known z values
        self.lknown = None  # Placeholder for known lambda values
        self.y = None      # Placeholder for y values
        self.z = None      # Placeholder for z values
        self.l = None      # Placeholder for lambda values
        self.s = None      # Placeholder for stock prices
        self.call_true = 7.88201865e+00 #7.881905800537484  # Placeholder for true call option price
        # self.put_true = 7.4809789340178146   # Placeholder for true put option price
        self.running_time = None  # Placeholder for running time
        self.kstep = None  # Number of steps in the explicit GA method
        self.early_price = []  # Early exercise stock price of the boundary
        self.early_t = []   # Early exercise time of the boundary

        

    
    def Call_payoff(self, S):
        """Payoff function for American call option"""
        return np.maximum(S - self.K, 0)

    def generator(self, t, x, y, z):
        """generator function of the BSDE"""
        return -self.r * y - (self.mu + self.q -self.r) * z /self.sigma
    
    def drift(self, x):
        """Drift function of the stock price"""
        return self.mu * x
    
    def diffusion(self, x):
        """Diffusion function of the stock price"""
        return self.sigma * x
        
    def GBM(self, t, Wt):
        """Geometric Brownian Motion for stock price"""
        return self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * Wt)
    
    def terminal(self, Wt):
        """Terminal condition for the BSDE"""
        S_T = self.GBM(self.T, Wt)
        return self.Call_payoff(S_T)
    
    def zterminal(self, Wt):
        """Terminal condition for z in the BSDE"""
        S_T = self.GBM(self.T, Wt)
        payoff_derivative = np.where(S_T > self.K, 1.0, 0.0)  # Derivative of call payoff
        return payoff_derivative * S_T * self.sigma
    
    def hfun(self, t, Wt):
        """Obstacle function for American option"""
        S_t = self.GBM(t, Wt)
        return self.Call_payoff(S_t)
    
    def generate_grid(self, m):
        """Generate time and space grid"""
        self.dt = self.T / self.N
        self.dh = 0.5 * np.sqrt(self.dt)
        self.tgrid = np.linspace(0, self.T, self.N+1)
        self.lb = np.zeros(self.N+1)  # Placeholder for lower bound
        self.rb = np.zeros(self.N+1)  # Placeholder for upper bound
        self.xnum = np.zeros(self.N+1, dtype=int)  # Placeholder for number of space steps
        self.lb[0] = 0.0
        self.rb[0] = 0.0
        self.xnum[0] = 1
        for i in range(1, self.N+1):
            self.lb[i] = self.lb[i-1] - m * self.dh
            self.rb[i] = self.rb[i-1] + m * self.dh
            self.xnum[i] = self.xnum[i-1] + 2 * m
        
        return



    def pre_compute(self, kstep):
        """Pre-compute values needed for the exGA solver"""
        
        yknown = np.zeros((self.xnum[-2], kstep))
        zknown = np.zeros((self.xnum[-2], kstep))
        lknown = np.zeros((self.xnum[-2], kstep))
        
        
        self.lb[-1] = self.lb[-1] - 2.0
        self.rb[-1] = self.rb[-1] + 2.0
        self.xnum[-1] = self.xnum[-1]*5.0
        
        terminal_xgrid = np.linspace(self.lb[-1], self.rb[-1], self.xnum[-1])
        terminal_ygrid = self.terminal(terminal_xgrid)
        terminal_zgrid = self.zterminal(terminal_xgrid)
        
        for i in range(kstep):
            n = self.N - 1 - i
            
            xsolve = np.linspace(self.lb[n], self.rb[n], self.xnum[n])
            ysolve = np.zeros_like(xsolve)
            zsolve = np.zeros_like(xsolve)
            lsolve = np.zeros_like(xsolve)
            
            ygrid = np.zeros((self.xnum[-1], i+1))
            ygrid[:,i] = terminal_ygrid

            if i == 0:
                
                # ysolve, zsolve, lsolve = self.exGA_step(n, 1, terminal_ygrid)
                ysolve, zsolve, lsolve = self.CN_step(n, terminal_ygrid, terminal_zgrid)
                # St_solve = self.GBM(self.tgrid[n], xsolve)
                # ysolve = BlackScholesCall(St_solve, self.K, self.T - self.tgrid[n], self.r, self.sigma, self.q)
                # zsolve = DeltaBlackScholesCall(St_solve, self.K, self.T - self.tgrid[n], self.r, self.sigma, self.q) \
                #                                 *self.sigma*St_solve
                
                yknown[:self.xnum[n], kstep-1] = ysolve
                zknown[:self.xnum[n], kstep-1] = zsolve
                lknown[:self.xnum[n], kstep-1] = lsolve
            else:
                for j in range(i):
                    ygrid[:self.xnum[n+j+1], i-1-j] = yknown[:self.xnum[n+j+1], kstep-1-j]
                
                ysolve, zsolve, lsolve = self.exGA_step(n, i+1, ygrid)
                # St_solve = self.GBM(self.tgrid[n], xsolve)
                # ysolve = BlackScholesCall(St_solve, self.K, self.T - self.tgrid[n], self.r, self.sigma, self.q)
                # zsolve = DeltaBlackScholesCall(St_solve, self.K, self.T - self.tgrid[n], self.r, self.sigma, self.q) \
                #                                 *self.sigma*St_solve

                yknown[:self.xnum[n], kstep-1-i] = ysolve
                zknown[:self.xnum[n], kstep-1-i] = zsolve
                lknown[:self.xnum[n], kstep-1-i] = lsolve
        
        return yknown, zknown, lknown


    def exGA_step(self, n, kstep, yknown):
        
        """One step of k-step explicit Genorator Approximation Solver for FBSDEs"""
        
        yknown = yknown.reshape((yknown.shape[0], kstep))
        alpha = self.get_alpha(kstep)
        beta = self.get_beta(kstep)
        
        X = np.linspace(self.lb[n], self.rb[n], self.xnum[n])
        Z = np.zeros_like(X)
        Y = np.zeros_like(X)
        L = np.zeros_like(X)
        EY = np.zeros((len(X), kstep))
        EYW = np.zeros((len(X), kstep))
        
        for i in range(1, kstep+1):
            xgrid = np.linspace(self.lb[n+i], self.rb[n+i], self.xnum[n+i])
            ygrid = yknown[:self.xnum[n+i], i-1]

            EY[:,i-1], EYW[:,i-1] = self.GaussQuad_sinc(X, i*self.dt, xgrid, ygrid)   
        
        Z = np.matmul(EYW, alpha[1:]).reshape(len(X)) / self.dt
        Yex = np.matmul(EY, beta).reshape(len(X))
        Yeu = -1.0/alpha[0]*(np.matmul(EY, alpha[1:]).reshape(len(X)) + self.generator(self.tgrid[n], X, Yex, Z) * self.dt)
        
        Y = np.maximum(Yeu, self.hfun(self.tgrid[n], X))
        L = -alpha[0]/self.dt * np.maximum(self.hfun(self.tgrid[n], X) - Yeu, 0)
        
        # early_index = np.argmax(L > 0.0) if np.any(L > 0.0) else None
        # early_index = np.where(L > 0 & self.GBM(self.tgrid[n], X)>self.K)[0][0] if np.any(L > 0) else None
        # early_index = np.where(Y-Yeu>1.0e-10)[0][0] if np.any(Y > Yeu) else None
        early_index = np.argmax((L > 0) & (self.GBM(self.tgrid[n], X)>self.K)) if np.any((L > 0) & (self.GBM(self.tgrid[n], X)>self.K)) else None
        self.early_price.append(self.GBM(self.tgrid[n], X[early_index]) if early_index is not None else None)
        self.early_t.append(self.tgrid[n])

        # print('early_index: ', early_index, self.GBM(self.tgrid[n], X[early_index]))
        if early_index is not None: 
            if self.GBM(self.tgrid[n], X[early_index]) < self.K:
                print('Early exercise at time t={:.4f}, stock price S={:.4f}'.format(self.tgrid[n], self.GBM(self.tgrid[n], X[early_index])), 
                    Y[early_index], Yeu[early_index], L[early_index])

        return Y, Z, L



    def exGA_solver(self, kstep):
        
        """explicit Genorator Approximation Solver for FBSDEs"""
        
        start = datetime.now()
        self.kstep = kstep
        self.early_price = []
        self.early_t = []

        self.yknown = np.zeros((self.xnum[-2], kstep))
        self.zknown = np.zeros((self.xnum[-2], kstep))
        self.lknown = np.zeros((self.xnum[-2], kstep))
        self.y = np.zeros((self.xnum[-2], self.N))
        self.z = np.zeros((self.xnum[-2], self.N))
        self.l = np.zeros((self.xnum[-2], self.N))
        self.s = np.zeros((self.xnum[-2], self.N))
        
        self.yknown, self.zknown, self.lknown = self.pre_compute(kstep)
        self.y[:,self.N-kstep:] = self.yknown
        self.z[:,self.N-kstep:] = self.zknown
        self.l[:,self.N-kstep:] = self.lknown
        
        
        for i in range(self.N-kstep-1, -1, -1):
            Y, Z, L = self.exGA_step(i, kstep, self.yknown)
            self.y[:self.xnum[i], i] = Y
            self.z[:self.xnum[i], i] = Z
            self.l[:self.xnum[i], i] = L
            
            for j in range(kstep-1, -1, -1):
                self.yknown[:,j] = self.yknown[:,j-1]
            if i>0:
                self.yknown[:self.xnum[i], 0] = Y
            
        end = datetime.now()
        self.running_time = end - start
        
        return
    

    # Crank-Nicolson method

    def CN_step(self, n, yknown, zknown):
        """
        0.5*dt*Z = -0.5 * dt * EZ + EYW + 0.5*dt*EFW
        Y = EY + 0.5*dt*f + 0.5*dt*EF
        """
        dt = self.dt
        t1 = self.tgrid[n]
        t2 = self.tgrid[n+1]
        
        X = np.linspace(self.lb[n], self.rb[n], self.xnum[n])
        Z = np.zeros_like(X)
        Y = np.zeros_like(X)
        Yeu = np.zeros_like(X)
        L = np.zeros_like(X)

        EY = np.zeros_like(X)
        EYW = np.zeros_like(X)
        EZ = np.zeros_like(X)
        EZW = np.zeros_like(X)
        EF = np.zeros_like(X)
        EFW = np.zeros_like(X)

        x2grid = np.linspace(self.lb[n+1], self.rb[n+1], self.xnum[n+1])
        f2grid = self.generator(t2, x2grid, yknown[:self.xnum[n+1]], zknown[:self.xnum[n+1]])

        EY, EYW = self.GaussQuad_sinc(X, dt, x2grid, yknown[:self.xnum[n+1]])
        EZ, EZW = self.GaussQuad_sinc(X, dt, x2grid, zknown[:self.xnum[n+1]])
        EF, EFW = self.GaussQuad_sinc(X, dt, x2grid, f2grid)
        
        Z = EFW-EZ + 2.0*EYW/dt

        for i in range(len(X)):
            func = lambda y: EY[i] + 0.5*self.generator(t1, X[i], y, Z[i]) * dt + 0.5*dt*EF[i] - y
            root = sp.optimize.root(func, EY[i])
            Yeu[i] = root.x[0]

        Y = np.maximum(Yeu, self.hfun(self.tgrid[n], X))
        L = 1.0/dt * np.maximum(self.hfun(self.tgrid[n], X) - Yeu, 0)

        # early_index = np.argmax(L > 0.0) if np.any(L > 0.0) else None
        # early_index = np.where(L > 0 & self.GBM(self.tgrid[n], X)>self.K)[0][0] if np.any(L > 0) else None
        # early_index = np.where(Y-Yeu>1.0e-10)[0][0] if np.any(Y > Yeu) else None
        early_index = np.argmax((L > 0) & (self.GBM(self.tgrid[n], X)>self.K)) if np.any((L > 0) & (self.GBM(self.tgrid[n], X)>self.K)) else None
        self.early_price.append(self.GBM(self.tgrid[n], X[early_index]) if early_index is not None else None)
        self.early_t.append(self.tgrid[n])

        if self.GBM(self.tgrid[n], X[early_index]) < self.K:
            print('Early exercise at time t={:.4f}, stock price S={:.4f}'.format(self.tgrid[n], self.GBM(self.tgrid[n], X[early_index])), 
                  Y[early_index], Yeu[early_index], L[early_index])

        return Y, Z, L


    def pre_CN_step(self):

        xgrid = np.linspace(self.lb[self.N]-2, self.rb[self.N]+2, 10*self.xnum[self.N])
        ygrid = self.terminal(xgrid)
        zgrid = self.zterminal(xgrid)
        fgrid = self.generator(self.T, xgrid, ygrid, zgrid)

        # yknown, zknown, lknown = self.CN_step(self.N-1, ygrid, zgrid)
        n = self.N - 1
        X = np.linspace(self.lb[n], self.rb[n], self.xnum[n])
        Z = np.zeros_like(X)
        Y = np.zeros_like(X)
        Yeu = np.zeros_like(X)
        L = np.zeros_like(X)
        
        EY = np.zeros_like(X)
        EYW = np.zeros_like(X)
        EZ = np.zeros_like(X)
        EZW = np.zeros_like(X)
        EF = np.zeros_like(X)
        EFW = np.zeros_like(X)


        EY, EYW = self.GaussQuad_sinc(X, self.dt, xgrid, ygrid)
        EZ, EZW = self.GaussQuad_sinc(X, self.dt, xgrid, zgrid)
        EF, EFW = self.GaussQuad_sinc(X, self.dt, xgrid, fgrid)
        
        Z = EFW-EZ + 2.0*EYW/self.dt

        for i in range(len(X)):
            func = lambda y: EY[i] + 0.5*self.generator(self.tgrid[self.N-1], X[i], y, Z[i]) * self.dt + 0.5*self.dt*EF[i] - y
            root = sp.optimize.root(func, EY[i])
            Yeu[i] = root.x[0]

        Y = np.maximum(Yeu, self.hfun(self.tgrid[self.N-1], X))
        L = 1.0/self.dt * np.maximum(self.hfun(self.tgrid[self.N-1], X) - Yeu, 0)
        
        # early_index = np.argmax(L > 0.0) if np.any(L > 0.0) else None
        # early_index = np.where((L > 0) & (self.GBM(self.tgrid[n], X)>self.K))[0][0] if np.any(L > 0) else None
        # early_index = np.where(Y-Yeu>1.0e-10)[0][0] if np.any(Y > Yeu) else None
        early_index = np.argmax((L > 0) & (self.GBM(self.tgrid[n], X)>self.K)) if np.any((L > 0) & (self.GBM(self.tgrid[n], X)>self.K)) else None
        self.early_price.append(self.GBM(self.tgrid[n], X[early_index]) if early_index is not None else None)
        self.early_t.append(self.tgrid[n])

        if self.GBM(self.tgrid[n], X[early_index]) < self.K:
            print('Early exercise at time t={:.4f}, stock price S={:.4f}'.format(self.tgrid[n], self.GBM(self.tgrid[n], X[early_index])), 
                  Y[early_index], Yeu[early_index], L[early_index])


        # use true solution as terminal condition
        # xsolve = np.linspace(self.lb[self.N-1], self.rb[self.N-1], self.xnum[self.N-1])
        # St_solve = self.GBM(self.tgrid[self.N-1], xsolve)
        # yknown = BlackScholesCall(St_solve, self.K, self.T - self.tgrid[self.N-1], self.r, self.sigma, self.q)
        # zknown = DeltaBlackScholesCall(St_solve, self.K, self.T - self.tgrid[self.N-1], self.r, self.sigma, self.q) \
        #                 *self.sigma*St_solve
        # lknown = np.zeros_like(yknown)
        
        return Y, Z, L
    
    
    def CN_solver(self):

        start = datetime.now()
        self.early_price = []
        self.early_t = []
        
        self.y = np.zeros((self.xnum[-2], self.N))
        self.z = np.zeros((self.xnum[-2], self.N))
        self.l = np.zeros((self.xnum[-2], self.N))
        
        self.yknown, self.zknown, self.lknown = self.pre_CN_step()
        self.y[:,self.N-1] = self.yknown
        self.z[:,self.N-1] = self.zknown
        self.l[:,self.N-1] = self.lknown
        

        for i in range(self.N-2, -1, -1):
            Y, Z, L = self.CN_step(i, self.yknown, self.zknown)
            # print('xnum[i]: ', self.xnum[i], ' len(Y): ', len(Y))
            self.y[:self.xnum[i], i] = Y
            self.z[:self.xnum[i], i] = Z
            self.l[:self.xnum[i], i] = L
            self.yknown = Y
            self.zknown = Z
            self.lknown = L
        
        end = datetime.now()
        self.running_time = end - start

        return


    def convergence_rate(self, dt, error):
        """Compute the convergence rate given dt and error arrays"""
        n = len(dt)
        dt_col = dt.reshape(n, 1)
        err_col = error.reshape(n, 1)

        Xmatrix = np.hstack((np.ones((n,1)), np.log(dt_col)))

        beta = np.matmul(np.linalg.inv(np.matmul(Xmatrix.T, Xmatrix)), (np.matmul(Xmatrix.T, np.log(err_col))))
        
        rate = beta[1]            

        return rate


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
    
    
    # compute the expectation by sinc method
    def GaussQuad_sinc(self, x, dt, xsinc, ysinc):
        """
        Computes the conditional expectation E[u(x + w)], where w is a Brownian motion.
        x is a fixed vector of points where the expectation is computed. 
        The function uses sinc interpolation to compute the expectation at the points xsinc, which are uniformly spaced.
        ysinc is the corresponding values of u(x + kh) at the points xsinc.
        """
        n = len(x)
        m = len(xsinc)
        EY = np.zeros_like(x)
        EYW = np.zeros_like(x)
        
        x_grid = np.ones((m,1)) * x.reshape((1,n))
        xsinc_grid = xsinc.reshape((m,1)) * np.ones((1,n))
        ysinc_grid = ysinc.reshape((m,1)) * np.ones((1,n))
        
        dist = xsinc_grid - x_grid
        EY_grid = ysinc_grid * 1.0/np.sqrt(2*np.pi)/np.sqrt(dt)*np.exp(-0.5 * dist**2 / dt)
        EYW_grid = ysinc_grid * 1.0/np.sqrt(2*np.pi)/np.sqrt(dt)*np.exp(-0.5 * dist**2 / dt) * dist
        EY = np.sum(EY_grid, axis=0) * (xsinc[1] - xsinc[0])
        EYW = np.sum(EYW_grid, axis=0) * (xsinc[1] - xsinc[0])

        return EY, EYW
    
    def print_result(self):
        # Compute the error between the numerical solution and the true solution
        err = np.abs(self.y[0,0] - self.call_true)
        print(f"dt = {self.dt:10.8f}, \t dh={self.dh:10.8f}, \t Y= {self.y[0,0]:10.8e}, \t True Call= {self.call_true:10.8e}, \t Error= {err:10.8e}")
        return



class AmericanPutOption:
    def __init__(self, S0, K, T, mu, sigma, r, q):
        self.S0 = S0      # Current stock price
        self.K = K        # Strike price
        self.T = T        # Time to expiration (years)
        self.mu = mu      # Expected return 
        self.r = r        # Risk-free interest rate
        self.q = q        # Continuous dividend yield
        self.sigma = sigma  # Volatility
        self.N = None    # Number of time steps
        self.dt = None   # Time step size
        self.dh = None   # Space step size = 0.5*sqrt(dt)
        self.tgrid = None # Time grid
        self.lb = None  # Lower bound of space grid
        self.rb = None  # Upper bound of space grid
        self.xnum = None # Number of space steps
        
        self.yknown = None  # Placeholder for known y values
        self.zknown = None  # Placeholder for known z values
        self.lknown = None  # Placeholder for known lambda values
        self.y = None      # Placeholder for y values
        self.z = None      # Placeholder for z values
        self.l = None      # Placeholder for lambda values
        self.s = None      # Placeholder for stock prices
        # self.call_true = 7.88201865e+00 #7.881905800537484  # Placeholder for true call option price
        self.put_true = 7.4809789340178146   # Placeholder for true put option price
        self.running_time = None  # Placeholder for running time
        self.kstep = None  # Number of steps in the explicit GA method
        self.early_price = []  # Early exercise stock price of the boundary
        self.early_t = []   # Early exercise time of the boundary
        

    def Put_payoff(self, S):
        """Payoff function for American put option"""
        return np.maximum(self.K - S, 0)

    def generator(self, t, x, y, z):
        """generator function of the BSDE"""
        return -self.r * y - (self.mu + self.q -self.r) * z /self.sigma
    
    def drift(self, x):
        """Drift function of the stock price"""
        return self.mu * x
    
    def diffusion(self, x):
        """Diffusion function of the stock price"""
        return self.sigma * x
        
    def GBM(self, t, Wt):
        """Geometric Brownian Motion for stock price"""
        return self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * Wt)
    
    def terminal(self, Wt):
        """Terminal condition for the BSDE"""
        S_T = self.GBM(self.T, Wt)
        # return self.Put_payoff(S_T)
        return self.Put_payoff(S_T)
    
    def zterminal(self, Wt):
        """Terminal condition for z in the BSDE"""
        S_T = self.GBM(self.T, Wt)
        payoff_derivative = np.where(S_T < self.K, -1.0, 0.0)  # Derivative of put payoff
        return payoff_derivative * S_T * self.sigma
    
    def hfun(self, t, Wt):
        """Obstacle function for American option"""
        S_t = self.GBM(t, Wt)
        return self.Put_payoff(S_t)
    
    def generate_grid(self, m):
        """Generate time and space grid"""
        self.dt = self.T / self.N
        self.dh = 0.5 * np.sqrt(self.dt)
        self.tgrid = np.linspace(0, self.T, self.N+1)
        self.lb = np.zeros(self.N+1)  # Placeholder for lower bound
        self.rb = np.zeros(self.N+1)  # Placeholder for upper bound
        self.xnum = np.zeros(self.N+1, dtype=int)  # Placeholder for number of space steps
        self.lb[0] = 0.0
        self.rb[0] = 0.0
        self.xnum[0] = 1
        for i in range(1, self.N+1):
            self.lb[i] = self.lb[i-1] - m * self.dh
            self.rb[i] = self.rb[i-1] + m * self.dh
            self.xnum[i] = self.xnum[i-1] + 2 * m
        return


    def pre_compute(self, kstep):
        """Pre-compute values needed for the exGA solver"""
        
        yknown = np.zeros((self.xnum[-2], kstep))
        zknown = np.zeros((self.xnum[-2], kstep))
        lknown = np.zeros((self.xnum[-2], kstep))
        
        
        self.lb[-1] = self.lb[-1] - 2.0
        self.rb[-1] = self.rb[-1] + 2.0
        self.xnum[-1] = self.xnum[-1]*5.0
        
        terminal_xgrid = np.linspace(self.lb[-1], self.rb[-1], self.xnum[-1])
        terminal_ygrid = self.terminal(terminal_xgrid)
        terminal_zgrid = self.zterminal(terminal_xgrid)
        
        for i in range(kstep):
            n = self.N - 1 - i
            
            xsolve = np.linspace(self.lb[n], self.rb[n], self.xnum[n])
            ysolve = np.zeros_like(xsolve)
            zsolve = np.zeros_like(xsolve)
            lsolve = np.zeros_like(xsolve)
            
            ygrid = np.zeros((self.xnum[-1], i+1))
            ygrid[:,i] = terminal_ygrid

            if i == 0:
                
                # ysolve, zsolve, lsolve = self.exGA_step(n, 1, terminal_ygrid)
                ysolve, zsolve, lsolve = self.CN_step(n, terminal_ygrid, terminal_zgrid)
                # St_solve = self.GBM(self.tgrid[n], xsolve)
                # ysolve = BlackScholesCall(St_solve, self.K, self.T - self.tgrid[n], self.r, self.sigma, self.q)
                # zsolve = DeltaBlackScholesCall(St_solve, self.K, self.T - self.tgrid[n], self.r, self.sigma, self.q) \
                #                                 *self.sigma*St_solve
                
                yknown[:self.xnum[n], kstep-1] = ysolve
                zknown[:self.xnum[n], kstep-1] = zsolve
                lknown[:self.xnum[n], kstep-1] = lsolve
            else:
                for j in range(i):
                    ygrid[:self.xnum[n+j+1], i-1-j] = yknown[:self.xnum[n+j+1], kstep-1-j]
                
                ysolve, zsolve, lsolve = self.exGA_step(n, i+1, ygrid)
                # St_solve = self.GBM(self.tgrid[n], xsolve)
                # ysolve = BlackScholesCall(St_solve, self.K, self.T - self.tgrid[n], self.r, self.sigma, self.q)
                # zsolve = DeltaBlackScholesCall(St_solve, self.K, self.T - self.tgrid[n], self.r, self.sigma, self.q) \
                #                                 *self.sigma*St_solve

                yknown[:self.xnum[n], kstep-1-i] = ysolve
                zknown[:self.xnum[n], kstep-1-i] = zsolve
                lknown[:self.xnum[n], kstep-1-i] = lsolve
        
        return yknown, zknown, lknown


    def exGA_step(self, n, kstep, yknown):
        
        """One step of k-step explicit Genorator Approximation Solver for FBSDEs"""
        
        yknown = yknown.reshape((yknown.shape[0], kstep))
        alpha = self.get_alpha(kstep)
        beta = self.get_beta(kstep)
        
        X = np.linspace(self.lb[n], self.rb[n], self.xnum[n])
        Z = np.zeros_like(X)
        Y = np.zeros_like(X)
        L = np.zeros_like(X)
        EY = np.zeros((len(X), kstep))
        EYW = np.zeros((len(X), kstep))
        
        for i in range(1, kstep+1):
            xgrid = np.linspace(self.lb[n+i], self.rb[n+i], self.xnum[n+i])
            ygrid = yknown[:self.xnum[n+i], i-1]

            EY[:,i-1], EYW[:,i-1] = self.GaussQuad_sinc(X, i*self.dt, xgrid, ygrid)   
        
        Z = np.matmul(EYW, alpha[1:]).reshape(len(X)) / self.dt
        Yex = np.matmul(EY, beta).reshape(len(X))
        Yeu = -1.0/alpha[0]*(np.matmul(EY, alpha[1:]).reshape(len(X)) + self.generator(self.tgrid[n], X, Yex, Z) * self.dt)
        
        Y = np.maximum(Yeu, self.hfun(self.tgrid[n], X))
        L = -alpha[0]/self.dt * np.maximum(self.hfun(self.tgrid[n], X) - Yeu, 0)

        # early_index = np.where(L > 0)[0][-1] if np.any(L > 0) else None
        early_index = np.where(Y-Yeu>1.0e-10)[0][-1] if np.any(Y > Yeu) else None
        self.early_price.append(self.GBM(self.tgrid[n], X[early_index]) if early_index is not None else None)
        self.early_t.append(self.tgrid[n])

        if early_index is not None:
            if self.GBM(self.tgrid[n], X[early_index]) > self.K:
                print('Early exercise at time t={:.4f}, stock price S={:.4f}'.format(self.tgrid[n], self.GBM(self.tgrid[n], X[early_index])), 
                      Y[early_index], Yeu[early_index], L[early_index])

        return Y, Z, L



    def exGA_solver(self, kstep):
        
        """explicit Genorator Approximation Solver for FBSDEs"""
        
        start = datetime.now()
        self.kstep = kstep
        self.early_price = []
        self.early_t = []

        self.yknown = np.zeros((self.xnum[-2], kstep))
        self.zknown = np.zeros((self.xnum[-2], kstep))
        self.lknown = np.zeros((self.xnum[-2], kstep))
        self.y = np.zeros((self.xnum[-2], self.N))
        self.z = np.zeros((self.xnum[-2], self.N))
        self.l = np.zeros((self.xnum[-2], self.N))
        self.s = np.zeros((self.xnum[-2], self.N))
        
        self.yknown, self.zknown, self.lknown = self.pre_compute(kstep)
        self.y[:,self.N-kstep:] = self.yknown
        self.z[:,self.N-kstep:] = self.zknown
        self.l[:,self.N-kstep:] = self.lknown
        
        
        for i in range(self.N-kstep-1, -1, -1):
            Y, Z, L = self.exGA_step(i, kstep, self.yknown)
            self.y[:self.xnum[i], i] = Y
            self.z[:self.xnum[i], i] = Z
            self.l[:self.xnum[i], i] = L
            
            for j in range(kstep-1, -1, -1):
                self.yknown[:,j] = self.yknown[:,j-1]
            if i>0:
                self.yknown[:self.xnum[i], 0] = Y
            
        end = datetime.now()
        self.running_time = end - start
        
        return
    

    # Crank-Nicolson method

    def CN_step(self, n, yknown, zknown):
        """
        0.5*dt*Z = -0.5 * dt * EZ + EYW + 0.5*dt*EFW
        Y = EY + 0.5*dt*f + 0.5*dt*EF
        """
        dt = self.dt
        t1 = self.tgrid[n]
        t2 = self.tgrid[n+1]
        
        X = np.linspace(self.lb[n], self.rb[n], self.xnum[n])
        Z = np.zeros_like(X)
        Y = np.zeros_like(X)
        Yeu = np.zeros_like(X)
        L = np.zeros_like(X)

        EY = np.zeros_like(X)
        EYW = np.zeros_like(X)
        EZ = np.zeros_like(X)
        EZW = np.zeros_like(X)
        EF = np.zeros_like(X)
        EFW = np.zeros_like(X)

        x2grid = np.linspace(self.lb[n+1], self.rb[n+1], self.xnum[n+1])
        f2grid = self.generator(t2, x2grid, yknown[:self.xnum[n+1]], zknown[:self.xnum[n+1]])

        EY, EYW = self.GaussQuad_sinc(X, dt, x2grid, yknown[:self.xnum[n+1]])
        EZ, EZW = self.GaussQuad_sinc(X, dt, x2grid, zknown[:self.xnum[n+1]])
        EF, EFW = self.GaussQuad_sinc(X, dt, x2grid, f2grid)
        
        Z = EFW-EZ + 2.0*EYW/dt

        for i in range(len(X)):
            func = lambda y: EY[i] + 0.5*self.generator(t1, X[i], y, Z[i]) * dt + 0.5*dt*EF[i] - y
            root = sp.optimize.root(func, EY[i])
            Yeu[i] = root.x[0]

        Y = np.maximum(Yeu, self.hfun(self.tgrid[n], X))
        L = 1.0/dt * np.maximum(self.hfun(self.tgrid[n], X) - Yeu, 0)

        # early_index = np.where(L > 0)[0][-1] if np.any(L > 0) else None
        early_index = np.where(Y-Yeu>1.0e-10)[0][-1] if np.any(Y > Yeu) else None
        self.early_price.append(self.GBM(self.tgrid[n], X[early_index]) if early_index is not None else None)
        self.early_t.append(self.tgrid[n])

        if self.GBM(self.tgrid[n], X[early_index]) > self.K:
            print('Early exercise at time t={:.4f}, stock price S={:.4f}'.format(self.tgrid[n], self.GBM(self.tgrid[n], X[early_index])), 
                  Y[early_index], Yeu[early_index], L[early_index])

        return Y, Z, L


    def pre_CN_step(self):

        xgrid = np.linspace(self.lb[self.N]-2, self.rb[self.N]+2, 10*self.xnum[self.N])
        ygrid = self.terminal(xgrid)
        zgrid = self.zterminal(xgrid)
        fgrid = self.generator(self.T, xgrid, ygrid, zgrid)

        # yknown, zknown, lknown = self.CN_step(self.N-1, ygrid, zgrid)
        n = self.N - 1
        X = np.linspace(self.lb[n], self.rb[n], self.xnum[n])
        Z = np.zeros_like(X)
        Y = np.zeros_like(X)
        Yeu = np.zeros_like(X)
        L = np.zeros_like(X)
        
        EY = np.zeros_like(X)
        EYW = np.zeros_like(X)
        EZ = np.zeros_like(X)
        EZW = np.zeros_like(X)
        EF = np.zeros_like(X)
        EFW = np.zeros_like(X)


        EY, EYW = self.GaussQuad_sinc(X, self.dt, xgrid, ygrid)
        EZ, EZW = self.GaussQuad_sinc(X, self.dt, xgrid, zgrid)
        EF, EFW = self.GaussQuad_sinc(X, self.dt, xgrid, fgrid)
        
        Z = EFW-EZ + 2.0*EYW/self.dt

        for i in range(len(X)):
            func = lambda y: EY[i] + 0.5*self.generator(self.tgrid[self.N-1], X[i], y, Z[i]) * self.dt + 0.5*self.dt*EF[i] - y
            root = sp.optimize.root(func, EY[i])
            Yeu[i] = root.x[0]

        Y = np.maximum(Yeu, self.hfun(self.tgrid[self.N-1], X))
        L = 1.0/self.dt * np.maximum(self.hfun(self.tgrid[self.N-1], X) - Yeu, 0)

        # early_index = np.where(L > 0)[0][-1] if np.any(L > 0) else None
        early_index = np.where(Y-Yeu>1.0e-10)[0][-1] if np.any(Y > Yeu) else None
        self.early_price.append(self.GBM(self.tgrid[n], X[early_index]) if early_index is not None else None)
        self.early_t.append(self.tgrid[n])
        
        if self.GBM(self.tgrid[n], X[early_index]) > self.K:
            print('Early exercise at time t={:.4f}, stock price S={:.4f}'.format(self.tgrid[n], self.GBM(self.tgrid[n], X[early_index])), 
                  Y[early_index], Yeu[early_index], L[early_index])

        # use true solution as terminal condition
        # xsolve = np.linspace(self.lb[self.N-1], self.rb[self.N-1], self.xnum[self.N-1])
        # St_solve = self.GBM(self.tgrid[self.N-1], xsolve)
        # yknown = BlackScholesCall(St_solve, self.K, self.T - self.tgrid[self.N-1], self.r, self.sigma, self.q)
        # zknown = DeltaBlackScholesCall(St_solve, self.K, self.T - self.tgrid[self.N-1], self.r, self.sigma, self.q) \
        #                 *self.sigma*St_solve
        # lknown = np.zeros_like(yknown)
        
        return Y, Z, L
    
    
    def CN_solver(self):

        start = datetime.now()
        
        self.early_price = []
        self.early_t = []

        self.y = np.zeros((self.xnum[-2], self.N))
        self.z = np.zeros((self.xnum[-2], self.N))
        self.l = np.zeros((self.xnum[-2], self.N))
        
        self.yknown, self.zknown, self.lknown = self.pre_CN_step()
        self.y[:,self.N-1] = self.yknown
        self.z[:,self.N-1] = self.zknown
        self.l[:,self.N-1] = self.lknown
        

        for i in range(self.N-2, -1, -1):
            Y, Z, L = self.CN_step(i, self.yknown, self.zknown)
            # print('xnum[i]: ', self.xnum[i], ' len(Y): ', len(Y))
            self.y[:self.xnum[i], i] = Y
            self.z[:self.xnum[i], i] = Z
            self.l[:self.xnum[i], i] = L
            self.yknown = Y
            self.zknown = Z
            self.lknown = L
        
        end = datetime.now()
        self.running_time = end - start

        return




    def convergence_rate(self, dt, error):
        """Compute the convergence rate given dt and error arrays"""
        n = len(dt)
        dt_col = dt.reshape(n, 1)
        err_col = error.reshape(n, 1)

        Xmatrix = np.hstack((np.ones((n,1)), np.log(dt_col)))

        beta = np.matmul(np.linalg.inv(np.matmul(Xmatrix.T, Xmatrix)), (np.matmul(Xmatrix.T, np.log(err_col))))
        
        rate = beta[1]            

        return rate


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
    
    
    # compute the expectation by sinc method
    def GaussQuad_sinc(self, x, dt, xsinc, ysinc):
        """
        Computes the conditional expectation E[u(x + w)], where w is a Brownian motion.
        x is a fixed vector of points where the expectation is computed. 
        The function uses sinc interpolation to compute the expectation at the points xsinc, which are uniformly spaced.
        ysinc is the corresponding values of u(x + kh) at the points xsinc.
        """
        n = len(x)
        m = len(xsinc)
        EY = np.zeros_like(x)
        EYW = np.zeros_like(x)
        
        x_grid = np.ones((m,1)) * x.reshape((1,n))
        xsinc_grid = xsinc.reshape((m,1)) * np.ones((1,n))
        ysinc_grid = ysinc.reshape((m,1)) * np.ones((1,n))
        
        dist = xsinc_grid - x_grid
        EY_grid = ysinc_grid * 1.0/np.sqrt(2*np.pi)/np.sqrt(dt)*np.exp(-0.5 * dist**2 / dt)
        EYW_grid = ysinc_grid * 1.0/np.sqrt(2*np.pi)/np.sqrt(dt)*np.exp(-0.5 * dist**2 / dt) * dist
        EY = np.sum(EY_grid, axis=0) * (xsinc[1] - xsinc[0])
        EYW = np.sum(EYW_grid, axis=0) * (xsinc[1] - xsinc[0])

        return EY, EYW
    
    def print_result(self):
        # Compute the error between the numerical solution and the true solution
        err = np.abs(self.y[0,0] - self.put_true)
        print(f"dt = {self.dt:10.8f}, \t dh={self.dh:10.8f}, \t Y= {self.y[0,0]:10.8e}, \t True Put= {self.put_true:10.8e}, \t Error= {err:10.8e}")
        return


def BlackScholesCall(S0, K, T, r, sigma, q):
    """Compute the Black-Scholes price of a European call option"""
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def DeltaBlackScholesCall(S0, K, T, r, sigma, q):
    """Compute the Delta of a European call option using Black-Scholes formula"""
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = np.exp(-q * T) * norm.cdf(d1)
    return delta

def BlackScholesPut(S0, K, T, r, sigma, q): 
    """Compute the Black-Scholes price of a European put option"""
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return put_price

def DeltaBlackScholesPut(S0, K, T, r, sigma, q): 
    """Compute the Delta of a European put option using Black-Scholes formula"""
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    return delta

example1 = AmericanCallOption(
    S0=100, 
    K=100, 
    T=1.0, 
    mu=0.5, 
    sigma=0.2, 
    r=0.05, 
    q=0.045)

example1.call_true = 7.881905800537484

step_num = [40, 80, 120, 160, 200]
dt_vec = example1.T/(np.array(step_num))
rt_vec = np.zeros(len(step_num))
ynum = np.zeros(len(step_num))
err = np.zeros(len(step_num))

for n in step_num:
    example1.N = n
    example1.generate_grid(m=10)
    example1.exGA_solver(kstep=2)
    # example1.CN_solver()
    example1.print_result()
    ynum[step_num.index(n)] = example1.y[0,0]
    err[step_num.index(n)] = np.abs(example1.y[0,0] - example1.call_true)
    rt_vec[step_num.index(n)] = example1.running_time.total_seconds()
    with open('rbsde_call_boundary.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(example1.early_t)
        writer.writerow(example1.early_price)

with open('rbsde_call_boundary.csv', 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["=" * 100])
    writer.writerow(['step_num:', step_num])
    writer.writerow(["=" * 100])

rate = example1.convergence_rate(dt_vec, err)
print(f"Convergence rate: {rate[0]:10.8f}")

with open('AmOptionResult.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["American Option Pricing - multi-step explicit GA method"])
        writer.writerow(["Current time: ", datetime.now()])
        writer.writerow(["=" * 50])
        writer.writerow(["S0=", example1.S0])
        writer.writerow(["K=", example1.K])
        writer.writerow(["T=", example1.T])
        writer.writerow(["r=", example1.r])
        writer.writerow(["sigma=", example1.sigma])
        writer.writerow(["q=", example1.q])
        writer.writerow(["kstep of the scheme=", example1.kstep])
        writer.writerow(["-" * 50])
        writer.writerow(['Reference call price', f"{example1.call_true:10.8f}"])
        for i in range(len(step_num)):
            writer.writerow([f"time steps:", step_num[i], f"dt:", f"{dt_vec[i]:10.5e}", 'numerical call price:', 
                             f"{ynum[i]:10.5e}", 'error:', f"{err[i]:10.5e}", 'running time:', f"{rt_vec[i]:10.8f}"])
        writer.writerow(["-" * 50])
        writer.writerow([f"Convergence rate: {rate[0]:10.8f}"])
        writer.writerow(["=" * 50])


example2 = AmericanPutOption(
    S0=100, 
    K=100, 
    T=1.0, 
    mu=0.5, 
    sigma=0.2, 
    r=0.05, 
    q=0.045)

# example.put_true = BlackScholesPut(example.S0, example.K, example.T, example.r, example.sigma, example.q)
example2.put_true = 7.4809789340178146

step_num = [40, 80, 120, 160, 200]
dt_vec = example2.T/(np.array(step_num))
rt_vec = np.zeros(len(step_num))
ynum = np.zeros(len(step_num))
err = np.zeros(len(step_num))

for n in step_num:
    example2.N = n
    example2.generate_grid(m=10)
    example2.exGA_solver(kstep=3)
    # example2.CN_solver()
    example2.print_result()
    ynum[step_num.index(n)] = example2.y[0,0]
    err[step_num.index(n)] = np.abs(example2.y[0,0] - example2.put_true)
    rt_vec[step_num.index(n)] = example2.running_time.total_seconds()
    with open('rbsde_put_boundary.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(example2.early_t)
        writer.writerow(example2.early_price)

with open('rbsde_put_boundary.csv', 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["=" * 100])
    writer.writerow(['step_num:', step_num])
    writer.writerow(["=" * 100])


rate = example2.convergence_rate(dt_vec, err)
print(f"Convergence rate: {rate[0]:10.8f}")

with open('AmOptionResult.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["American Option Pricing - multi-step explicit GA method"])
        writer.writerow(["Current time: ", datetime.now()])
        writer.writerow(["=" * 50])
        writer.writerow(["S0=", example2.S0])
        writer.writerow(["K=", example2.K])
        writer.writerow(["T=", example2.T])
        writer.writerow(["r=", example2.r])
        writer.writerow(["sigma=", example2.sigma])
        writer.writerow(["q=", example2.q])
        writer.writerow(["kstep of the scheme=", example2.kstep])
        writer.writerow(["-" * 50])
        writer.writerow(['Reference put price', f"{example2.put_true:10.8f}"])
        for i in range(len(step_num)):
            writer.writerow([f"time steps:", step_num[i], f"dt:", f"{dt_vec[i]:10.8f}", 'numerical put price:', 
                             f"{ynum[i]:10.5e}", 'error:', f"{err[i]:10.5e}", 'running time:', f"{rt_vec[i]:10.8f}"])
        writer.writerow(["-" * 50])
        writer.writerow([f"Convergence rate: {rate[0]:10.8f}"])
        writer.writerow(["=" * 50])