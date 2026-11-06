import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class RFBSDE_dcpl:
    def __init__(self, t0, T, mu, sigma, f, g, init, ytrue, ztrue, ylb, yub):
        self.t0 = t0
        self.T = T
        self.mu = mu
        self.sigma = sigma
        self.f = f
        self.g = g
        self.init = init
        self.ytrue = ytrue
        self.ztrue = ztrue
        self.ylb = ylb  # lower bound for y
        self.yub = yub  # upper bound for y
        # paramters for the FBSDE solver
        self.N = 100  # number of time steps
        self.dt = (self.T - self.t0) / self.N  # time step size
        self.dh = 0.1  # grid size for x
        self.tgrid = None  # time grid
        self.lb = None  # lower bound for x
        self.ub = None  # upper bound for x
        self.xnum = None  # number of grid points for x
        self.mu_bound = 0.0  # bounds for mu default is 0, which means no drift
        self.sigma_bound = 1.0  # bounds for sigma default is 1, which means a standard Brownian motion
        self.X = None  # solution for x which coincides with the self.init
        self.Y = None  # solution for y at t=t0
        self.Z = None  # solution for z at t=t0
        self.L = None 
        self.Yerr = None  # error for y at t=t0
        self.Zerr = None  # error for z at t=t0
        self.Ysurface = None  # solution surface for y which coincides with the tgrid and lb, ub.
        self.Zsurface = None  # solution surface for z
        self.Lsurface = None  # solution surface for lambda
        self.ynum_lb = None
        self.ynum_ub = None
        
        


    def drift(self, t, x):
        # Placeholder for the drift function f(t, x)
        return self.mu(t, x)
    
    def diffusion(self, t, x):
        # Placeholder for the diffusion function sigma(t, x)
        return self.sigma(t, x)

    def terminal(self, x):
        # Placeholder for the terminal condition function
        return self.g(self.T, x) #terminal_condition
    
    def generator(self, t, x, y, z):
        # Placeholder for the generator function
        return self.f(t, x, y, z)

    def plot_solution(self):
        plt.plot(self.X, self.Y, 'x',label='Ynum')
        plt.plot(self.X, self.ytrue(self.t0,self.X), label='Ytrue')
        plt.plot(self.X, self.Z, 'x',label='Znum')
        plt.plot(self.X, self.ztrue(self.t0, self.X), label='Ztrue')
        plt.grid()
        plt.legend()
        plt.show()
        return
    
    def plot_y(self):
        plt.plot(self.X, self.Y, 'x', label='Ynum')
        plt.plot(self.X, self.ytrue(self.t0, self.X), label='Ytrue')
        plt.grid()
        plt.legend()
        plt.show()
        return
    
    def plot_z(self):
        plt.plot(self.X, self.Z, 'x', label='Znum')
        plt.plot(self.X, self.ztrue(self.t0, self.X), label='Ztrue')
        plt.grid()
        plt.legend()
        plt.show()
        return


    # tbd
    def plot_Ysurface(self):
        plt.imshow(self.Ysurface, aspect='auto', origin='lower', extent=(self.tgrid[0], self.tgrid[-2], self.lb[-1], self.ub[-1]), cmap='hot')
        plt.colorbar(label='Y')
        plt.xlabel('Time')
        plt.ylabel('X')
        plt.title('Y surface')
        plt.show()
    # tbd    
    def plot_Zsurface(self):
        plt.imshow(self.Zsurface, aspect='auto', origin='lower', extent=(self.tgrid[0], self.tgrid[-2], self.lb[-1], self.ub[-1]), cmap='hot')
        plt.colorbar(label='Z')
        plt.xlabel('Time')
        plt.ylabel('X')
        plt.title('Z surface')
        plt.show()

     # tbd    
    def plot_Lsurface(self):
        plt.imshow(self.Lsurface, aspect='auto', origin='lower', extent=(self.tgrid[0], self.tgrid[-2], self.lb[-1], self.ub[-1]), cmap='hot')
        plt.colorbar(label='L')
        plt.xlabel('Time')
        plt.ylabel('X')
        plt.title('L surface')
        plt.show()



    def print_error(self):
        # Compute the error between the numerical solution and the true solution
        self.Yerr = np.max(np.abs(self.Y - self.ytrue(self.t0, self.X)))
        self.Zerr = np.max(np.abs(self.Z - self.ztrue(self.t0, self.X)))
        print(f"dt = {self.dt:10.8f}, \t dh={self.dh:10.8f}, \t Yerror= {self.Yerr:10.5e}, \t Zerror = {self.Zerr:10.5e}")
        return
    

    def fixed_grid(self, lowerbound, upperbound):
        dh = self.dh
        self.tgrid = np.linspace(self.t0, self.T, self.N+1)
        self.lb = np.zeros_like(self.tgrid)
        self.ub = np.zeros_like(self.tgrid)
        self.xnum = np.zeros(len(self.tgrid), dtype=int)
        self.ynum_lb = np.zeros(self.N)
        self.ynum_ub = np.zeros(self.N)

        self.lb[1:] = lowerbound
        self.ub[1:] = upperbound
        self.xnum[1:] = np.ceil((upperbound-lowerbound)/dh)+1
        self.ub[1:] = self.lb[1:] + (self.xnum[1:]-1)*dh

        self.xnum[0] = len(self.init)
        self.lb[0] = np.min(self.init)
        self.ub[0] = np.max(self.init)

        return
    # tbd
    def generate_grid(self, ndh):
        
        dh = self.dh
        mu_bound = self.mu_bound
        sigma_bound = self.sigma_bound
        self.tgrid = np.linspace(self.t0, self.T, self.N+1)
        self.lb = np.zeros_like(self.tgrid)
        self.ub = np.zeros_like(self.tgrid)
        self.ynum_lb = np.zeros(self.N)
        self.ynum_ub = np.zeros(self.N)
        self.xnum = np.zeros(len(self.tgrid), dtype=int)

        self.xnum[0] = len(self.init)
        self.lb[0] = np.min(self.init)
        self.ub[0] = np.max(self.init)

        for i in range(1, len(self.tgrid)):
            self.lb[i] = self.lb[i-1] + mu_bound * self.dt - sigma_bound * np.sqrt(self.dt)*ndh
            self.ub[i] = self.ub[i-1] + mu_bound * self.dt + sigma_bound * np.sqrt(self.dt)*ndh
            self.xnum[i] = np.ceil((self.ub[i] - self.lb[i]) / dh) + 1
            self.ub[i] = self.lb[i] + (self.xnum[i]-1)*dh
            
        self.Ysurface = np.zeros((self.xnum[-2], self.N))
        self.Zsurface = np.zeros((self.xnum[-2], self.N))
        self.Lsurface = np.zeros((self.xnum[-2], self.N))

        return

    


    # implicit Euler method
    
    def pre_Euler_step(self, t1, lb, ub, xnum):
        dt = self.dt
        dh = self.dh
        alpha = self.get_alpha(1)
        xgrid = np.linspace(lb, ub, xnum)
        Z = np.zeros(len(xgrid))
        Y = np.zeros(len(xgrid))
        L = np.zeros(len(xgrid))
        
        xsinc = np.arange(lb-10*dh, ub+10*dh, dh)
        ysinc = self.terminal(xsinc)
        EY, EYW = self.GaussQuad_sinc(xgrid, self.drift(t1, xgrid), self.diffusion(t1, xgrid), dt, xsinc, ysinc)
        Z = EYW/dt
        for i in range(len(xgrid)):
            func = lambda y: y - EY[i] - self.generator(t1, xgrid[i], y, Z[i])*dt
            root = sp.optimize.root(func, EY[i])
            Y[i] = root.x[0]
            Y[i] = np.maximum(Y[i], 0)
            L[i] = -alpha[0]/self.dt * np.maximum(- Y[i], 0)
        
        return xgrid, Y, Z, L
    
    def Euler_step(self, t1, lb, ub, xnum, t2, x2grid, y2grid, z2grid):
        dt = self.dt
        x1grid = np.linspace(lb, ub, xnum)
        alpha = self.get_alpha(1)
        Z = np.zeros_like(x1grid)
        Y = np.zeros_like(x1grid)
        L = np.zeros_like(x1grid)

        EY, EYW = self.GaussQuad_sinc(x1grid, self.drift(t1, x1grid), self.diffusion(t1, x1grid), dt, x2grid, y2grid)
        Z = EYW / dt

        for i in range(len(x1grid)):
            func = lambda y: y - EY[i] - self.generator(t1, x1grid[i], y, Z[i]) * dt
            root = sp.optimize.root(func, EY[i])
            Y[i] = root.x[0]
            Y[i] = np.maximum(Y[i], 0)
            L[i] = -alpha[0]/self.dt * np.maximum(- Y[i], 0)

        return x1grid, Y, Z, L
    
    def Euler_solver(self):
        
        X, Y, Z, L = self.pre_Euler_step(self.tgrid[-2], self.lb[-2], self.ub[-2], self.xnum[-2])

        for i in range(self.N-2, -1, -1):
            t1 = self.tgrid[i]
            t2 = self.tgrid[i+1]
            X, Y, Z, L = self.Euler_step(t1, self.lb[i], self.ub[i], self.xnum[i], t2, X, Y, Z)
        
        self.X = X
        self.Y = Y
        self.Z = Z
        self.L = L

        
        return



    # implicit Genorator approximation method

    def pre_GenoratorApp_step(self, kstep):
        dt = self.dt
        dh = self.dh

        yknown = np.zeros((self.xnum[-2], kstep))
        zknown = np.zeros((self.xnum[-2], kstep))

        terminal_xgrid = np.linspace(self.lb[-1], self.ub[-1], self.xnum[-1])
        terminal_ygrid = self.terminal(terminal_xgrid)

        for i in range(kstep):
            n = self.N - 1 - i
            xsolve = np.linspace(self.lb[n], self.ub[n], self.xnum[n])
            ysolve = np.zeros_like(xsolve)
            zsolve = np.zeros_like(xsolve)
            ygrid = np.zeros((self.xnum[-1], i+1))
            
            ygrid[:,i] = terminal_ygrid

            if i == 0:
                ysolve, zsolve = self.GenoratorApp_step(1, n, terminal_ygrid)
                yknown[:self.xnum[n], kstep-1] = ysolve
            else:
                for j in range(i):
                    ygrid[:self.xnum[n+j+1], i-1-j] = yknown[:self.xnum[n+j+1], kstep-1-j]
                ysolve, zsolve = self.GenoratorApp_step(i+1, n, ygrid)
                yknown[:self.xnum[n], kstep-1-i] = ysolve
                zknown[:self.xnum[n], kstep-1-i] = zsolve

        return yknown, zknown
    
    def pre_GenoratorAppCN_step(self, kstep):
        dt = self.dt
        dh = self.dh

        yknown = np.zeros((self.xnum[-2], kstep))
        zknown = np.zeros((self.xnum[-2], kstep))

        terminal_xgrid = np.linspace(self.lb[-1], self.ub[-1], self.xnum[-1])
        terminal_ygrid = self.terminal(terminal_xgrid)
        terminal_zgrid = self.ztrue(self.T, terminal_xgrid)

        for i in range(kstep):
            n = self.N - 1 - i
            xsolve = np.linspace(self.lb[n], self.ub[n], self.xnum[n])
            ysolve = np.zeros_like(xsolve)
            zsolve = np.zeros_like(xsolve)
            ygrid = np.zeros((self.xnum[-1], i+1))
            
            ygrid[:,i] = terminal_ygrid

            if i == 0:
                ysolve, zsolve = self.CN_solver_step(n, terminal_ygrid, terminal_zgrid)
                yknown[:self.xnum[n], kstep-1] = ysolve
            else:
                for j in range(i):
                    ygrid[:self.xnum[n+j+1], i-1-j] = yknown[:self.xnum[n+j+1], kstep-1-j]
                ysolve, zsolve = self.GenoratorApp_step(i+1, n, ygrid)
                yknown[:self.xnum[n], kstep-1-i] = ysolve
                zknown[:self.xnum[n], kstep-1-i] = zsolve

        return yknown, zknown
    
    

    def GenoratorApp_step(self, kstep, n1, yknown):
        dt = self.dt
        t1 = self.tgrid[n1]

        yknown = yknown.reshape((yknown.shape[0], kstep))
        
        alpha = self.get_alpha(kstep)

        X = np.linspace(self.lb[n1], self.ub[n1], self.xnum[n1])
        Z = np.zeros_like(X)
        Y = np.zeros_like(X)
        L = np.zeros_like(X)
        EY = np.zeros((len(X), kstep))
        EYW = np.zeros((len(X), kstep))

        for i in range(1, kstep+1):
            t2 = self.tgrid[n1+i]
            x2grid = np.linspace(self.lb[n1+i], self.ub[n1+i], self.xnum[n1+i])
            y2grid = yknown[:self.xnum[n1+i], i-1]

            EY[:,i-1], EYW[:,i-1] = self.GaussQuad_sinc(X, self.drift(t1, X), self.diffusion(t1, X), i*dt, x2grid, y2grid)
    
        Z = np.matmul(EYW, alpha[1:]).reshape(len(X)) / dt


        for i in range(len(X)):
            func = lambda y: alpha[0]*y + np.dot(alpha[1:], EY[i,:]) + self.generator(t1, X[i], y, Z[i]) * dt
            root = sp.optimize.root(func, np.mean(EY[i, :]))
            Y[i] = root.x[0]
            Y[i] = np.maximum(Y[i], 0)
            L[i] = -alpha[0]/self.dt * np.maximum(- Y[i], 0)

        return Y, Z, L

    def GenoratorApp_solver(self, kstep):
        
        yknown, zknown = self.pre_GenoratorApp_step(kstep)
        self.Ysurface[:, self.N-kstep:] = yknown
        self.Zsurface[:, self.N-kstep:] = zknown
            
        for i in range(self.N-kstep-1, -1, -1):
            Y, Z, L = self.GenoratorApp_step(kstep, i, yknown)
            self.Ysurface[:self.xnum[i], i] = Y
            self.Zsurface[:self.xnum[i], i] = Z
            self.Lsurface[:self.xnum[i], i] = L
            for j in range(kstep-1, -1, -1):
                yknown[:,j] = yknown[:,j-1]
            if i>0:
                yknown[:self.xnum[i], 0] = Y
        
        self.X = self.init
        self.Y = Y
        self.Z = Z
        self.L = L
        self.Ysurface[:self.xnum[0], 0] = Y
        self.Zsurface[:self.xnum[0], 0] = Z
        self.Lsurface[:self.xnum[0], 0] = L
        
        return

    # with true terminal steps
    def GenoratorAppTrue_solver(self, kstep):
        
        yknown = np.zeros((self.xnum[-2], kstep))
        zknown = np.zeros((self.xnum[-2], kstep))
        for i in range(kstep):
            xgrid = np.linspace(self.lb[self.N-i-1], self.ub[self.N-i-1], self.xnum[self.N-i-1])
            yknown[:self.xnum[self.N-i-1], kstep-1-i] = self.ytrue(self.tgrid[self.N-i-1], xgrid)
            zknown[:self.xnum[self.N-i-1], kstep-1-i] = self.ztrue(self.tgrid[self.N-i-1], xgrid)
        self.Ysurface[self.N-i-1, :self.xnum[self.N-i-1]] = yknown[:self.xnum[self.N-i-1], kstep-1-i]
        self.Zsurface[self.N-i-1, :self.xnum[self.N-i-1]] = zknown[:self.xnum[self.N-i-1], kstep-1-i]
        
        for i in range(self.N-kstep-1, -1, -1):
            Y, Z, L = self.GenoratorApp_step(kstep, i, yknown)
            self.Ysurface[:self.xnum[i], i] = Y
            self.Zsurface[:self.xnum[i], i] = Z
            self.Lsurface[:self.xnum[i], i] = L
            for j in range(kstep-1, -1, -1):
                yknown[:,j] = yknown[:,j-1]
            if i>0:
                yknown[:self.xnum[i], 0] = Y
            
        self.X = self.init
        self.Y = Y
        self.Z = Z
        self.L = L
        self.Ysurface[:self.xnum[0], 0] = Y
        self.Zsurface[:self.xnum[0], 0] = Z
        self.Lsurface[:self.xnum[0], 0] = L
        
        return

    # with Crank-Nicolson for the first step
    def GenoratorAppCN_solver(self, kstep):
        
        yknown, zknown = self.pre_GenoratorAppCN_step(kstep)
        self.Ysurface[:, self.N-kstep:] = yknown
        self.Zsurface[:, self.N-kstep:] = zknown

            
        for i in range(self.N-kstep-1, -1, -1):
            Y, Z, L = self.GenoratorApp_step(kstep, i, yknown)
            self.Ysurface[:self.xnum[i], i] = Y
            self.Zsurface[:self.xnum[i], i] = Z
            self.Lsurface[:self.xnum[i], i] = L
            for j in range(kstep-1, -1, -1):
                yknown[:,j] = yknown[:,j-1]
            if i>0:
                yknown[:self.xnum[i], 0] = Y
        
        self.X = self.init
        self.Y = Y
        self.Z = Z
        self.L = L
        self.Ysurface[:self.xnum[0], 0] = Y
        self.Zsurface[:self.xnum[0], 0] = Z
        self.Lsurface[:self.xnum[0], 0] = L
        
        return





    # explicit Genorator approximation method
    def exGenoratorApp_step(self, kstep, n1, yknown):
        dt = self.dt
        t1 = self.tgrid[n1]

        yknown = yknown.reshape((yknown.shape[0], kstep))
        
        alpha = self.get_alpha(kstep)
        beta = self.get_beta(kstep)

        X = np.linspace(self.lb[n1], self.ub[n1], self.xnum[n1])
        Z = np.zeros_like(X)
        Y = np.zeros_like(X)
        L = np.zeros_like(X)
        EY = np.zeros((len(X), kstep))
        EYW = np.zeros((len(X), kstep))

        for i in range(1, kstep+1):
            t2 = self.tgrid[n1+i]
            x2grid = np.linspace(self.lb[n1+i], self.ub[n1+i], self.xnum[n1+i])
            y2grid = yknown[:self.xnum[n1+i], i-1]

            EY[:,i-1], EYW[:,i-1] = self.GaussQuad_sinc(X, self.drift(t1, X), self.diffusion(t1, X), i*dt, x2grid, y2grid)
    
        Z = np.matmul(EYW, alpha[1:]).reshape(len(X)) / dt
        
        Yex = np.matmul(EY, beta).reshape(len(X))


        Y = -1.0/alpha[0]*(np.matmul(EY, alpha[1:]).reshape(len(X)) + self.generator(t1, X, Yex, Z) * dt)
        Y = np.maximum(Y, 0)
        L = -alpha[0]/self.dt * np.maximum(-Y, 0)

        return Y, Z, L

    # with Crank-Nicolson for the first step
    def pre_exGenoratorAppCN_step(self, kstep):
        
        yknown = np.zeros((self.xnum[-2], kstep))
        zknown = np.zeros((self.xnum[-2], kstep))

        terminal_xgrid = np.linspace(self.lb[-1], self.ub[-1], self.xnum[-1])
        terminal_ygrid = self.terminal(terminal_xgrid)
        terminal_zgrid = self.ztrue(self.T, terminal_xgrid)

        for i in range(kstep):
            n = self.N - 1 - i
            xsolve = np.linspace(self.lb[n], self.ub[n], self.xnum[n])
            ysolve = np.zeros_like(xsolve)
            zsolve = np.zeros_like(xsolve)
            lsolve = np.zeros_like(xsolve)
            ygrid = np.zeros((self.xnum[-1], i+1))
            
            ygrid[:,i] = terminal_ygrid

            if i == 0:
                ysolve, zsolve, lsolve = self.CN_step(n, terminal_ygrid, terminal_zgrid)
                yknown[:self.xnum[n], kstep-1] = ysolve
                zknown[:self.xnum[n], kstep-1] = zsolve
                self.ynum_lb[n] = np.min(ysolve)
                self.ynum_ub[n] = np.max(ysolve)
            else:
                for j in range(i):
                    ygrid[:self.xnum[n+j+1], i-1-j] = yknown[:self.xnum[n+j+1], kstep-1-j]
                ysolve, zsolve, lsolve = self.exGenoratorApp_step(i+1, n, ygrid)
                yknown[:self.xnum[n], kstep-1-i] = ysolve
                zknown[:self.xnum[n], kstep-1-i] = zsolve
                self.ynum_lb[n] = np.min(ysolve)
                self.ynum_ub[n] = np.max(ysolve)

        return yknown, zknown
    
    def pre_exGenoratorApp_step(self, kstep):
        
        yknown = np.zeros((self.xnum[-2], kstep))
        zknown = np.zeros((self.xnum[-2], kstep))

        terminal_xgrid = np.linspace(self.lb[-1], self.ub[-1], self.xnum[-1])
        terminal_ygrid = self.terminal(terminal_xgrid)
        terminal_zgrid = self.ztrue(self.T, terminal_xgrid)

        for i in range(kstep):
            n = self.N - 1 - i
            xsolve = np.linspace(self.lb[n], self.ub[n], self.xnum[n])
            ysolve = np.zeros_like(xsolve)
            zsolve = np.zeros_like(xsolve)
            ygrid = np.zeros((self.xnum[-1], i+1))
            
            ygrid[:,i] = terminal_ygrid

            if i == 0:
                ysolve, zsolve = self.exGenoratorApp_step(1, n, terminal_ygrid)
                yknown[:self.xnum[n], kstep-1] = ysolve
                zknown[:self.xnum[n], kstep-1] = zsolve
                self.ynum_lb[n] = np.min(ysolve)
                self.ynum_ub[n] = np.max(ysolve)
            else:
                for j in range(i):
                    ygrid[:self.xnum[n+j+1], i-1-j] = yknown[:self.xnum[n+j+1], kstep-1-j]
                ysolve, zsolve = self.exGenoratorApp_step(i+1, n, ygrid)
                yknown[:self.xnum[n], kstep-1-i] = ysolve
                zknown[:self.xnum[n], kstep-1-i] = zsolve
                self.ynum_lb[n] = np.min(ysolve)
                self.ynum_ub[n] = np.max(ysolve)

        return yknown, zknown



    def exGenoratorApp_solver(self, kstep):
        
        yknown, zknown = self.pre_exGenoratorApp_step(kstep)
        self.Ysurface[:, self.N-kstep:] = yknown
        self.Zsurface[:, self.N-kstep:] = zknown

        for i in range(self.N-kstep-1, -1, -1):
            Y, Z, L = self.exGenoratorApp_step(kstep, i, yknown)
            self.Ysurface[:self.xnum[i], i] = Y
            self.Zsurface[:self.xnum[i], i] = Z
            self.Lsurface[:self.xnum[i], i] = L
            
            for j in range(kstep-1, -1, -1):
                yknown[:,j] = yknown[:,j-1]
            if i>0:
                yknown[:self.xnum[i], 0] = Y
        
        self.X = self.init
        self.Y = Y
        self.Z = Z
        self.L = L
        
        return

    # with Crank-Nicolson for the first step
    def exGenoratorAppCN_solver(self, kstep):
        
        yknown, zknown = self.pre_exGenoratorAppCN_step(kstep)
        self.Ysurface[:, self.N-kstep:] = yknown
        self.Zsurface[:, self.N-kstep:] = zknown

        for i in range(self.N-kstep-1, -1, -1):
            Y, Z, L = self.exGenoratorApp_step(kstep, i, yknown)
            self.Ysurface[:self.xnum[i], i] = Y
            self.Zsurface[:self.xnum[i], i] = Z
            self.Lsurface[:self.xnum[i], i] = L
            self.ynum_lb[i] = np.min(Y)
            self.ynum_ub[i] = np.max(Y)
            
            for j in range(kstep-1, -1, -1):
                yknown[:,j] = yknown[:,j-1]
            if i>0:
                yknown[:self.xnum[i], 0] = Y
        
        self.X = self.init
        self.Y = Y
        self.Z = Z
        self.L = L
        
        return

    # with terminal steps from the true solution
    def exGenoratorAppTrue_solver(self, kstep):
        
        yknown = np.zeros((self.xnum[-2], kstep))
        zknown = np.zeros((self.xnum[-2], kstep))
        for i in range(kstep):
            xgrid = np.linspace(self.lb[self.N-i-1], self.ub[self.N-i-1], self.xnum[self.N-i-1])
            yknown[:self.xnum[self.N-i-1], kstep-1-i] = self.ytrue(self.tgrid[self.N-i-1], xgrid)
            zknown[:self.xnum[self.N-i-1], kstep-1-i] = self.ztrue(self.tgrid[self.N-i-1], xgrid)
        self.Ysurface[:self.xnum[self.N-i-1], self.N-i-1] = yknown[:self.xnum[self.N-i-1], kstep-1-i]
        self.Zsurface[:self.xnum[self.N-i-1], self.N-i-1] = zknown[:self.xnum[self.N-i-1], kstep-1-i]
        
        for i in range(self.N-kstep-1, -1, -1):
            Y, Z, L = self.exGenoratorApp_step(kstep, i, yknown)
            self.Ysurface[:self.xnum[i], i] = Y
            self.Zsurface[:self.xnum[i], i] = Z
            self.Lsurface[:self.xnum[i], i] = L
            
            for j in range(kstep-1, -1, -1):
                yknown[:,j] = yknown[:,j-1]
            if i>0:
                yknown[:self.xnum[i], 0] = Y
        
        self.X = self.init
        self.Y = Y
        self.Z = Z
        self.L = L
        
        return


    # Crank-Nicolson method

    def CN_step(self, n, yknown, zknown):
        """
        0.5*dt*Z = -0.5 * dt * EZ + EYW + 0.5*dt*EFW
        Y = EY + 0.5*dt*f + 0.5*dt*EF
        """
        dt = self.dt
        dh = self.dh
        t1 = self.tgrid[n]
        t2 = self.tgrid[n+1]
        
        X = np.linspace(self.lb[n], self.ub[n], self.xnum[n])
        Z = np.zeros_like(X)
        Y = np.zeros_like(X)
        L = np.zeros_like(X)
        EY = np.zeros_like(X)
        EYW = np.zeros_like(X)
        EZ = np.zeros_like(X)
        EZW = np.zeros_like(X)
        EF = np.zeros_like(X)
        EFW = np.zeros_like(X)
        

        x2grid = np.linspace(self.lb[n+1], self.ub[n+1], self.xnum[n+1])
        f2grid = self.generator(t2, x2grid, yknown[:self.xnum[n+1]], zknown[:self.xnum[n+1]])

        EY, EYW = self.GaussQuad_sinc(X, self.drift(t1, X), self.diffusion(t1, X), dt, x2grid, yknown[:self.xnum[n+1]])
        EZ, EZW = self.GaussQuad_sinc(X, self.drift(t1, X), self.diffusion(t1, X), dt, x2grid, zknown[:self.xnum[n+1]])
        EF, EFW = self.GaussQuad_sinc(X, self.drift(t1, X), self.diffusion(t1, X), dt, x2grid, f2grid)
        
        Z = EFW-EZ + 2.0*EYW/dt

        for i in range(len(X)):
            func = lambda y: EY[i] + 0.5*self.generator(t1, X[i], y, Z[i]) * dt + 0.5*dt*EF[i] - y
            root = sp.optimize.root(func, EY[i])
            Y[i] = root.x[0]
            Y[i] = np.maximum(Y[i], 0)
            L[i] = 1.0/self.dt * np.maximum(- Y[i], 0)

        self.ynum_lb[n] = np.min(Y)
        self.ynum_ub[n] = np.max(Y)

        return Y, Z, L

    def pre_CN_step(self):

        dt = self.dt
        dh = self.dh
        t = self.tgrid[self.N-1]

        xgrid = np.linspace(self.lb[self.N], self.ub[self.N], self.xnum[self.N])
        ygrid = self.terminal(xgrid)
        zgrid = self.ztrue(self.T, xgrid)

        yknown, zknown = self.CN_step(self.N-1, ygrid, zgrid)

        return yknown, zknown

    def CN_solver(self):

        Y, Z = self.pre_CN_step()

        for i in range(self.N-2, -1, -1):
            Y, Z, L = self.CN_step(i, Y, Z)
        
        self.X = self.init
        self.Y = Y
        self.Z = Z
        self.L = L

        return 




    # interpolation methods
    def interp_sinc(self, x, y, xinterp):
        """
        Interpolates the values yinterp at xinterp by values (x, y) using sinc interpolation. x must be uniformly spaced.
        The interpolation is done at the points xinterp.
        """
        n = len(x)
        yinterp = np.zeros_like(xinterp)
        for i in range(n):
            yinterp += y[i] * np.sinc((xinterp - x[i]) / (x[1] - x[0]))
        return yinterp
    
    # compute the expectation
    def GaussQuad_sinc(self, x, mu, sigma, dt, xsinc, ysinc):
        """
        Computes the condition expectation E[u(x + y)], where y is a Gaussian random variable with mean mu and standard deviation sigma.
        x is a fixed vector of points where the expectation is computed. mu and sigma are also vectors of the same length as x.
        The function uses sinc interpolation to compute the expectation at the points xsinc, which are uniformly spaced.
        ysinc is the corresponding values of u(x + kh) at the points xsinc.
        """
        n = len(x)
        m = len(xsinc)
        quad_grid = np.zeros((m, n))
        EY = np.zeros_like(x)
        EYW = np.zeros_like(x)
        x_grid = np.ones((m,1)) * x.reshape((1,n))
        mu_grid = np.ones((m,1)) * mu.reshape((1,n)) * dt
        sigma_grid = np.ones((m,1)) * sigma.reshape((1,n)) * np.sqrt(dt)
        xsinc_grid = xsinc.reshape((m,1)) * np.ones((1,n))
        ysinc_grid = ysinc.reshape((m,1)) * np.ones((1,n))
        dist = xsinc_grid - x_grid
        EY_grid = ysinc_grid * 1.0/np.sqrt(2*np.pi)/sigma_grid*np.exp(-0.5 * (dist - mu_grid)**2 / sigma_grid**2)
        EYW_grid = ysinc_grid * 1.0/np.sqrt(2*np.pi)/sigma_grid*np.exp(-0.5 * (dist - mu_grid)**2 / sigma_grid**2) * (dist - mu_grid) / sigma_grid *np.sqrt(dt)
        EY = np.sum(EY_grid, axis=0) * (xsinc[1] - xsinc[0])
        EYW = np.sum(EYW_grid, axis=0) * (xsinc[1] - xsinc[0])

        return EY, EYW
    
    
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

    def convergence_rate(self, dt, error):
        

        n = len(dt)
        dt_col = dt.reshape(n, 1)
        err_col = error.reshape(n, 1)

        Xmatrix = np.hstack((np.ones((n,1)), np.log(dt_col)))

        beta = np.matmul(np.linalg.inv(np.matmul(Xmatrix.T, Xmatrix)), (np.matmul(Xmatrix.T, np.log(err_col))))
        
        rate = beta[1]            

        return rate



###########################################################################################################################################
# example1
###########################################################################################################################################


    
# def example1_drift(t, x, y, z):
#     return np.zeros(x.shape)
# def example1_diffusion(t, x, y, z):
#     return np.ones(x.shape)
# def example1_generator(t, x, y, z):
#     return y - z - 0.5*np.sin(x)
# def example1_terminal(t, x):
#     return np.sin(t+x)
# def example1_ytrue(t, x):
#     return np.sin(t + x)
# def example1_ztrue(t, x):
#     return np.cos(t + x)



# example1 = FBSDE(
#     t0=0.0,
#     T=1.0,
#     mu=example1_drift,
#     sigma=example1_diffusion,
#     f=example1_generator,
#     g=example1_terminal,
#     init=np.linspace(-2, 2, 101),
#     ytrue=example1_ytrue,
#     ztrue=example1_ztrue
# )




###########################################################################################################################################
# example2
###########################################################################################################################################

# def example2_drift(t, x, y, z):
#     return np.zeros(x.shape)
# def example2_diffusion(t, x, y, z):
#     return np.ones(x.shape)
# def example2_generator(t, x, y, z):
#     return z*y - 0.75*z
# def example2_terminal(t, x):
#     return np.exp(0.25*t + x)/(1.0+np.exp(0.25*t+x))
# def example2_ytrue(t, x):
#     return np.exp(0.25*t + x)/(1.0+np.exp(0.25*t+x))
# def example2_ztrue(t, x):
#     return np.exp(0.25*t + x)/(1.0+np.exp(0.25*t+x))**2


# example2 = FBSDE(
#     t0=0.0,
#     T=1.0,
#     mu=example2_drift,
#     sigma=example2_diffusion,
#     f=example2_generator,
#     g=example2_terminal,
#     init=np.linspace(-2, 2, 101),
#     ytrue=example2_ytrue,
#     ztrue=example2_ztrue
# )


# for N in [10, 20, 40, 80, 160]:

#     example2.N = N  # number of time steps
#     example2.dh = 0.05  # grid size for x
#     example2.dt = (example2.T - example2.t0) / example2.N  # time step size
#     example2.generate_grid(10)
#     example2.CN_solver()
#     example2.print_error()

# example2.plot_solution()


###########################################################################################################################################
# example3 Allen Cahn from Ju2021
###########################################################################################################################################

eps = 0.05
s = 3/(np.sqrt(2.0)*eps)
T = np.sqrt(2.0)*eps/4.0
print("T = ", T)

def example3_drift(t, x):
    return np.zeros(x.shape)
def example3_diffusion(t, x):
    return np.ones(x.shape) * np.sqrt(2.0)
def example3_generator(t, x, y, z):
    return (y - y**3)/(eps**2)
def example3_terminal(t, x):
    return 0.5*(1.0-np.tanh((x + s*t - s*T)/(2*np.sqrt(2)*eps)))
def example3_ytrue(t, x):
    return 0.5*(1.0-np.tanh((x + s*t - s*T)/(2*np.sqrt(2)*eps)))
def example3_ztrue(t, x):
    return 1.0/np.sqrt(2.0)*(- 1.0/(np.cosh((x + s*t - s*T)/(2*np.sqrt(2)*eps))**2*2*np.sqrt(2.0)*eps))
def example3_ylb(t, x):
    return 0.0  # lower bound for y
def example3_yub(t, x):
    return 1.0  # upper bound for y


example3 = RFBSDE_dcpl(
    t0=0.0,
    T=T,
    mu=example3_drift,
    sigma=example3_diffusion,
    f=example3_generator,
    g=example3_terminal,
    init=np.linspace(-0.5, 2.5, 301),
    ytrue=example3_ytrue,
    ztrue=example3_ztrue,
    ylb=example3_ylb,
    yub=example3_yub
)

Nvec = [80]
dt_vec = np.zeros(len(Nvec))
Yerr = np.zeros(len(Nvec))
Zerr = np.zeros(len(Nvec))

for i in range(len(Nvec)):

    example3.N = Nvec[i]  # number of time steps
    example3.dt = (example3.T - example3.t0) / example3.N  # time step size
    example3.dh = 0.5*np.sqrt(example3.dt)  # grid size for x
    example3.generate_grid(20)
    example3.exGenoratorAppCN_solver(2)
    example3.print_error()
    dt_vec[i] = example3.dt
    Yerr[i] = example3.Yerr
    Zerr[i] = example3.Zerr


Ycr = example3.convergence_rate(dt_vec, Yerr)
Zcr = example3.convergence_rate(dt_vec, Zerr)

print(example3.ynum_lb, example3.ynum_ub)
print(np.max(example3.Lsurface))

# example3.plot_Lsurface()
# print(f"convergence rate of Y: {Ycr}, \t\t convergence rate of Z: {Zcr}")
# print(f"Y:{np.log(Yerr[1:]/Yerr[:-1])/np.log(dt_vec[1:]/dt_vec[:-1])}")
# print(f"Z:{np.log(Zerr[1:]/Zerr[:-1])/np.log(dt_vec[1:]/dt_vec[:-1])}")

# example3.plot_solution()


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",   # STIX 是 Times 的数学配套字体
    "font.size": 14,
})

fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.45, 1],
                      wspace=0.22, hspace=0.35)

ax_l  = fig.add_subplot(gs[:, 0])   # 左：跨两行的大图
ax_m1 = fig.add_subplot(gs[0, 1])   # 中上
ax_m2 = fig.add_subplot(gs[1, 1])   # 中下
ax_r  = fig.add_subplot(gs[:, 2])   # 右：跨两行的大图


i=0
x = np.linspace(example3.lb[i], example3.ub[i], example3.xnum[i])
ax_l.plot(x, example3.Ysurface[:example3.xnum[i], i], 'r-', label=r'$t=0$')
ax_m1.plot(x, example3.Ysurface[:example3.xnum[i], i], 'r-')
ax_m2.plot(x, example3.Ysurface[:example3.xnum[i], i], 'r-')
i=20
x = np.linspace(example3.lb[i], example3.ub[i], example3.xnum[i])
ax_l.plot(x, example3.Ysurface[:example3.xnum[i], i], 'b-', label=r'$t=\frac{1}{4}T$')
ax_m1.plot(x, example3.Ysurface[:example3.xnum[i], i], 'b-')
ax_m2.plot(x, example3.Ysurface[:example3.xnum[i], i], 'b-')
i=40
x = np.linspace(example3.lb[i], example3.ub[i], example3.xnum[i])
ax_l.plot(x, example3.Ysurface[:example3.xnum[i], i], 'k-', label=r'$t=\frac{1}{2}T$')
ax_m1.plot(x, example3.Ysurface[:example3.xnum[i], i], 'k-')
ax_m2.plot(x, example3.Ysurface[:example3.xnum[i], i], 'k-')
i=60
x = np.linspace(example3.lb[i], example3.ub[i], example3.xnum[i])
ax_l.plot(x, example3.Ysurface[:example3.xnum[i], i], 'g-', label=r'$t=\frac{3}{4}T$')
ax_m1.plot(x, example3.Ysurface[:example3.xnum[i], i], 'g-')
ax_m2.plot(x, example3.Ysurface[:example3.xnum[i], i], 'g-')
ax_l.legend(loc='best')
ax_l.axhline(0.0, color="#7E2F8E", ls="--", lw=1.0)
ax_l.axhline(1.0, color="#7E2F8E", ls="--", lw=1.0)
ax_l.set_xlim(example3.lb[0], example3.ub[0])
ax_l.set_ylim(0.0, 1.0)
ax_l.set_xlabel('x')
ax_l.set_ylabel('Ynum')


ax_m1.axhline(1.0, color="#7E2F8E", ls="--", lw=1.0)
ax_m1.set_xlim(-0.5, -0.45)
ax_m1.set_ylim(0.999995, 1.0000001)
ax_m1.set_xticks([-0.5, -0.45])
ax_m1.set_yticks([1.0])

ax_m2.axhline(0.0, color="#7E2F8E", ls="--", lw=1.0)
ax_m2.set_xlim(1.45, 1.50)
ax_m2.set_ylim(-0.0000001, 0.0000005)
ax_m2.set_xticks([1.45, 1.50])
ax_m2.set_yticks([0.0])

ax_r.yaxis.tick_right()
ax_r.yaxis.set_label_position("right")
ax_r.axhline(0.0, color="#7E2F8E", ls="-", lw=1.0)
ax_r.set_xlim(example3.tgrid[0], example3.tgrid[-1])
ax_r.set_xlabel('t')
ax_r.set_ylabel(r'supremum of $\lambda_t$')
ax_r.set_ylim(-0.000001, 0.000001)

plt.show()

# print(max(example3.Ysurface[:example3.xnum[i], i]), min(example3.Ysurface[:example3.xnum[i], i]))
# print(max(example3.Zsurface[:example3.xnum[i], i]), min(example3.Zsurface[:example3.xnum[i], i]))
# print(np.max(example3.Ysurface), np.min(example3.Ysurface))
# for i in range(example3.Ysurface.shape[1]):
#     print(np.sum(example3.Ysurface[:example3.xnum[i], i]>1.0), example3.xnum[i])

# example3.plot_Ysurface()
# example3.plot_Zsurface()





# """
# 复现 SIFRK(2,2)/(3,2)/(4,2) 的能量演化与最大模演化双子图（含局部放大 inset）。

# 用法：把 "数据区" 换成你自己的 t / energy / sup-norm 数组即可，绘图部分无需改动。
# """


# # ----------------------------------------------------------------------
# # 数据区：用你自己的数据替换这一段
# # ----------------------------------------------------------------------
# SCHEMES = ["SIFRK(2,2)", "SIFRK(3,2)", "SIFRK(4,2)"]
# COLORS = ["#0072BD", "#D95319", "#EDB120"]  # MATLAB 默认前三色

# BOUND = 0.9575  # 右图紫色虚线（理论上界 / 稳态值）


# def demo_energy(t, shift):
#     """示例能量曲线：快速下降后缓慢耗散。"""
#     return -0.185 / (1 + np.exp(-1.5 * (t - 6.5))) - 0.055 * (1 - np.exp(-t / 50)) - shift


# def demo_supnorm(t, shift):
#     """示例最大模曲线：先跌至谷底再单调上升趋于 BOUND。"""
#     decay = 0.90 * np.exp(-3.5 * t) + 0.10 * np.exp(-0.90 * t)
#     rise = (BOUND - 0.0020 - shift) / (1 + np.exp(-2.2 * (t - 4.0)))
#     return decay + rise


# t_e = np.linspace(0, 100, 4001)      # 左图时间轴
# t_s = np.linspace(0, 20, 2001)       # 右图时间轴

# # 不同格式之间的微小差异（仅用于让 inset 里能看出分层）
# offsets = [0.0, 6e-4, 9e-4]

# energy = [demo_energy(t_e, d) for d in offsets]
# supnorm = [demo_supnorm(t_s, d) for d in offsets]

# # ----------------------------------------------------------------------
# # 绘图区
# # ----------------------------------------------------------------------
# plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8})

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))


# def add_inset(ax, t_list, y_list, xlim, rect, pad=0.08, legend=False):
#     """在 ax 上加一个放大子图。rect = [x0, y0, w, h]（axes 坐标 0~1）。"""
#     axi = ax.inset_axes(rect)
#     lo, hi = np.inf, -np.inf
#     for t, y, c, lab in zip(t_list, y_list, COLORS, SCHEMES):
#         m = (t >= xlim[0]) & (t <= xlim[1])
#         axi.plot(t[m], y[m], color=c, lw=1.0, marker=".", ms=2.5,
#                  markevery=3, label=lab)
#         lo, hi = min(lo, y[m].min()), max(hi, y[m].max())
#     span = hi - lo
#     axi.set_xlim(*xlim)
#     axi.set_ylim(lo - pad * span, hi + pad * span)
#     axi.tick_params(labelsize=8)
#     if legend:
#         axi.legend(fontsize=7, loc="center right")
#     return axi


# # ---- 左图：Energy ----
# for y, c, lab in zip(energy, COLORS, SCHEMES):
#     ax1.plot(t_e, y, color=c, lw=1.4, label=lab)
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Energy")
# ax1.set_xlim(0, 100)
# ax1.set_ylim(-0.25, 0.05)
# ax1.legend(loc="upper right", fontsize=9, markerscale=1)
# add_inset(ax1, [t_e] * 3, energy, xlim=(8.9, 9.1),
#           rect=[0.28, 0.42, 0.42, 0.42])

# # ---- 右图：Supremum norm ----
# for y, c, lab in zip(supnorm, COLORS, SCHEMES):
#     ax2.plot(t_s, y, color=c, lw=1.4, label=lab)
# ax2.axhline(BOUND, color="#7E2F8E", ls="--", lw=1.2)
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Supremum norm")
# ax2.set_xlim(0, 20)
# ax2.set_ylim(0, 1)
# ax2.set_xticks(np.arange(0, 21, 2))
# ax2.legend(loc="center right", fontsize=9)
# axi2 = add_inset(ax2, [t_s] * 3, supnorm, xlim=(6.5, 7.5),
#                  rect=[0.22, 0.30, 0.42, 0.42], pad=0.35)
# axi2.axhline(BOUND, color="#7E2F8E", ls="--", lw=1.0)

# fig.tight_layout()
# fig.savefig("sifrk_energy_supnorm.png", dpi=300, bbox_inches="tight")
# plt.show()