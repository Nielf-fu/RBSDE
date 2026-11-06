import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class RFBSDE_cpl:
    def __init__(self, t0, T, mu, sigma, f, g, ytrue, ztrue):
        self.t0 = t0
        self.T = T
        self.mu = mu
        self.sigma = sigma
        self.f = f
        self.g = g
        self.ytrue = ytrue
        self.ztrue = ztrue

        # paramters for the FBSDE solver
        self.N = 100  # number of time steps
        self.M = 100  # number of space points
        self.lb = -np.pi  # lower bound for x
        self.ub = np.pi  # upper bound for x
        self.eps = 1e-8  # tolerance for convergence
        
        self.X = None  # solution for x which coincides with the self.init
        self.Y = None  # solution for y at t=t0
        self.Z = None  # solution for z at t=t0
        self.L = None
        self.Yerr = None  # error for y at t=t0
        self.Zerr = None  # error for z at t=t0
        self.Ysurface = None  # solution surface for y 
        self.Zsurface = None  # solution surface for z
        self.Lsurface = None  # solution surface for L
        


    def drift(self, t, x, y):
        # Placeholder for the drift function f(t, x)
        return self.mu(t, x, y)
    
    def diffusion(self, t, x, y):
        # Placeholder for the diffusion function sigma(t, x)
        return self.sigma(t, x, y)

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
        plt.figure(figsize=(5, 5))
        plt.imshow(self.Ysurface, aspect='auto', origin='lower', extent=(self.tgrid[0], self.tgrid[-1], self.lb, self.ub), cmap='viridis')
        plt.colorbar(label='Y')
        plt.xlabel('Time')
        plt.ylabel('X')
        plt.title('numerical solution of $Y_t$')
        plt.show()
    # tbd    
    def plot_Zsurface(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.Zsurface, aspect='auto', origin='lower', extent=(self.tgrid[0], self.tgrid[-1], self.lb, self.ub), cmap='viridis')
        plt.colorbar(label='Z')
        plt.xlabel('Time')
        plt.ylabel('X')
        plt.title('numerical solution of $Z_t$')
        plt.show()



    def print_error(self):
        # Compute the error between the numerical solution and the true solution
        self.Yerr = np.max(np.abs(self.Y - self.ytrue(self.t0, self.X)))
        self.Zerr = np.max(np.abs(self.Z - self.ztrue(self.t0, self.X)))
        print(f"dt = {self.dt:10.8f}, \t dh={self.dh:10.8f}, \t Yerror= {self.Yerr:10.5e}, \t Zerror = {self.Zerr:10.5e}")
        return
    

    def generate_grid(self):
        
        self.dt = (self.T - self.t0) / self.N  # time step size
        self.dh = (self.ub - self.lb) / self.M  # grid size for x
        self.tgrid = np.linspace(self.t0, self.T, self.N+1)
        self.xgrid = np.linspace(self.lb, self.ub - self.dh, self.M)
        self.X = self.xgrid

        self.Ysurface = np.zeros((self.M, self.N+1))
        self.Zsurface = np.zeros((self.M, self.N+1))
        self.Lsurface = np.zeros((self.M, self.N+1))
        
        return

    
    # implicit Euler method
    
    def pre_Euler_step(self):
        dt = self.dt
        dh = self.dh
        xgrid = self.xgrid
        Z = np.zeros(len(xgrid))
        Y = np.zeros(len(xgrid))

        xsinc = xgrid
        ysinc = self.terminal(xsinc)
        Y0 = self.terminal(xgrid)
        EY, EYW = self.GaussQuad_sinc(self.N-1, 1, Y0, ysinc)
        Z = EYW/dt

        # explicit scheme for Y
        Y = EY + self.generator(self.tgrid[-2], xgrid, EY, Z) * dt
        Y = np.maximum(Y, -1)
        L = 1.0/self.dt * np.maximum(-1-Y, 0)
        # for i in range(len(xgrid)):
        #     func = lambda y: y - EY[i] - self.generator(self.tgrid[-2], xgrid[i], y, Z[i])*dt
        #     root = sp.optimize.root(func, EY[i])
        #     Y[i] = root.x[0]

        while (np.max(np.abs(Y-Y0)) > self.eps):
            Y0 = Y
            EY, EYW = self.GaussQuad_sinc(self.N-1, 1, Y0, ysinc)
            Z = EYW/dt

            # explicit scheme for Y
            Y = EY + self.generator(self.tgrid[-2], xgrid, EY, Z) * dt
            Y = np.maximum(Y, -1)
            L = 1.0/self.dt * np.maximum(-1-Y, 0)


            # for i in range(len(xgrid)):
            #     func = lambda y: y - EY[i] - self.generator(self.tgrid[-2], xgrid[i], y, Z[i])*dt
            #     root = sp.optimize.root(func, EY[i])
            #     Y[i] = root.x[0]

        return Y, Z, L
    
    def Euler_step(self, n):
        dt = self.dt
        xgrid = self.xgrid
        Z = np.zeros_like(xgrid)
        Y = np.zeros_like(xgrid)
        L = np.zeros_like(xgrid)
        Y0 = self.Ysurface[:, n+1]
        ysinc = self.Ysurface[:, n+1]

        EY, EYW = self.GaussQuad_sinc(n, 1, Y0, ysinc)
        Z = EYW / dt

        # implicit scheme for Y
        # for i in range(len(xgrid)):
        #     func = lambda y: y - EY[i] - self.generator(self.tgrid[n], xgrid[i], y, Z[i]) * dt
        #     root = sp.optimize.root(func, EY[i])
        #     Y[i] = root.x[0] 

        # explicit scheme for Y
        Y = EY + self.generator(self.tgrid[n], xgrid, EY, Z) * dt
        Y = np.maximum(Y, -1)
        L = 1.0/self.dt * np.maximum(-1-Y, 0)

        while (np.max(np.abs(Y-Y0)) > 1e-6):
            Y0 = Y
            EY, EYW = self.GaussQuad_sinc(n, 1, Y0, ysinc)
            Z = EYW / dt
            # implicit scheme for Y
            # for i in range(len(xgrid)):
            #     func = lambda y: y - EY[i] - self.generator(self.tgrid[n], xgrid[i], y, Z[i]) * dt
            #     root = sp.optimize.root(func, EY[i])
            #     Y[i] = root.x[0]

            # explicit scheme for Y
            Y = EY + self.generator(self.tgrid[n], xgrid, EY, Z) * dt
            Y = np.maximum(Y, -1)
            L = 1.0/self.dt * np.maximum(-1-Y, 0)

        return Y, Z, L
    
    def Euler_solver(self):

        self.Ysurface[:, -1] = self.terminal(self.xgrid)
        self.Zsurface[:, -1] = self.ztrue(self.tgrid[-1], self.xgrid)
        
        Y, Z, L = self.pre_Euler_step()
        self.Ysurface[:, -2] = Y
        self.Zsurface[:, -2] = Z

        for i in range(self.N-2, -1, -1):
            Y, Z, L = self.Euler_step(i)
            self.Ysurface[:, i] = Y
            self.Zsurface[:, i] = Z
            self.Lsurface[:, i] = L
        
        self.Y = Y
        self.Z = Z
        self.L = L
        
        return

    
    

    def exGenoratorApp_step(self, n, kstep):
        dt = self.dt
        t = self.tgrid[n]

        yknown = self.Ysurface[:, n+1:n+kstep+1].reshape((self.Ysurface.shape[0], kstep))
        
        alpha = self.get_alpha(kstep)
        beta = self.get_beta(kstep)

        xgrid = self.xgrid
        Z = np.zeros_like(xgrid)
        Y = np.zeros_like(xgrid)
        Y0 = yknown[:, 0]
        EY = np.zeros((len(xgrid), kstep))
        EYW = np.zeros((len(xgrid), kstep))

        for i in range(1, kstep+1):
            EY[:,i-1], EYW[:,i-1] = self.GaussQuad_sinc(n, i, Y0, yknown[:, i-1])
    
        Z = np.matmul(EYW, alpha[1:]).reshape(len(xgrid)) / dt

        # implicit scheme for Y
        # for i in range(len(xgrid)):
        #     func = lambda y: alpha[0]*y + np.dot(alpha[1:], EY[i,:]) + self.generator(t, xgrid[i], y, Z[i]) * dt
        #     root = sp.optimize.root(func, np.mean(EY[i, :]))
        #     Y[i] = root.x[0]

        # explicit scheme for Y
        Y = np.matmul(EY, beta).reshape(len(xgrid))
        Y = -1.0/alpha[0]*(np.matmul(EY, alpha[1:]).reshape(len(xgrid)) + self.generator(t, xgrid, Y, Z) * dt)
        Y = np.maximum(Y, -1)
        L = 1.0/self.dt * np.maximum(-1-Y, 0)


        while (np.max(np.abs(Y-Y0)) > self.eps):
            Y0 = Y
            for i in range(1, kstep+1):
                EY[:,i-1], EYW[:,i-1] = self.GaussQuad_sinc(n, i, Y0, yknown[:, i-1])
            Z = np.matmul(EYW, alpha[1:]).reshape(len(xgrid)) / dt

            # implicit scheme for Y
            # for i in range(len(xgrid)):
            #     func = lambda y: alpha[0]*y + np.dot(alpha[1:], EY[i,:]) + self.generator(t, xgrid[i], y, Z[i]) * dt
            #     root = sp.optimize.root(func, np.mean(EY[i, :]))
            #     Y[i] = root.x[0]
            
            # explicit scheme for Y
            Y = np.matmul(EY, beta).reshape(len(xgrid))
            Y = -1.0/alpha[0]*(np.matmul(EY, alpha[1:]).reshape(len(xgrid)) + self.generator(t, xgrid, Y, Z) * dt)
            Y = np.maximum(Y, -1)
            L = 1.0/self.dt * np.maximum(-1-Y, 0)

        return Y, Z, L

    
    def pre_exGenoratorApp_step(self, kstep):



        Y, Z, L = self.pre_Euler_step()
        self.Ysurface[:, -2] = Y
        self.Zsurface[:, -2] = Z
        self.Lsurface[:, -2] = L

 

        for i in range(1, kstep):
            n = self.N - 1 - i
            
            Y, Z, L = self.exGenoratorApp_step(n, i)

            self.Ysurface[:, n] = Y
            self.Zsurface[:, n] = Z
            self.Lsurface[:, n] = L

        return



    def exGenoratorApp_solver(self, kstep):

        self.Ysurface[:, -1] = self.terminal(self.xgrid)
        self.Zsurface[:, -1] = self.ztrue(self.tgrid[-1], self.xgrid)
        
        self.pre_exGenoratorApp_step(kstep)

        for i in range(self.N-kstep-1, -1, -1):
            Y, Z, L = self.exGenoratorApp_step(i, kstep)
            self.Ysurface[:, i] = Y
            self.Zsurface[:, i] = Z
            self.Lsurface[:, i] = L
        
        self.Y = Y
        self.Z = Z
        self.L = L
        
        return

    

    def shift(self, x):

        x_shifted = (x - self.lb) % (self.ub - self.lb) + self.lb

        return x_shifted


    # compute the expectation
    def GaussQuad_sinc(self, n, k, y, ysinc):
        """
        Computes the conditional expectation E[u(x + w)], where w is a Brownian motion.
        x is a fixed vector of points where the expectation is computed. 
        The function uses sinc interpolation to compute the expectation at the points xsinc, which are uniformly spaced.
        ysinc is the corresponding values of u(x + kh) at the points xsinc.
        k: number of steps for generator approximation method (1, 2, 3, 4, 5, or 6)
        ysinc: values of u(t_{n+k}, x + kh) at the points xsinc, shape (m,)
        """

        x = self.xgrid
        x_center = x + self.drift(self.tgrid[n], x, y) * k * self.dt

        x_matrix = x.reshape(-1, 1) * np.ones_like(x_center).reshape((1, -1))
        x_cmatrix = np.ones_like(x).reshape(-1, 1) * x_center.reshape(1, -1)

        dist = self.shift(x_matrix - x_cmatrix)

        ysinc_matrix = ysinc.reshape(-1, 1) * np.ones_like(x_center).reshape((1, -1))
        sigma = np.ones_like(x).reshape(-1, 1) * self.diffusion(self.tgrid[n], x, y).reshape(1, -1)

        temp = ysinc_matrix * 1.0/np.sqrt(2*np.pi*k*self.dt)/np.abs(sigma)*np.exp(-0.5/(k*self.dt) * (dist**2/sigma**2))

        EY = np.sum(temp, axis=0) * self.dh
        EYW = np.sum(temp * dist/sigma, axis=0) * self.dh


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
# coupled example
# d X_t &= Y_t dt + \left(\frac{1}{2}Y_t + 1\right) dW_t,\\
#-d Y_t &= \left(\sin(t+X_t)(1 + Y_t)+\frac{1}{2}\cos^3(t+X_t)\left(1+\frac{1}{2}Y_t\right)^2 + \frac{1}{2}Y_tZ_t^2\right)dt  - Z_t dW_t + dK_t,\\
# Y_t &\geq 0, \quad 0 \leq t \leq T, \quad K_0 = 0, \quad \int_0^T Y_t dK_t = 0.
###########################################################################################################################################

T = 1.0

def coupled_drift(t, x, y):
    return y
def coupled_diffusion(t, x, y):
    return 0.5*y + 1
def coupled_generator(t, x, y, z):
    return np.sin(t+x)*(1.0+y) + 0.5*np.cos(t+x)**3*(1+0.5*y)**2 + 0.5*y*z**2
def coupled_terminal(t, x):
    return np.cos(t + x)
def coupled_ytrue(t, x):
    return np.cos(t + x)
def coupled_ztrue(t, x):
    return -np.sin(t+x)*(1 + 0.5*np.cos(t+x))


coupled_ex = RFBSDE_cpl(
    t0=0.0,
    T=T,
    mu=coupled_drift,
    sigma=coupled_diffusion,
    f=coupled_generator,
    g=coupled_terminal,
    ytrue=coupled_ytrue,
    ztrue=coupled_ztrue
)

# Nvec = [10, 20, 40, 80, 160, 320]
Nvec = [80]
dt_vec = np.zeros(len(Nvec))
Yerr = np.zeros(len(Nvec))
Zerr = np.zeros(len(Nvec))

for i in range(len(Nvec)):

    coupled_ex.N = Nvec[i]  # number of time steps
    coupled_ex.M = 200
    coupled_ex.generate_grid()
    # coupled_ex.Euler_solver()
    coupled_ex.exGenoratorApp_solver(kstep=2)
    coupled_ex.print_error()
    # coupled_ex.plot_solution()
    dt_vec[i] = coupled_ex.dt
    Yerr[i] = coupled_ex.Yerr
    Zerr[i] = coupled_ex.Zerr


# Ycr = coupled_ex.convergence_rate(dt_vec, Yerr)
# Zcr = coupled_ex.convergence_rate(dt_vec, Zerr)

# print(f"convergence rate of Y: {Ycr}, \t\t convergence rate of Z: {Zcr}")
# print(f"Y:{np.log(Yerr[1:]/Yerr[:-1])/np.log(dt_vec[1:]/dt_vec[:-1])}")
# print(f"Z:{np.log(Zerr[1:]/Zerr[:-1])/np.log(dt_vec[1:]/dt_vec[:-1])}")

# example3.plot_solution()


L_inf = np.max(coupled_ex.Lsurface, axis=0)

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

x = coupled_ex.X

i=0
ax_l.plot(x, coupled_ex.Ysurface[:, i], 'r-', label=r'$t=0$')
ax_m1.plot(x, coupled_ex.Ysurface[:, i], 'r-')
ax_m2.plot(x, coupled_ex.Ysurface[:, i], 'r-')
i=20
ax_l.plot(x, coupled_ex.Ysurface[:, i], 'b-', label=r'$t=\frac{1}{4}T$')
ax_m1.plot(x, coupled_ex.Ysurface[:, i], 'b-')
ax_m2.plot(x, coupled_ex.Ysurface[:, i], 'b-')
i=40
ax_l.plot(x, coupled_ex.Ysurface[:, i], 'k-', label=r'$t=\frac{1}{2}T$')
ax_m1.plot(x, coupled_ex.Ysurface[:, i], 'k-')
ax_m2.plot(x, coupled_ex.Ysurface[:, i], 'k-')
i=60
ax_l.plot(x, coupled_ex.Ysurface[:, i], 'g-', label=r'$t=\frac{3}{4}T$')
ax_m1.plot(x, coupled_ex.Ysurface[:, i], 'g-')
ax_m2.plot(x, coupled_ex.Ysurface[:, i], 'g-')
ax_l.legend(loc='best')
ax_l.axhline(1.0, color="#7E2F8E", ls="--", lw=1.0)
ax_l.axhline(-1.0, color="#7E2F8E", ls="--", lw=1.0)
# ax_l.set_xlim(coupled_ex.lb, coupled_ex.ub)
# ax_l.set_ylim(0.0, 1.0)
ax_l.set_xlabel('x')
ax_l.set_ylabel('Ynum')


ax_m1.axhline(1.0, color="#7E2F8E", ls="--", lw=1.0)
ax_m1.set_xlim(-1.0, 0.2)
ax_m1.set_ylim(0.95, 1.01)
ax_m1.set_xticks([-0.5, 0,0.5])
ax_m1.set_yticks([1.0])

ax_m2.axhline(-1.0, color="#7E2F8E", ls="--", lw=1.0)
ax_m2.set_xlim(2.0, 3.2)
ax_m2.set_ylim(-1.01, -0.95)
ax_m2.set_xticks([2.5, 3.1])
ax_m2.set_yticks([-1.0])

ax_r.yaxis.tick_right()
ax_r.yaxis.set_label_position("right")
# ax_r.axhline(0.0, color="#7E2F8E", ls="-", lw=1.0)
ax_r.plot(coupled_ex.tgrid, L_inf, '-', color="#7E2F8E")
ax_r.set_xlim(coupled_ex.tgrid[0], coupled_ex.tgrid[-1])
ax_r.set_xlabel('t')
ax_r.set_ylabel(r'supremum of $\lambda_t$')
ax_r.set_ylim(-0.000001, 0.000001)

plt.show()












# print(max(coupled_ex.Ysurface[:coupled_ex.xnum[i], i]), min(coupled_ex.Ysurface[:coupled_ex.xnum[i], i]))
# print(max(coupled_ex.Zsurface[:coupled_ex.xnum[i], i]), min(coupled_ex.Zsurface[:coupled_ex.xnum[i], i]))
# # print(np.max(coupled_ex.Ysurface), np.min(coupled_ex.Ysurface))
# # for i in range(coupled_ex.Ysurface.shape[1]):
# #     print(np.sum(coupled_ex.Ysurface[:coupled_ex.xnum[i], i]>1.0), coupled_ex.xnum[i])

# coupled_ex.plot_Ysurface()
# coupled_ex.plot_Zsurface()
