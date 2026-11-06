import math
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from enum import Enum

class OptionType(Enum):
    CALL = 1
    PUT = 2

class AmericanOptionPricer:
    def __init__(self, S0, K, T, r, sigma, q, n_steps, option_type):
        """
        American Option Pricer using Binomial Tree Model
        
        Parameters:
        S0: Current underlying asset price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free interest rate
        sigma: Volatility
        q: Continuous dividend yield
        n_steps: Number of steps in binomial tree
        option_type: Option type (OptionType.CALL or OptionType.PUT)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q  # Dividend yield
        self.n_steps = n_steps
        self.option_type = option_type
        
        # Calculate binomial tree parameters
        self.dt = T / n_steps
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (math.exp((r - q) * self.dt) - self.d) / (self.u - self.d)
        
        # Check for arbitrage opportunity
        if self.p < 0 or self.p > 1:
            print("Warning: Risk-neutral probability not in [0,1] range, potential arbitrage opportunity")
    
    def calculate_call_option_price(self):
        """Calculate American call option price and early exercise boundary"""

        time_vec = np.linspace(0, self.T, self.n_steps+1)
        early_time = []
        early_price = []
        stock_price = np.zeros(self.n_steps+1)
        option_price_known = np.zeros(self.n_steps+1)
        d_num = np.linspace(0, self.n_steps, self.n_steps + 1)
        u_num = self.n_steps - d_num
        stock_price = self.S0 * (self.d ** d_num) * (self.u ** u_num) # [0: self.n_steps] decreasing
        option_price_known = np.maximum(0, stock_price - self.K) # [0: self.n_steps]
        early_flag = False

        for i in range(self.n_steps-1, -1, -1):
            
            d_num = np.linspace(0, i, i+1)
            u_num = i - d_num
            stock_price = self.S0 * (self.d ** d_num) * (self.u ** u_num) # [0: i] decrasing
            option_price_solve = np.zeros(i+1) # [0: i]
            for j in range(i + 1):
                # Calculate holding value (present value of expected future payoff)
                hold_value = math.exp(-self.r * self.dt) * (self.p * option_price_known[j] + (1 - self.p) * option_price_known[j+1])
                # Calculate immediate exercise value
                exercise_value = max(0, stock_price[j] - self.K)

                if exercise_value > hold_value:
                    option_price_solve[j] = exercise_value
                    early_flag = True
                    early_time_tmp = time_vec[i]
                    early_price_tmp = stock_price[j]
                else:
                    option_price_solve[j] = hold_value
                    
            
            option_price_known = option_price_solve
            print("current step:", i, "early_flag", early_flag)
            if early_flag:
                early_time.append(early_time_tmp)
                early_price.append(early_price_tmp)
            early_flag = False

        call_price = option_price_known[0]

        return call_price, early_time, early_price
    

    def calculate_put_option_price(self):
        """Calculate American put option price and early exercise boundary"""

        time_vec = np.linspace(0, self.T, self.n_steps+1)
        early_time = []
        early_price = []
        stock_price = np.zeros(self.n_steps+1)
        option_price_known = np.zeros(self.n_steps+1)
        u_num = np.linspace(0, self.n_steps, self.n_steps + 1)
        d_num = self.n_steps - u_num
        stock_price = self.S0 * (self.d ** d_num) * (self.u ** u_num) # [0: self.n_steps] increasing
        option_price_known = np.maximum(0, self.K - stock_price) # [0: self.n_steps]
        early_flag = False

        for i in range(self.n_steps-1, -1, -1):
            u_num = np.linspace(0, i, i+1)
            d_num = i - u_num
            stock_price = self.S0 * (self.d ** d_num) * (self.u ** u_num) # [0: i] increasing
            option_price_solve = np.zeros(i+1) # [0: i]
            for j in range(i + 1):
                # Calculate holding value (present value of expected future payoff)
                hold_value = math.exp(-self.r * self.dt) * (self.p * option_price_known[j+1] + (1 - self.p) * option_price_known[j])
                # Calculate immediate exercise value
                exercise_value = max(0, self.K - stock_price[j])

                if hold_value >= exercise_value:
                    option_price_solve[j] = hold_value
                else:
                    option_price_solve[j] = exercise_value
                    early_flag = True
                    early_time_tmp = time_vec[i]
                    early_price_tmp = stock_price[j]
            
            option_price_known = option_price_solve
            print("current step:", i, "early_flag", early_flag)
            if early_flag:
                early_time.append(early_time_tmp)
                early_price.append(early_price_tmp)
            early_flag = False

        put_price = option_price_known[0]

        return put_price, early_time, early_price


    
    
    
    def plot_early_exercise_boundary(self, times, boundary_prices):
        """Plot early exercise boundary"""
        
        plt.figure(figsize=(10, 6))
        
        # Plot boundary points
        plt.plot(times, boundary_prices, 'r-', linewidth=1, label='Early Exercise Boundary')
        
        # Add horizontal line for strike price
        plt.axhline(y=self.K, color='gray', linestyle='--', alpha=0.7, label=f'Strike Price K={self.K}')
        
        # Set chart title and labels
        option_type_str = "Call" if self.option_type == OptionType.CALL else "Put"
        plt.title(f'American {option_type_str} Option Early Exercise Boundary\n'
                 f'(S0={self.S0}, K={self.K}, T={self.T} years, r={self.r:.1%}, σ={self.sigma:.1%}, q={self.q:.1%})')
        plt.xlabel('Time (years)')
        plt.ylabel('Stock Price')
        
        # Set Y-axis range based on option type
        if self.option_type == OptionType.PUT:
            # For put options, boundary is below strike price
            plt.ylim(self.K * 0.7, self.K * 1.05)
            plt.fill_between(times, boundary_prices, self.K * 0.7, alpha=0.2, color='red', 
                           label='Early Exercise Region')
            plt.text(times[-1] if times else self.T/2, self.K * 0.75, 
                   'Early Exercise Region', fontsize=10, color='red')
        else:  # CALL
            # For call options, boundary is above strike price
            plt.ylim(self.K * 0.95, self.K * 1.3)
            plt.fill_between(times, boundary_prices, self.K * 1.3, alpha=0.2, color='red',
                           label='Early Exercise Region')
            plt.text(times[-1] if times else self.T/2, self.K * 1.25, 
                   'Early Exercise Region', fontsize=10, color='red')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        
        return 

def main():
    # Example parameters - put options are more likely to be exercised early
    S0 = 100      # Current stock price
    K = 100       # Strike price
    T = 1.0       # Time to expiration (years)
    r = 0.05      # Risk-free interest rate
    sigma = 0.2   # Volatility
    q = 0.045      # Continuous dividend yield
    n_steps = 1000000 # Number of steps in binomial tree
    
    print("American Option Pricing - Binomial Tree Model")
    print("=" * 50)
    print(f"Underlying asset price: {S0}")
    print(f"Strike price: {K}")
    print(f"Time to expiration: {T} years")
    print(f"Risk-free interest rate: {r*100}%")
    print(f"Volatility: {sigma*100}%")
    print(f"Dividend yield: {q*100}%")
    print(f"Number of tree steps: {n_steps}")
    print("=" * 50)
    
    # Calculate American put option price and plot boundary
    print("\n1. American Put Option Analysis:")
    start = datetime.now()
    put_pricer = AmericanOptionPricer(S0, K, T, r, sigma, q, n_steps, OptionType.PUT)
    put_price, early_time, early_price = put_pricer.calculate_put_option_price()
    end = datetime.now()
    put_rt = end - start
    print(f"Computation Time for Put Option: {put_rt.total_seconds():.2f} seconds")
    print(f"American put option price: {put_price:.6f}")
    put_pricer.plot_early_exercise_boundary(early_time, early_price)
    

    with open('put_boundary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(early_time)
        writer.writerow(early_price)
    
    # Calculate American call option price and plot boundary
    # print("\n2. American Call Option Analysis:")
    # start = datetime.now()
    # call_pricer = AmericanOptionPricer(S0, K, T, r, sigma, q, n_steps, OptionType.CALL)
    # call_price, early_time, early_price = call_pricer.calculate_call_option_price()
    # end = datetime.now()
    # call_rt = end - start
    # print(f"Computation Time for Call Option: {call_rt.total_seconds():.2f} seconds")
    # print(f"American call option price: {call_price:.6f}")
    # call_pricer.plot_early_exercise_boundary(early_time, early_price)


    
    # with open('call_boundary.csv', 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(early_time)
    #     writer.writerow(early_price)



    with open('BiTreeResult.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["American Option Pricing - Binomial Tree Model"])
        writer.writerow(["=" * 50])
        writer.writerow(["S0=", S0])
        writer.writerow(["K=", K])
        writer.writerow(["T=", T])
        writer.writerow(["r=", r])
        writer.writerow(["sigma=", sigma])
        writer.writerow(["q=", q])
        writer.writerow(["Number of tree steps=", n_steps])
        writer.writerow(["-" * 50])
        writer.writerow(['Put Price', put_price])
        writer.writerow(['Computation Time (seconds)', f"{put_rt.total_seconds():.2f}"])
        # writer.writerow(['Call Price', call_price])
        # writer.writerow(['Computation Time (seconds)', f"{call_rt.total_seconds():.2f}"])
        writer.writerow(["=" * 50])




if __name__ == "__main__":


    # main()


    put_data = np.genfromtxt('put_boundary30.csv', delimiter=',', skip_header=0)
    put_times = put_data[0]
    put_boundary = put_data[1]


    K = 100
    T = 1.0
    plt.figure(figsize=(16, 8))
    plt.subplot(1,2,1)
    plt.plot(put_times, put_boundary, 'r-', linewidth=1, label='Early Exercise Boundary for American Put Option')
    plt.fill_between(put_times, put_boundary, K * 0.7, alpha=0.2, color='red', label='Early Exercise Region')
    plt.text(T/2, K * 0.75, 'Early Exercise Region', fontsize=10, color='red')
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Strike Price K=100')
    plt.title('Early Exercise Boundary For American Put Option')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid(True, alpha=0.3)
    plt.legend()

    call_data = np.genfromtxt('call_boundary30.csv', delimiter=',', skip_header=0)

    call_times = call_data[0]
    call_boundary = call_data[1]

    plt.subplot(1,2,2)
    plt.plot(call_times, call_boundary, 'r-', linewidth=1, label='Early Exercise Boundary for American Call Option')
    plt.fill_between(call_times, call_boundary, K * 1.48, alpha=0.2, color='red', label='Early Exercise Region')
    plt.text(T/2, K * 1.4, 'Early Exercise Region', fontsize=10, color='red')
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Strike Price K=100')
    plt.title('Early Exercise Boundary for American Call Option')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.show()