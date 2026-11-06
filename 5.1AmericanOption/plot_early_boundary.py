import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from datetime import datetime



put_data = np.genfromtxt('put_boundary30.csv', delimiter=',', skip_header=0)
put_times = put_data[0]
put_boundary = put_data[1]

put_rbsde_data = []
with open('rbsde_put_boundary3.csv', 'r') as f:
    for line in f:
        row = [float(x) if x.strip() else None for x in line.split(',')] 
        put_rbsde_data.append(row)
put_rbsde_t1 = put_rbsde_data[0]
put_rbsde_boundary1 = put_rbsde_data[1]
put_rbsde_t2 = put_rbsde_data[2]
put_rbsde_boundary2 = put_rbsde_data[3]
put_rbsde_t3 = put_rbsde_data[4]
put_rbsde_boundary3 = put_rbsde_data[5]




K = 100
T = 1.0
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.plot(put_times, put_boundary, 'r-', linewidth=1, label='Binomial Tree (N=300000)')
plt.plot(put_rbsde_t1, put_rbsde_boundary1, 'g--', linewidth=1, label='1-step method (N=200)')
plt.plot(put_rbsde_t2, put_rbsde_boundary2, 'k-o', markersize = 3, linewidth=1, label='2-step method (N=200)')
plt.plot(put_rbsde_t3, put_rbsde_boundary3, 'b-x', markersize = 4, linewidth=1, label='3-step method (N=200)')
plt.fill_between(put_times, put_boundary, K * 0.7, alpha=0.2, color='red', label='Early Exercise Region')
plt.text(T/2, K * 0.75, 'Early Exercise Region', fontsize=12, color='red')
plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Strike Price K=100')
plt.title('Early Exercise Boundary For American Put Option', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Stock Price', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(0.0, 0.7), fontsize=12, loc='center left')

call_data = np.genfromtxt('call_boundary30.csv', delimiter=',', skip_header=0)
call_times = call_data[0]
call_boundary = call_data[1]

call_rbsde_data = []
with open('rbsde_call_boundary3.csv', 'r') as f:
    for line in f:
        row = [float(x) if x.strip() else None for x in line.split(',')] 
        call_rbsde_data.append(row)
call_rbsde_t1 = call_rbsde_data[0]
call_rbsde_boundary1 = call_rbsde_data[1]
call_rbsde_t2 = call_rbsde_data[2]
call_rbsde_boundary2 = call_rbsde_data[3]
call_rbsde_t3 = call_rbsde_data[4]
call_rbsde_boundary3 = call_rbsde_data[5]

plt.subplot(1,2,2)
plt.plot(call_times, call_boundary, 'r-', linewidth=1, label='Binomial Tree (N=300000)')
plt.plot(call_rbsde_t1, call_rbsde_boundary1, 'g--', linewidth=1, label='1-step method (N=200)')
plt.plot(call_rbsde_t2, call_rbsde_boundary2, 'k-o', markersize = 3, linewidth=1, label='2-step method (N=200)')
plt.plot(call_rbsde_t3, call_rbsde_boundary3, 'b-x', markersize = 4, linewidth=1, label='3-step method (N=200)')
plt.fill_between(call_times, call_boundary, K * 1.48, alpha=0.2, color='red', label='Early Exercise Region')
plt.text(T/2, K * 1.4, 'Early Exercise Region', fontsize=12, color='red')
plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Strike Price K=100')
plt.title('Early Exercise Boundary for American Call Option', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Stock Price', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(0.0, 0.3), fontsize=12, loc='center left')

plt.show()