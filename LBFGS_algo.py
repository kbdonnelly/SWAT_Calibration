# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 09:41:41 2025

@author: kbdon
"""

import numpy as np
from scipy.optimize import minimize
import torch

from calib_objective import objective_function
from SWATrun import SWATrun

simulator = SWATrun()

dim = 45

x_init = torch.rand(45)
x_init = (x_init - simulator.LB)/(simulator.UB - simulator.LB)


def g(x):
    """This is a helper function we use to unnormalize and evaluate a point"""
    # print(x)
    return -objective_function(x)

# Define the function to optimize
def f(x, weights):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    g_vals = g(x_tensor)  # Evaluate g(x)
    return torch.sum(weights * g_vals).item()

# multi-start optimization function
def multi_start_optimization(weights, dim, x0=None, num_starts=20):
    best_solution = None
    best_value = float('-inf')

    bounds = np.array([[0, 1]] * dim)  # Scaled bounds to (0,1) space

    if x0 is not None:
        result = minimize(f, x0, args=(weights,), bounds=bounds, method='L-BFGS-B',options = {'maxfev': 500})
        if result.success and -result.fun > best_value:
            best_value = -result.fun
            best_solution = result.x

    for _ in range(num_starts + 1):
        x0 = np.random.uniform(0, 1, size=dim)  # Random initialization in (0,1) space
        result = minimize(f, x0, args=(weights,), bounds=bounds, method='L-BFGS-B',options = {'maxfev': 500})

        if result.success and -result.fun > best_value:
            best_value = -result.fun
            best_solution = result.x

    return best_solution, best_value

weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# run the multi-start optimization to get a hopefully accurate global optimum
best_x, best_f = multi_start_optimization(weights, dim, x0=x_init.numpy(), num_starts=10)
print("Best solution:", best_x)
print("Best function value:", best_f)