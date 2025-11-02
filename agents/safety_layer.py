import torch
import cvxpy as cp
import numpy as np

R_EARTH = 6378e3
H_MIN = 120e3

def cbf_safe_action(state, a_rl):
    r = state[:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)
    h = r_norm - R_EARTH - H_MIN
    if h < 0: h = 0.0
    A = r / r_norm
    b = -np.dot(A, v) - 1.0 * h
    a = cp.Variable(3)
    objective = cp.Minimize(cp.sum_squares(a - a_rl))
    constraints = [A @ a >= b]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    return np.array(a.value, dtype=np.float32)
