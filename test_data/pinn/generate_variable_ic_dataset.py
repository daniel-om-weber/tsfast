#!/usr/bin/env python3
"""Generate variable-IC PINN dataset for spring-damper system.

Extends the minimal generator with varying initial conditions.
Output: test_data/pinn_var_ic/{train,valid,test}/
"""

import numpy as np
import h5py
from scipy.integrate import solve_ivp
from pathlib import Path

# Physical parameters (must match training)
MASS = 1.0
SPRING_CONSTANT = 1.0
DAMPING_COEFFICIENT = 0.1

# Simulation parameters
SAMPLING_RATE = 100.0
DURATION = 5.0
DT = 1.0 / SAMPLING_RATE
T_SPAN = (0.0, DURATION)
T_EVAL = np.arange(0, DURATION, DT)
N_SAMPLES = len(T_EVAL)

def mass_spring_damper_dynamics(t, state, u_func, m, k, c):
    """Mass-spring-damper: m*a + c*v + k*x = u(t)"""
    x, v = state
    u = u_func(t)
    a = (u - c * v - k * x) / m
    return [v, a]

def simulate_system(u_func, x0=0.0, v0=0.0):
    """Simulate the system with given initial conditions."""
    sol = solve_ivp(
        fun=lambda t, state: mass_spring_damper_dynamics(
            t, state, u_func, MASS, SPRING_CONSTANT, DAMPING_COEFFICIENT),
        t_span=T_SPAN,
        y0=[x0, v0],
        t_eval=T_EVAL,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    t = sol.t
    x = sol.y[0]
    v = sol.y[1]
    u = np.array([u_func(ti) for ti in t])
    return t, x, v, u

def save_trajectory_to_hdf5(filename, u, x, v, x0, v0):
    """Save trajectory to HDF5 with IC attributes."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('u', data=u.astype(np.float32))
        f.create_dataset('x', data=x.astype(np.float32))
        f.create_dataset('v', data=v.astype(np.float32))
        f.attrs['mass'] = MASS
        f.attrs['spring_constant'] = SPRING_CONSTANT
        f.attrs['damping_coefficient'] = DAMPING_COEFFICIENT
        f.attrs['dt'] = DT
        f.attrs['sampling_rate'] = SAMPLING_RATE
        f.attrs['duration'] = DURATION
        f.attrs['n_samples'] = N_SAMPLES
        f.attrs['x0'] = x0
        f.attrs['v0'] = v0

# --- Input signal definitions ---
def sine_1hz(t):    return 1.0 * np.sin(2 * np.pi * 1.0 * t)
def sine_05hz(t):   return 0.8 * np.sin(2 * np.pi * 0.5 * t)
def step_at_2(t):   return 1.5 if t >= 2.0 else 0.0
def multisine(t):   return (0.6 * np.sin(2*np.pi*0.3*t) +
                            0.4 * np.sin(2*np.pi*1.2*t) +
                            0.3 * np.sin(2*np.pi*2.5*t))
def sine_15hz(t):   return 1.2 * np.sin(2 * np.pi * 1.5 * t)
def chirp(t):
    f0, f1, T = 0.2, 3.0, DURATION
    return 1.0 * np.sin(2*np.pi * (f0*t + (f1-f0)/(2*T)*t**2))

def generate_dataset():
    """Generate variable-IC dataset."""
    output_path = Path(__file__).parent.parent / 'pinn_var_ic'

    for subdir in ['train', 'valid', 'test']:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)

    print(f"Generating variable-IC PINN dataset → {output_path}")

    # --- Train: 5 ICs × 4 inputs = 20 files ---
    train_ics = [(0, 0), (1, 0), (-1, 0), (0, 2), (0.5, -1.5)]
    train_inputs = [
        ('sine_1hz', sine_1hz),
        ('sine_05hz', sine_05hz),
        ('step', step_at_2),
        ('multisine', multisine),
    ]
    for x0, v0 in train_ics:
        for name, u_func in train_inputs:
            ic_tag = f"x{x0}_v{v0}".replace('.', 'p').replace('-', 'n')
            t, x, v, u = simulate_system(u_func, x0, v0)
            fname = output_path / 'train' / f'{ic_tag}_{name}.h5'
            save_trajectory_to_hdf5(str(fname), u, x, v, x0, v0)
            print(f"  train: {fname.name}")

    # --- Valid: 3 interpolation ICs × 2 inputs = 6 files ---
    valid_ics = [(0.5, 0), (-0.5, 1), (0, -1)]
    valid_inputs = [
        ('sine_1hz', sine_1hz),
        ('multisine', multisine),
    ]
    for x0, v0 in valid_ics:
        for name, u_func in valid_inputs:
            ic_tag = f"x{x0}_v{v0}".replace('.', 'p').replace('-', 'n')
            t, x, v, u = simulate_system(u_func, x0, v0)
            fname = output_path / 'valid' / f'{ic_tag}_{name}.h5'
            save_trajectory_to_hdf5(str(fname), u, x, v, x0, v0)
            print(f"  valid: {fname.name}")

    # --- Test: 4 ICs (includes extrapolation) × 2 inputs = 8 files ---
    test_ics = [(2, 3), (-2, -2), (1.5, 1), (0, 0)]
    test_inputs = [
        ('sine_15hz', sine_15hz),
        ('chirp', chirp),
    ]
    for x0, v0 in test_ics:
        for name, u_func in test_inputs:
            ic_tag = f"x{x0}_v{v0}".replace('.', 'p').replace('-', 'n')
            t, x, v, u = simulate_system(u_func, x0, v0)
            fname = output_path / 'test' / f'{ic_tag}_{name}.h5'
            save_trajectory_to_hdf5(str(fname), u, x, v, x0, v0)
            print(f"  test:  {fname.name}")

    n_train = len(train_ics) * len(train_inputs)
    n_valid = len(valid_ics) * len(valid_inputs)
    n_test = len(test_ics) * len(test_inputs)
    print(f"\nDone: {n_train} train, {n_valid} valid, {n_test} test "
          f"({n_train + n_valid + n_test} total)")

if __name__ == "__main__":
    generate_dataset()
