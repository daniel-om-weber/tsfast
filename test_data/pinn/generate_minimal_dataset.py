#!/usr/bin/env python3
"""Generate minimal PINN test dataset for tsfast."""

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
    """Simulate the system."""
    initial_state = [x0, v0]
    sol = solve_ivp(
        fun=lambda t, state: mass_spring_damper_dynamics(t, state, u_func, MASS, SPRING_CONSTANT, DAMPING_COEFFICIENT),
        t_span=T_SPAN,
        y0=initial_state,
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

def save_trajectory_to_hdf5(filename, u, x, v):
    """Save trajectory to HDF5."""
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
        f.attrs['x0'] = 0.0
        f.attrs['v0'] = 0.0

def generate_dataset():
    """Generate minimal dataset."""
    output_path = Path(__file__).parent
    
    # Create directory structure
    for subdir in ['train', 'valid', 'test']:
        (output_path / subdir).mkdir(exist_ok=True)
    
    print("Generating minimal PINN dataset")
    print(f"Output: {output_path}")
    
    # Define trajectories
    trajectories = [
        ('train', 'sine_1hz', lambda t: 1.0 * np.sin(2 * np.pi * 1.0 * t)),
        ('train', 'step', lambda t: 1.5 if t >= 2.0 else 0.0),
        ('valid', 'sine_0.5hz', lambda t: 0.8 * np.sin(2 * np.pi * 0.5 * t)),
        ('test', 'sine_1.5hz', lambda t: 1.2 * np.sin(2 * np.pi * 1.5 * t)),
    ]
    
    for split, name, u_func in trajectories:
        t, x, v, u = simulate_system(u_func, x0=0.0, v0=0.0)
        filename = output_path / split / f'trajectory_{name}.h5'
        save_trajectory_to_hdf5(str(filename), u, x, v)
        print(f"Created {filename}")
    
    print(f"Dataset generation complete: 4 files total")

if __name__ == "__main__":
    generate_dataset()

