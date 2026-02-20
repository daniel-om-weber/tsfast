# PINN Test Dataset

This minimal dataset is used for testing Physics-Informed Neural Networks (PINN) functionality in tsfast.

## Dataset Details

- **System**: Mass-spring-damper (m=1.0, k=1.0, c=0.1)
- **Equation**: ma + cv + kx = u
- **Duration**: 5 seconds @ 100Hz (500 samples per file)
- **Initial Conditions**: x₀=0, v₀=0 (all trajectories start from rest)
- **Total Size**: ~52KB (4 files)

## Files

- `train/trajectory_sine_1hz.h5` - 1 Hz sine wave input
- `train/trajectory_step.h5` - Step input at t=2s
- `valid/trajectory_sine_0.5hz.h5` - 0.5 Hz sine wave input
- `test/trajectory_sine_1.5hz.h5` - 1.5 Hz sine wave input

Each HDF5 file contains:
- `u`: forcing input [500 samples]
- `x`: position response [500 samples]
- `v`: velocity response [500 samples]
- Metadata: physical parameters, sampling rate, etc.

## Usage

See `tsfast/pinn/core.py` for PhysicsLossCallback and CollocationPointsCB.

## Generation

Run `generate_minimal_dataset.py` to regenerate the dataset.

