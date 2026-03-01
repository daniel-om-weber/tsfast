"""Excitation signal generation for PINN collocation points."""

__all__ = [
    "DEFAULT_SIGNAL_TYPES",
    "DEFAULT_SIGNAL_PARAMS",
    "generate_excitation_signals",
    "generate_random_states",
]

import numpy as np
import torch

DEFAULT_SIGNAL_TYPES = ["sine", "multisine", "step", "ramp", "chirp", "noise", "prbs", "square", "doublet"]

DEFAULT_SIGNAL_PARAMS = {
    "sine": {"phase_range": (0, 2 * np.pi)},
    "multisine": {"n_components": 3},
    "step": {"time_range": (0.2, 0.8)},
    "ramp": {"slope_range": (-0.5, 0.5), "start_time_range": (0.1, 0.6)},
    "chirp": {"f0_range": (0.1, 1.0), "f1_range": (1.0, 5.0)},
    "prbs": {"switch_probability": 0.02},
    "square": {"duty_cycle_range": (0.3, 0.7)},
    "doublet": {"duration_range": (0.05, 0.2), "start_time_range": (0.2, 0.5)},
}


def _generate_sine(
    seq_length: int,
    dt: float,
    amplitudes: np.ndarray,
    frequencies: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Generate multiple sine waves vectorized."""
    t = np.arange(seq_length) * dt
    return amplitudes[:, np.newaxis] * np.sin(2 * np.pi * frequencies[:, np.newaxis] * t + phases[:, np.newaxis])


def _generate_multisine(
    seq_length: int,
    dt: float,
    amplitudes: np.ndarray,
    all_frequencies: list,
) -> np.ndarray:
    """Generate multiple multisine signals vectorized."""
    num_signals = len(amplitudes)
    t = np.arange(seq_length) * dt
    signals = np.zeros((num_signals, seq_length))
    for i in range(num_signals):
        freqs = all_frequencies[i]
        amp_per_component = amplitudes[i] / len(freqs)
        for freq in freqs:
            signals[i] += amp_per_component * np.sin(2 * np.pi * freq * t)
    return signals


def _generate_step(
    seq_length: int,
    amplitudes: np.ndarray,
    step_indices: np.ndarray,
) -> np.ndarray:
    """Generate multiple step signals vectorized."""
    num_signals = len(amplitudes)
    signals = np.zeros((num_signals, seq_length))
    for i in range(num_signals):
        signals[i, step_indices[i] :] = amplitudes[i]
    return signals


def _generate_ramp(
    seq_length: int,
    amplitudes: np.ndarray,
    slopes: np.ndarray,
    start_indices: np.ndarray,
) -> np.ndarray:
    """Generate multiple ramp signals vectorized."""
    num_signals = len(amplitudes)
    signals = np.zeros((num_signals, seq_length))
    for i in range(num_signals):
        start_idx = start_indices[i]
        ramp_length = seq_length - start_idx
        signals[i, start_idx:] = slopes[i] * np.arange(ramp_length)
        signals[i] = np.clip(signals[i], -abs(amplitudes[i]), abs(amplitudes[i]))
    return signals


def _generate_chirp(
    seq_length: int,
    dt: float,
    amplitudes: np.ndarray,
    f0s: np.ndarray,
    f1s: np.ndarray,
) -> np.ndarray:
    """Generate multiple chirp signals vectorized."""
    t = np.arange(seq_length) * dt
    duration = seq_length * dt
    k = (f1s - f0s) / duration
    phases = 2 * np.pi * (f0s[:, np.newaxis] * t + 0.5 * k[:, np.newaxis] * t * t)
    return amplitudes[:, np.newaxis] * np.sin(phases)


def _generate_noise(
    seq_length: int,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Generate multiple Gaussian white noise signals vectorized."""
    num_signals = len(amplitudes)
    return amplitudes[:, np.newaxis] * np.random.randn(num_signals, seq_length)


def _generate_prbs(
    seq_length: int,
    amplitudes: np.ndarray,
    switch_probs: np.ndarray,
) -> np.ndarray:
    """Generate multiple PRBS signals."""
    num_signals = len(amplitudes)
    signals = np.zeros((num_signals, seq_length))
    for i in range(num_signals):
        current_level = amplitudes[i] if np.random.rand() < 0.5 else -amplitudes[i]
        signals[i, 0] = current_level
        for j in range(1, seq_length):
            if np.random.rand() < switch_probs[i]:
                current_level = -current_level
            signals[i, j] = current_level
    return signals


def _generate_square(
    seq_length: int,
    dt: float,
    amplitudes: np.ndarray,
    frequencies: np.ndarray,
    duty_cycles: np.ndarray,
) -> np.ndarray:
    """Generate multiple square wave signals vectorized."""
    t = np.arange(seq_length) * dt
    periods = 1.0 / frequencies
    phases = (t[np.newaxis, :] % periods[:, np.newaxis]) / periods[:, np.newaxis]
    return amplitudes[:, np.newaxis] * ((phases < duty_cycles[:, np.newaxis]).astype(float) * 2 - 1)


def _generate_doublet(
    seq_length: int,
    amplitudes: np.ndarray,
    duration_indices: np.ndarray,
    start_indices: np.ndarray,
) -> np.ndarray:
    """Generate multiple doublet signals."""
    num_signals = len(amplitudes)
    signals = np.zeros((num_signals, seq_length))
    for i in range(num_signals):
        end_idx = min(start_indices[i] + duration_indices[i], seq_length)
        signals[i, start_indices[i] : end_idx] = amplitudes[i]
    return signals


def generate_excitation_signals(
    batch_size: int,
    seq_length: int,
    n_inputs: int = 1,
    dt: float = 0.01,
    device: str = "cpu",
    signal_types: list | None = None,
    amplitude_range: tuple = (0.5, 2.0),
    frequency_range: tuple = (0.1, 3.0),
    input_configs: list | None = None,
    noise_probability: float = 0.0,
    noise_std_range: tuple = (0.05, 0.15),
    bias_probability: float = 0.0,
    bias_range: tuple = (-0.5, 0.5),
    synchronized_inputs: bool = False,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate standard excitation signals for PINN collocation points (vectorized).

    Args:
        batch_size: number of sequences in batch
        seq_length: length of each sequence
        n_inputs: number of input dimensions
        dt: time step
        device: device for tensors
        signal_types: signal types to use (None = all core types)
        amplitude_range: global amplitude range
        frequency_range: global frequency range
        input_configs: per-input configuration (list of dicts)
        noise_probability: probability of adding noise per sequence
        noise_std_range: noise std as fraction of amplitude
        bias_probability: probability of adding DC bias per sequence
        bias_range: DC bias range
        synchronized_inputs: if True, all inputs get same signal type
        seed: random seed

    Returns:
        Excitation signal tensor of shape [batch_size, seq_length, n_inputs].
    """
    if seed is not None:
        np.random.seed(seed)

    signal_types = signal_types or DEFAULT_SIGNAL_TYPES
    total_signals = batch_size * n_inputs

    # Build configuration for each signal
    configs = []
    for batch_idx in range(batch_size):
        for input_idx in range(n_inputs):
            if input_configs and input_idx < len(input_configs):
                input_cfg = input_configs[input_idx]
                configs.append(
                    {
                        "amplitude_range": input_cfg.get("amplitude_range", amplitude_range),
                        "frequency_range": input_cfg.get("frequency_range", frequency_range),
                        "signal_types": input_cfg.get("signal_types", signal_types),
                        "signal_params": input_cfg.get("signal_params", {}),
                    }
                )
            else:
                configs.append(
                    {
                        "amplitude_range": amplitude_range,
                        "frequency_range": frequency_range,
                        "signal_types": signal_types,
                        "signal_params": {},
                    }
                )

    # Pre-sample signal types (handling synchronized_inputs)
    signal_type_choices = np.empty(total_signals, dtype=object)
    if synchronized_inputs:
        for batch_idx in range(batch_size):
            sig_type = np.random.choice(signal_types)
            for input_idx in range(n_inputs):
                idx = batch_idx * n_inputs + input_idx
                signal_type_choices[idx] = sig_type
    else:
        for idx, cfg in enumerate(configs):
            signal_type_choices[idx] = np.random.choice(cfg["signal_types"])

    # Pre-sample amplitudes
    amplitudes = np.array([np.random.uniform(*cfg["amplitude_range"]) for cfg in configs])

    # Group signals by type
    result = np.zeros((total_signals, seq_length))
    for sig_type in set(signal_type_choices):
        indices = np.where(signal_type_choices == sig_type)[0]
        if len(indices) == 0:
            continue

        type_amps = amplitudes[indices]
        type_configs = [configs[i] for i in indices]

        # Generate based on signal type
        if sig_type == "sine":
            freqs = np.array(
                [
                    np.random.uniform(
                        *cfg["signal_params"].get("sine", {}).get("frequency_range", cfg["frequency_range"])
                    )
                    for cfg in type_configs
                ]
            )
            phases = np.array(
                [
                    np.random.uniform(
                        *cfg["signal_params"]
                        .get("sine", {})
                        .get("phase_range", DEFAULT_SIGNAL_PARAMS["sine"]["phase_range"])
                    )
                    for cfg in type_configs
                ]
            )
            result[indices] = _generate_sine(seq_length, dt, type_amps, freqs, phases)

        elif sig_type == "multisine":
            all_freqs = []
            for cfg in type_configs:
                n_comp = (
                    cfg["signal_params"]
                    .get("multisine", {})
                    .get("n_components", DEFAULT_SIGNAL_PARAMS["multisine"]["n_components"])
                )
                freq_range = cfg["signal_params"].get("multisine", {}).get("frequency_range", cfg["frequency_range"])
                all_freqs.append(np.random.uniform(*freq_range, size=n_comp))
            result[indices] = _generate_multisine(seq_length, dt, type_amps, all_freqs)

        elif sig_type == "step":
            time_ranges = np.array(
                [
                    [
                        *cfg["signal_params"]
                        .get("step", {})
                        .get("time_range", DEFAULT_SIGNAL_PARAMS["step"]["time_range"])
                    ]
                    for cfg in type_configs
                ]
            )
            step_times = np.random.uniform(time_ranges[:, 0], time_ranges[:, 1])
            step_indices = (step_times * seq_length).astype(int)
            result[indices] = _generate_step(seq_length, type_amps, step_indices)

        elif sig_type == "ramp":
            slope_ranges = np.array(
                [
                    [
                        *cfg["signal_params"]
                        .get("ramp", {})
                        .get("slope_range", DEFAULT_SIGNAL_PARAMS["ramp"]["slope_range"])
                    ]
                    for cfg in type_configs
                ]
            )
            start_time_ranges = np.array(
                [
                    [
                        *cfg["signal_params"]
                        .get("ramp", {})
                        .get("start_time_range", DEFAULT_SIGNAL_PARAMS["ramp"]["start_time_range"])
                    ]
                    for cfg in type_configs
                ]
            )
            slopes = np.random.uniform(slope_ranges[:, 0], slope_ranges[:, 1])
            start_times = np.random.uniform(start_time_ranges[:, 0], start_time_ranges[:, 1])
            start_indices = (start_times * seq_length).astype(int)
            result[indices] = _generate_ramp(seq_length, type_amps, slopes, start_indices)

        elif sig_type == "chirp":
            f0_ranges = np.array(
                [
                    [*cfg["signal_params"].get("chirp", {}).get("f0_range", DEFAULT_SIGNAL_PARAMS["chirp"]["f0_range"])]
                    for cfg in type_configs
                ]
            )
            f1_ranges = np.array(
                [
                    [*cfg["signal_params"].get("chirp", {}).get("f1_range", DEFAULT_SIGNAL_PARAMS["chirp"]["f1_range"])]
                    for cfg in type_configs
                ]
            )
            f0s = np.random.uniform(f0_ranges[:, 0], f0_ranges[:, 1])
            f1s = np.random.uniform(f1_ranges[:, 0], f1_ranges[:, 1])
            result[indices] = _generate_chirp(seq_length, dt, type_amps, f0s, f1s)

        elif sig_type == "noise":
            result[indices] = _generate_noise(seq_length, type_amps)

        elif sig_type == "prbs":
            switch_probs = np.array(
                [
                    cfg["signal_params"]
                    .get("prbs", {})
                    .get("switch_probability", DEFAULT_SIGNAL_PARAMS["prbs"]["switch_probability"])
                    for cfg in type_configs
                ]
            )
            result[indices] = _generate_prbs(seq_length, type_amps, switch_probs)

        elif sig_type == "square":
            freqs = np.array(
                [
                    np.random.uniform(
                        *cfg["signal_params"].get("square", {}).get("frequency_range", cfg["frequency_range"])
                    )
                    for cfg in type_configs
                ]
            )
            duty_cycle_ranges = np.array(
                [
                    [
                        *cfg["signal_params"]
                        .get("square", {})
                        .get("duty_cycle_range", DEFAULT_SIGNAL_PARAMS["square"]["duty_cycle_range"])
                    ]
                    for cfg in type_configs
                ]
            )
            duty_cycles = np.random.uniform(duty_cycle_ranges[:, 0], duty_cycle_ranges[:, 1])
            result[indices] = _generate_square(seq_length, dt, type_amps, freqs, duty_cycles)

        elif sig_type == "doublet":
            duration_ranges = np.array(
                [
                    [
                        *cfg["signal_params"]
                        .get("doublet", {})
                        .get("duration_range", DEFAULT_SIGNAL_PARAMS["doublet"]["duration_range"])
                    ]
                    for cfg in type_configs
                ]
            )
            start_time_ranges = np.array(
                [
                    [
                        *cfg["signal_params"]
                        .get("doublet", {})
                        .get("start_time_range", DEFAULT_SIGNAL_PARAMS["doublet"]["start_time_range"])
                    ]
                    for cfg in type_configs
                ]
            )
            durations = np.random.uniform(duration_ranges[:, 0], duration_ranges[:, 1])
            start_times = np.random.uniform(start_time_ranges[:, 0], start_time_ranges[:, 1])
            duration_indices = np.maximum(1, (durations * seq_length).astype(int))
            start_indices = (start_times * seq_length).astype(int)
            result[indices] = _generate_doublet(seq_length, type_amps, duration_indices, start_indices)

    # Vectorized composition: noise and bias
    if noise_probability > 0:
        noise_flags = np.random.rand(total_signals) < noise_probability
        if noise_flags.any():
            noise_stds = np.random.uniform(*noise_std_range, size=total_signals) * np.abs(amplitudes)
            noise_signals = _generate_noise(seq_length, noise_stds[noise_flags])
            result[noise_flags] += noise_signals

    if bias_probability > 0:
        bias_flags = np.random.rand(total_signals) < bias_probability
        if bias_flags.any():
            biases = np.random.uniform(*bias_range, size=bias_flags.sum())
            result[bias_flags] += biases[:, np.newaxis]

    # Reshape and convert to torch
    result = result.reshape(batch_size, n_inputs, seq_length).transpose(0, 2, 1)
    return torch.from_numpy(result).to(device).float()


def generate_random_states(
    batch_size: int,
    n_outputs: int,
    output_ranges: list,
    device: str = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    """Generate random physical states for PINN collocation points.

    Args:
        batch_size: number of states to generate
        n_outputs: number of output dimensions
        output_ranges: list of (min, max) tuples for each dimension
        device: device for tensor
        seed: random seed

    Returns:
        Random state tensor of shape [batch_size, n_outputs].
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure output_ranges is a list
    if not isinstance(output_ranges, list):
        output_ranges = [output_ranges] * n_outputs

    states = np.zeros((batch_size, n_outputs))
    for i in range(n_outputs):
        min_val, max_val = output_ranges[i]
        states[:, i] = np.random.uniform(min_val, max_val, batch_size)

    return torch.from_numpy(states).to(device).float()
