"""The model architectures — the classes a user instantiates.

One module (or, when a model carries private fused kernels, one package) per
architecture: recurrent (:mod:`rnn`), convolutional (:mod:`cnn`), :mod:`transformer`,
the scan-family state-space models (:mod:`lru`, :mod:`s5`, :mod:`mamba`), :mod:`dynonet`,
the neural state space (:mod:`ssm`), :mod:`narx`, the port-Hamiltonian model (:mod:`phnn`),
and :mod:`subnet` building blocks. Shared machinery lives in :mod:`tsfast.models._core`.

All public names are re-exported through the :mod:`tsfast.models` facade.
"""
