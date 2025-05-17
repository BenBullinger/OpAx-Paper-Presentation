# Paper Presentation of "Optimistic Active Exploration of Dynamical Systems" (Sukhija et al., 2023) 

Fork of [lasgroup/opax](https://github.com/lasgroup/opax) for a presentation of their paper ["Optimistic Active Exploration of Dynamical Systems" (Sukhija et al., 2023)](https://arxiv.org/abs/2306.12371) as part of the course "Foundations of Reinforcement Learning (263-5255-00L)" at ETH Zurich.

Original README below.
____

# Optimistic Active Exploration of Dynamical Systems
Implementation Model-based RL algorithms with basic optimizer.

Currently implemented:
1. SAC
2. CEM optimizer
3. BNNs for model learning with SVGD.

# Installation of library
pip install .

# Installation for GPU with Cuda 11
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.13+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl