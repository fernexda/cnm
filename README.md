# CNM

**CNM**: Cluster-based network modeling

## Description
**CNM** is a Python package for modeling of dynamical systems.

Cluster-based network modeling is a data-driven methodology for modeling nonlinear dynamical systems. The approach is developed within the network science and statistical physics frameworks. Cluster-based network modelling yields a deterministic stochastic gray-box model with adaptive coarse-graining resolution.

This method constitutes a novel automatable approach in data-driven nonlinear dynamical modeling and online control, opening up examples in cardiology, seismology, turbulence, etc. Details of the method are presented in Fernex ("Cluster-based network modeling: toward automated
robust modeling of complex dynamical systems"). 
The model is accurate and inherently robust, as the state will never leave the network, thus allowing accurate long-term
predictions.

These various features are demonstrated on the provided examples, that include the Lorenz attractor, the Kolmogorov
flow, and a high-dimensional actuated turbulent boundary layer. The different examples are specially selected to highlight the various capabilities of the method.

## Installation and dependencies

### Downloading the source
The official distribution is on GitHub, and you can clone the repository using
```console
git clone https://github.com/fernexda/cnm.git
```

### Dependencies
**CNM** requires the packages `numpy`, `matplotlib`, `sklearn` and `tqdm`. The examples require additionally `scipy`. The code is tested for Python 3 and not compatibility for Python 2 is guaranteed. The dependencies can be installed using
```console
pip install -r requirements.txt
```

## Getting started
The examples are the best place to start. They are located in `examples/` and include:
1. The Lorenz system (`lorenz.py`)
2. The Rössler attractor (`roessler.py`)
3. An electrocardiogram signal (`ecg.py`)
4. The dissipation energy of the Kolmogorov flow (`kolmogorov.py`)
5. An actuated turbulent boundary layer (`boundary_layer.py`)

The necessary data for these systems is either created directly by integrating the ODE (for the Lorenz and Rössler systems), or read from `example/data/`.

To run **CNM** on the Lorenz system, simply go into the `examples/` folder and run
```console
python lorenz.py
```
This will create the data, run **CNM** and generate the relevant plots.

## Getting help

If you encounter any issues using cnm, please use the repository's issue tracker. Consider the following steps before and when opening a new issue:

1. Have you searched for similar issues that may have been already reported? The issue tracker has a filter function to search for keywords in open issues.
2. Click on the green New issue button in the upper right corner, and describe your problem as detailed as possible. The issue should state what the problem is, what the expected behavior should be, and, maybe, suggest a solution. Note that you can also attach files or images to the issue.
3. Select a suitable label from the drop-down menu called Labels.
4. Click on the green Submit new issue button and wait for a reply.

## Reference

TODO: add citeable reference

## License

cnm is [GPL-3.0-licensed](https://en.wikipedia.org/wiki/GNU_General_Public_License#Version_3); refer to the LICENSE file for more information.
