# NoFAS:
Variational Inference with NoFAS: Normalizing Flow with Adaptive Surrogate for Computationally Expensive Models

NoFAS estimates posterior distribution of hidden variables for a computationally expensive model with adaptive surrogate and normalizing flow. In particular, masked autoregressive flow (MAF) and RealNVP are used in the code, which are implemented by [Kamen Bliznashki](https://github.com/kamenbliznashki/normalizing_flows). The code includes four experiments in the paper: Closed form, RC, RCR, and non-isomorphic Sobol function. RC and RCR are implemented by [Daniele Schiavazzi](https://github.com/desResLab/tulip/commits?author=daneschi).

## Requirements:
* PyTorch 1.4.0
* Numpy 1.19.2
* Scipy 1.6.1

## Usage
* __run experiment__: Execute run_experiment.py by
```bash
python run_experiment.py
```
* __surrogate__: If surrogate models are enabled, .npz and .sur files containing surrogate information must be put in root directory. As a reference, surrogate model files for the four experiments are stored in source/surrogate. For generating new surrogate models, 


## Brief Documentation
* __run_experiment.py__:        Main file for running MAF on RC model.
* __experiment_setting.py__:    File containing class of experiment parameters
* __maf.py__:                   MAF Implementation, credit: [Kamen Bliznashki](https://github.com/kamenbliznashki/normalizing_flows)
* __circuitModels_Torch.py__:   Circuit Model Implementation, credit: [Daniele Schiavazzi](https://github.com/desResLab/tulip/commits?author=daneschi)
* __circuitModels_Trivial.py__:   Analytical Experiment Implementation
* __densities.py__:             Provides analytical densities or toy models
* __FNN_surrogate_nested.py__:            PyTorch Implementation for Fully-connected Surrogate Model for RC
