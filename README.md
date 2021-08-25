# NoFAS:
Variational Inference with NoFAS: Normalizing Flow with Adaptive Surrogate for Computationally Expensive Models

Enabling MAF on Circuit Model RC and RCR (under construction) with PyTorch.


## Requirements:
* PyTorch 1.5.0
* Numpy 1.19.2
* Scipy 1.5.2

## Usage

```bash
python run_experiment.py
```
Parameters for Normalizing Flow type, architecture, optimizer, experiment setting could be modified in run_experiment.py
## Brief Documentation
* __run_experiment.py__:        Main file for running MAF on RC model.
* __experiment_setting.py__:    File containing class of experiment parameters
* __maf.py__:                   MAF Implementation, credit: [Kamen Bliznashki](https://github.com/kamenbliznashki/normalizing_flows)
* __circuitModels_Torch.py__:   Circuit Model Implementation, credit: [Daniele Schiavazzi](https://github.com/desResLab/tulip/commits?author=daneschi)
* __circuitModels_Trivial.py__:   Analytical Experiment Implementation
* __densities.py__:             Provides analytical densities or toy models
* __FNN_surrogate_nested.py__:            PyTorch Implementation for Fully-connected Surrogate Model for RC
