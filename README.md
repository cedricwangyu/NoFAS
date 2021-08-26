# NoFAS
Variational Inference with NoFAS: Normalizing Flow with Adaptive Surrogate for Computationally Expensive Models

NoFAS estimates posterior distribution of hidden variables for a computationally expensive model with adaptive surrogate
and normalizing flow. In particular, masked autoregressive flow (MAF) and RealNVP are used in the code, which are 
implemented by [Kamen Bliznashki](https://github.com/kamenbliznashki/normalizing_flows). 
The code includes four experiments in the paper: Closed form, RC, RCR, and non-isomorphic Sobol function. 
RC and RCR are implemented by [Daniele Schiavazzi](https://github.com/desResLab/tulip/commits?author=daneschi).

## Requirements:
* PyTorch 1.4.0
* Numpy 1.19.2
* Scipy 1.6.1

## Usage
* __run experiment__: Execute run_experiment.py by
```bash
python run_experiment.py
```
* __surrogate__: If surrogate models are enabled, `.npz` and `.sur` files containing surrogate information must be put under 
  root directory. As a reference, surrogate model files for the four experiments are stored in `/source/surrogate`. 
  Generating new surrogate models could be done by calling `gen_grid` and `surrogate_save` methods in `FNN_surrogate_nested.py`.
* __results__: Hidden variable estimations are stored in `/result` but they are transformed from original scale. They could
  be converted back by calling `transform` method in model files.
* __Metropolis Hastings__: This method is implemented in `mh.py` which also contains testing functions designed for the four
experiments in the paper.

  
## Recommended Hyper Parameters
All experiments used RMSprop as the optimizer equipped exponential decay scheduler with decay factor 0.9999. All normalizing flows use relu as activation 
function and maximum number of iterations is 25001. All MADE contains 1 hidden layer with 100 nodes.

| Experiment  | NF type | NF layers | batch size | budget | updating size | updating period | learning rate |
| ----------- | ------- | --------- | ---------- | ------ | ------------- | --------------- | ------------- |
| closed form | RealNVP | 5         | 200        | 64     | 2             | 1000            | 0.002         |
| RC          | MAF     | 5         | 250        | 64     | 2             | 1000            | 0.003         |
| RCR         | MAF     | 5         | 500        | 216    | 2             | 300             | 0.003         |
| 5-dim       | RealNVP | 15        | 250        | 1023   | 12            | 250             | 0.0005        |