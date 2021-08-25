Three simple hemodynamic models are provided. Note how these model are a python/cython re-implementation of those provided through the tulip library used in the paper.

## RC Model

A model of a simple RC circuit. A pulsatile inflow is provided through the ```../data/inlet.flow``` file. 
Additionally, a simple dataset with multiple measurements of minimum, maximum and average pressure is provided in the file ```../data/rc_dataset.csv```.

The parameters are:
- **R**, the resistance parameter (default value is 1000.0 in GCS units).
- **C**, the capacitance parameter (default value is 0.00005 in CGS units).

The distal pressure is assumed to be constant in time and equal to 55 mmHg. 

The outputs are:
- **min_pressure**, minimum value of the proximal pressure over the last heart cycle. 
- **max_pressure**, maximum value of the proximal pressure over the last heart cycle. 
- **avg_pressure**, average value of the proximal pressure over the last heart cycle. 

The only state variable is the pressure at the capacitor, with a default initial condition set to 55.0 mmHg. 

## RCR Model

A model of a simple RC circuit. A pulsatile inflow is provided through the ```../data/inlet.flow``` file. 
Additionally, a simple dataset with multiple measurements of minimum, maximum and average pressure is provided in the file ```../data/rc_dataset.csv```.

The parameters are:
- **R1**, the proximal resistance parameter (default value is 1000.0 in GCS units).
- **R2**, the distal resistance parameter (default value is 1000.0 in GCS units).
- **C**, the capacitance parameter  (default value is 0.00005 in CGS units).

The distal pressure is assumed to be constant in time and equal to 55 mmHg. 

The outputs are:
- **min_pressure**, minimum value of the proximal pressure over the last heart cycle. 
- **max_pressure**, maximum value of the proximal pressure over the last heart cycle. 
- **avg_pressure**, average value of the proximal pressure over the last heart cycle. 

The only state variable is the pressure at the capacitor, with a default initial condition set to 55.0 mmHg. 

## Model for Adult Cardiovascular Physiology

For additional details on this model, please refer to the paper. 

## Compile Cython Module

To compile the cython module , cd in the ```models``` folder and type

```
python3 setup.py build_ext --inplace
```

****
