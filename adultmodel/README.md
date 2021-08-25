## Repository Information 

This repository contains the supplementary material to the paper *K.K. Harrod, J.L. Rogers, J.A. Feinstein, A.L. Marsden and D.E. Schiavazzi*, **Predictive modeling of secondary pulmonary hypertension in left ventricular diastolic dysfunction**. A draft is available from [medrxiv](https://www.medrxiv.org/content/10.1101/2020.04.23.20073601v2).

We provide the two datasets used to generate the results in the paper. Additionally we also provide some python scripts to run zero-dimensional hemodynamic models representing simple RC and RCR models, and one model for adult cardiovascular physiology.

## Datasets

Two datasets are provided for the adult model:

- **validation_dataset.csv** includes anonymized hemodynamic data for a healthy patient and a patient with moderate and severe left ventricular diastolic dysfunction. See the paper for further details on this dataset.

- **EHR_dataset.csv** includes anonymized hemodynamic data for 84 patients. See the paper for further details on this dataset.

Additionally, other two files are used by the non-autonomous RC and RCR models:

- **inlet.flow** contains a time table wit the pulsatile inflow from the two models.

- **rc_dataset.csv** contains two measurements for minimum, maximum and average pressure at the inlet of the two models.

## Hemodynamic Models

- **rcModel** - A model of a simple RC circuit subject to a time dependent inflow. 

- **rcrModel** - A model of a simple RCR circuit subject to a time dependent inflow.

- **lpnAdultModel** - The circulation model used in the paper. Note how this model is already setup to assemble a Gaussian likelihood using the validation and EHR datasets provided in the **data** folder, as discussed in the paper. 

## Dependencies

The following libraries are required:

- **numpy** >= 1.19.2
- **scipy.signal** >= 1.5.2 - peek finding functionalities to compute acceletation/deceleration times in valve and E/A peak ratios for volumetric flow across valves.
- **cython** >= 0.29.19 - to reduce computation time RHS of the system of ODE and the RK4 time integration routines are implemented in cython. 
