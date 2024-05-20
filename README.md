# Spatio-temporal-distribution-prediction-and-relationship-of-three-major-acute-cardiovascular-events

This is the repository for the analyses of the paper: *"Spatio-temporal distribution, prediction and relationship of three major acute cardiovascular events: out-of-hospital cardiac arrest, ST-elevation myocardial infarction and stroke"*.

The models are implemented in the `models/` directory and contain the files to reproduce the analyses presented in the paper.

- `INLA_univariate.R` contains the code for the univariate analysis of the three different series considered: OHCA, STEMI, and stroke.
- `INLA_aggregated.R` contains the code for the joint analysis of the three series.
- `Stacking_model.py` contains the code for the analysis of the three historical series using an ensemble model based on Machine Learning models.

<img src="https://github.com/Fede-stack/Spatio-temporal-distribution-prediction-and-relationship-of-three-major-acute-cardiovascular-events/blob/main/images/StackingModel.jpg" alt="" width="600">

**Figure 1. A representation of the MLMM model.**

<img src="https://github.com/Fede-stack/Spatio-temporal-distribution-prediction-and-relationship-of-three-major-acute-cardiovascular-events/blob/main/images/maps.png" alt="" width="500">

**Figure 2. Predicted risk map of Ticino for out-of-hospital cardiac arrest (OHCA), ST-elevation myocardial infarction (STEMI), stroke, and OHCA supported by past STEMIs and strokes.**
