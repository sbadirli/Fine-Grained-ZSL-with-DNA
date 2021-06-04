# Bayesian Zero-Shot Learning (BZSL) model

## Getting Started

This is a ReadMe file for running the BZSL model and reproduce the results presented in the paper.

## Prerequisites

The code was implemented in Matlab 2020. Any version greater 2016 should be fine to run the code.

## Data

You can find datasets inside the supplementary material under the folder `data`.

## Experiments

To reproduce the results from the paper, open the `Demo.m` script and specify the dataset and model version (*unconstrained* for this work). Please change the datapath to your project path in `Demo.m` script.

If you want to perform hyperparameter tuning, please comment out relevant sections from `Demo.m` script.

You may alter the side information source for CUB data.
 
