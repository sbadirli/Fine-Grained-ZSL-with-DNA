# Bayesian Zero-Shot Learning (BZSL) model

## Getting Started

This is a ReadMe file for running the BZSL model and reproduce the results presented in the paper.

## Prerequisites

The code was implemented in Matlab 2020. Any version greater 2016 should be fine to run the code.

## Data

You may download the data from this anonymous [link](https://www.dropbox.com/sh/gt6tkech0nvftk5/AADOUJc_Bty3sqOsqWHxhmULa?dl=0). Please put dataset into `data` folder and move the `data` folder into the same directory which contains the folders for codes.

## Experiments

To reproduce the results from the paper, open the `Demo.m` script and specify the dataset and model version (*unconstrained* for this work). Please change the datapath to your project path in `Demo.m` script.

If you want to perform hyperparameter tuning, please comment out relevant sections from `Demo.m` script.

You may alter the side information source for CUB data.
 
