# Bayesian Zero-Shot Learning (BZSL) model

## Getting Started

This is the Python version of Bayesian classifier - BZSL.

## Prerequisites

The code was implemented in Python 3.7.10 and utilized just 2 packages listed below. The platform I used was Windows 10. 
```
numpy=1.19.2 
scipy=1.6.1 
```

## Installing

To run the code, You may create a conda environment (assuming you already have miniconda3 installed) by the following command on terminal:

```
conda create --name bzsl --file requirements.txt
```

If you have already Python installed, you may just go ahead and manually install numpy nad scipy without creating virtual environment by `pip install numpy=1.19.2`.

## Data

You may download the data from this [link](https://www.dropbox.com/sh/gt6tkech0nvftk5/AADOUJc_Bty3sqOsqWHxhmULa?dl=0). Please put dataset into `data` folder and move the `data` folder into the same directory which contains the folders for codes.
 
## Experiments

To run the code and reproduce results from the paper, first activate conda virtual environment (If you chose to use virtual environment) by
```
conda activate bzsl
```
Then navigate to the `BZSL-Python` folder and execute the folowing code:
```
python Demo.py --dataset CUB --side_info dna
```

Main options for input arguments  are listed below (for the detailed lsit please check the code):
```
datatset: INSECT, CUB
side_info: original, w2v, dna
```


 
