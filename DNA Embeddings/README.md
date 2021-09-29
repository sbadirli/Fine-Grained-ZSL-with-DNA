# DNA Embeddings

## Getting Started

This is the repo for running the Jupyter Notebooks created for learning DNA-Embeddings

## Prerequisites

The code was implemented in Python 3.7.10 and utilized the packages (full list) in requirements.txt file. The platform I used was Windows-10. Most important packages you need are the followings:
```
tensorflow=2.1.0
numpy=1.19.2 
python=3.7.10 
scipy=1.6.1 
jupyter notebook
```

## Installing

To run the code, You may create a conda environment (assuming you already have miniconda3 installed) by the following command on terminal:

```
conda create --name dna_embedding --file requirements.txt
```

## Data

You may download the data from this Dropbox [link](https://www.dropbox.com/sh/gt6tkech0nvftk5/AADOUJc_Bty3sqOsqWHxhmULa?dl=0). Please put dataset into `data` folder and move the `data` folder into the same directory which contains the folders for codes.


## Experiments

To reproduce the results from paper, first activate conda virtual environment

```
conda activate dna_embedding
```
Then simply navigate to the `DNA Embeddings` folder, and run following command to initialize the jupyter notebook

```
jupyter notebook
```
Then select the notebook you want to open and start the experiment.
