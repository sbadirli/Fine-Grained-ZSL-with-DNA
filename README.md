# Fine-grained ZSL with DNA as Side Information.

GitHub repo for our paper "Fine-Grained Zero-Shot Learning with DNA as Side Information" (to appear in NeurIPS'21)

Preprint can be accessed at: [Arxiv](https://arxiv.org/abs/2109.14133)

## Abstract
Fine-grained zero-shot learning task requires some form of side-information totransfer discriminative information from seen to unseen classes.  As manually annotated visual attributes are extremely costly and often impractical to obtain fora large number of classes, in this study we use DNA as side information for the first time for fine-grained zero-shot classification of species. Mitochondrial DNA plays an important role as a genetic marker in evolutionary biology and has been used to achieve near perfect accuracy in species classification of living organisms. We implement a simple hierarchical Bayesian model that uses DNA informationto establish the hierarchy in the image space and employs local priors to define surrogate classes for unseen ones. On the benchmark CUB dataset we show that DNA can be equally promising, yet in general a more accessible alternative than word vectors as a side information. This is especially important as obtaining robustword representations for fine-grained species names is not a practicable goal wheninformation about these species in free-form text is limited. On a newly compiledfine-grained insect dataset that uses DNA information from over a thousand specieswe show that the Bayesian approach outperforms state-of-the-art by a wide margin.

<p align="center">
  <img width="1000" height="200" src="NIPS_att_diagram.png">
</p>
<p align="justify">
  
## INSECT Images 
<p align="center">
<img width="600" src="NIPS_image_samples_final.png">
</p>
<p align="justify">
  
The raw (RGB) INSECT images (2.5GB) can be obtained from this [One Drive](https://indiana-my.sharepoint.com/:f:/g/personal/sbadirli_iu_edu/Ek2KDBxTndlFl_7XblTL-8QBZ6b0C0izgDJIBJQlWtiRKA?e=bCfCMH). Each folder inside represents INSECT species with its scientific names. For the processed data to run the code, please see the relevant folders.
  
## Notes
Please check the `BZSL-Python` (Python code for Bayesian classifier) and `BZSL` (Matlab code for Bayesian classifier) to run experiments and reproduce the results from the paper. To check the CNN model for DNA embeddings, please see  `DNA embeddings`.
  
## Contact
Feel free to drop me an email if you have any questions: s.badirli@gmail.com
  
## Citation
If you use the data or code please cite our papers as below:
```
@inproceedings{badirli2021bzsl,
  title={Fine-Grained Zero-Shot Learning with DNA as Side Information},
  author={Badirli, Sarkhan and Akata, Zeynep and Mohler, George and Picard, Christine and Dundar, Murat},
  booktitle={Neural Information Processing Systems},
  year={2021}
}
```

