# LESA-BERT (Label Embedding for Self-Attention in BERT) code
Code for "Students Need More Attention: BERT-based Attention Model for Small Data with Application to Automatic Patient Message Triage".

# Environment
The code is built on pytorch 1.2.0 and python 3.6. There may be slightly variation in performance with different running environment.

# Run
To run the modified BERT with label embedding, first install the package by,
```
cd text_classifiers/
pip install .
```
Then, run the following for training with MRPC,
```
python CUDA_VISIBLE_DEVICES=0 ./script_glue_label.sh
```
Currently, it only supports single GPU. Please edit `examples/run_glue_label.py` for running with multiple GPUs.

# Citation
Unfortunately, we can't release the message-urgency data used in our [paper](http://proceedings.mlr.press/v126/si20a.html) due to privacy issues. If you use our methods, please cite our work:
```bibtex
@InProceedings{pmlr-v126-si20a, 
title = {Students Need More Attention: BERT-based Attention Model for Small Data with Application to Automatic Patient Message Triage}, 
author = {Si, Shijing and Wang, Rui and Wosik, Jedrek and Zhang, Hao and Dov, David and Wang, Guoyin and Carin, Lawrence}, 
booktitle = {Proceedings of the 5th Machine Learning for Healthcare Conference}, 
pages = {436--456}, 
year = {2020}, 
editor = {Finale Doshi-Velez and Jim Fackler and Ken Jung and David Kale and Rajesh Ranganath and Byron Wallace and Jenna Wiens}, 
volume = {126}, 
series = {Proceedings of Machine Learning Research}, address = {Virtual}, month = {07--08 Aug}, publisher = {PMLR}, 
pdf = {http://proceedings.mlr.press/v126/si20a/si20a.pdf}, 
url = {http://proceedings.mlr.press/v126/si20a.html} }
```
