# BERT Lable Embedding
Code for "Integrating Task Specific Information into Pretrained Language Models for Low Resource Fine Tuning".

# Environment
The code is built on pytorch 1.2.0 and python 3.6. There may be slightly variation in performance with different running environment.

# Run
To run the modified BERT with label embedding, first install the package by,
```
cd BERT_label_embedding/
pip install .
```
Then, run the following for training with MRPC,
```
python CUDA_VISIBLE_DEVICES=0 ./script_glue_label.sh
```
Currently, it only supports single GPU. Please edit `examples/run_glue_label.py` for running with multiple GPUs.