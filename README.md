# MVSTT: A Multi-View Spatial-Temporal Transformer Network for Traffic Forecasting
<p align="center">
  <img width="1000"  src=./model/model.png>
</p>


## Requirements
- python 3.7.0
- torch==1.7.1
- scipy==1.7.0
- pandas==1.3.1
- numpy==1.20.3
## Data Preparation
MVSTT is implemented on those several public traffic datasets.
- **PEMS03**, **PEMS04**, **PEMS07** and **PEMS08** from [STSGCN (AAAI-20)](https://github.com/Davidham3/STSGCN).
- Due to the size limitation of the dataset, you can access our dataset in Google Cloud Drive.[PEMS](https://drive.google.com/drive/folders/1wxNZtR_a8uYm7E-JT1qIwEWNUehlQ6xM?usp=sharing)
## Model Train
PEMS03, PEMS04, PEMS07, PEMS08:

Run python train.py with  parameters.
- --dataset: The datasets name.
- --train_rate: The ratio of training set.
- --seq_len: The length of input sequence.
- --pre_len: The length of output sequence.
- --batchsize: Number of training batches.
- --heads: The number of heads of multi-head attention.
- --dropout: Dropout.
- --lr: Learning rate.
- --in_dim: Dimensionality of input data.
- --embed_size: Embedded dimension.
- --epochs：The number of train epochs.
```
python train.py --dataset PEMS08
```



## Model Test
The pre-trained model is available in Google Cloud Drive.[pre-trained model](https://drive.google.com/drive/folders/1SoO00z2BO_9sbZMNh2lx9WIanlOZb6B7?usp=sharing)

PEMS03, PEMS04, PEMS07, PEMS08:

Run python test.py with  parameters.
- --model: Trained model paths.

```
python test.py --dataset PEMS08 --model model/PEMS08/PEMS08.pkl
```

# MVSTT: A Multi-View Spatial-Temporal Transformer Network for Traffic Forecasting

# Abstrct


