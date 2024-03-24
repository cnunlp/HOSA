#  High-Order Semantic Alignment for Unsupervised Fine-Grained Image-Text Retrieval (HOSA)

This repository contains the implementation of our cutting-edge cross-modal visual-linguistic retrieval method, titled" High-Order Semantic Alignment for Unsupervised Fine-Grained Image-Text Retrieval". Our work has been accepted for publication in LREC-Coling 2024.

we present a novel High-Order Semantic Alignment (HOSA) model for unsupervised fine-grained image-text retrieval. Our main idea is to construct one modal's information of another modal's information with the linear combination of circulation in a common latent space.  Below are highlights showcasing the core concepts of our approach:
<p align="center">
  <b>The framework of the HOSA model.</b> <br> <br>
  <img src="model.png" width="80%">
</p>

<p align="center">
  <b>Retrieval</b> <br> <br>    
  <img src="retrieval.png" width="80%">
</p>

## Getting Started
### Prerequisites
    Git
    Conda environment

### Setup

1. Clone the Repository
```
git clone https://github.com/mesnico/HOAS
cd HOAS
```

2. Environment Setup
```
conda env create --file environment.yml
conda activate hosa
export PYTHONPATH=.
```
## Data Preparation
NOTE: The data files were temporarily moved to Google Drive due to a NAS failure.
### 1.Get the data
Download and extract the necessary data, including annotations and precomputed relevances.

```
wget http://datino.isti.cnr.it/hosa/data.tar
tar -xvf data.tar
```

### 2. Download Bottom-up Features
Obtain bottom-up features extracted using [Anderson et al.](https://github.com/peteanderson80/bottom-up-attention) for extracting them.
```
# for MS-COCO
wget http://datino.isti.cnr.it/hosa/features_36_coco.tar
tar -xvf features_36_coco.tar -C data/coco

# for Flickr30k
wget http://datino.isti.cnr.it/hosa/features_36_f30k.tar
tar -xvf features_36_f30k.tar -C data/f30k
```


## Evaluate
To evaluate our pre-trained models, follow these steps:

1.Download and extract our pre-trained HOAS models:
```
wget http://datino.isti.cnr.it/hosa/pretrained_models.tar
tar -xvf pretrained_models.tar
```

2.Run Evaluation
```
python3 test.py [model].pth --size 1k
python3 test.py [model].pth --size 5k
```


## Train
Train your model using a specific HOAS configuration with the following command:
```
python3 train.py --config configs/[config].yaml --logger_name runs/hosa
```

The training outputs, including tensorboard logs and checkpoints, will be saved in runs/hosa.

