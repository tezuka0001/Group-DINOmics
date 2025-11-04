# Group-DINOmics

This is an implementation of the paper: **Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning**

## Environment
Python 3.10.10
PyTorch 2.1.0

```
pip install -r requiremets/requirements.txt
```

We use LaMa as the inpainting model.
Please download the pretrained weights from here (https://github.com/advimman/lama).
And place the downloaded weights at: `./lama/big-lama/models/best.ckpt`

## Data preparation
### 1. Download dataset
* Volleyball dataset (Dataset/volleyball)
  https://github.com/mostafa-saad/deep-activity-rec
* NBA dataset (Dataset/NBA_dataset)
  The dataset can be obtained by contacting the authors of “Social Adaptive Module for Weakly-supervised Group Activity Recognition, ECCV 2020” (https://ruiyan1995.github.io/SAM.html).

### 2. Optical Flow
We use RAFT (https://github.com/princeton-vl/RAFT) to compute optical flow.
The precomputed optical flow is published here (coming soon).

  
### 3 Group-relevant Object location
We use a ball as a group-relevant object in our experiments.
* Volleyball dataset
Annotated ball locations in the frames are available here (https://github.com/mostafa-saad/deep-activity-rec).
Detected ball locations using the WASB detector (https://github.com/nttcom/WASB-SBDT) are published here (coming soon) for Weakly Supervised GAR.

* NBA dataset
Detected ball locations using the WASB detector are published here (coming soon), as in the Volleyball dataset.

## Training
### 1. First stage
Our model is trained with the Flow Estimation Loss in the first stage.
* Volleyball dataset
  
```
bash scripts/train_VBD.sh
```

* NBA dataset
  
```
bash scripts/train_NBA.sh
```

### 2. Second stage
Our model is trained with the Group-relevant Object Location Loss in the second stage.
* Volleyball dataset
  
```
bash scripts/finetune_VBD.sh
```

* NBA dataset
  
```
bash scripts/finetune_NBA.sh
```

## Evaluation
Trained models are published here (coming soon).
* Volleyball dataset
  
```
bash scripts/test_VBD.sh
```

* NBA dataset
  
```
bash scripts/test_NBA.sh
```