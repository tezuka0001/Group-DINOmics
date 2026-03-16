# Group-DINOmics

This is an implementation of the paper: **Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning**
このリポジトリは研究室に保存用のものです．

## Environment
* Python 3.10.10  
* PyTorch 2.2.2
1. Please install the appropriate version of PyTorch from here (https://pytorch.org).
2. Please install the remaining dependencies by running.
```
pip install -r requiremets/requirements.txt
```

### DINOv3
We use [DINOv3](https://github.com/facebookresearch/dinov3) as the image feature extractor.
This project uses DINOv3 via Hugging Face Transformers. For detailed usage instructions, please refer to [this page](https://github.com/facebookresearch/dinov3)

### Inpaint
We use LaMa as the inpainting model.  
Please download the pretrained weights from here (https://github.com/advimman/lama).  
And place the downloaded weights at: `./lama/big-lama/models/best.ckpt`

## Data preparation
### 1. Download dataset
* Volleyball dataset (Dataset/volleyball)
  Please dawnload the dataset from here (https://github.com/mostafa-saad/deep-activity-rec).
* NBA dataset (Dataset/NBA_dataset)
  The dataset can be obtained by contacting the authors of “Social Adaptive Module for Weakly-supervised Group Activity Recognition, ECCV 2020” (https://ruiyan1995.github.io/SAM.html).

### 2. Optical Flow
We use RAFT (https://github.com/princeton-vl/RAFT) to compute optical flow.  
The precomputed optical flow is published here (coming soon).

  
### 3. Group-relevant Object Location
We use a ball as a group-relevant object in our experiments.
* Volleyball dataset
Annotated ball locations in the frames are available here (https://github.com/mostafa-saad/deep-activity-rec).  
Detected ball locations using the WASB detector (https://github.com/nttcom/WASB-SBDT) are published here (coming soon) for Weakly Supervised GAR.

* NBA dataset
Detected ball locations using the WASB detector are published here (coming soon), as in the Volleyball dataset.

### 4. File structure
│── Dataset/ <br/>
│   │── volleyball/ <br/>
│   │    └── videos/ <br/>
│   │    └── flow_numpy_sub_med/ <br/>
│   │    └── flow_numpy_sub_med_36x64/ <br/>
│   │    └── wasb/ <br/>
│   │    └── volleyball_tracks_deep_eiou/ <br/>
│   │    └── volleyball_weak/ <br/>
│   │    └── tracks_normalized.pkl/ <br/>
│   │    └── volleyball_net_detection_all_frames (if you use volleyball net location)/ <br/>
│   │── NBA_dataset/ <br/>
│   │    └── videos/ <br/>
│   │    └── train_video_ids <br/>
│   │    └── test_video_ids <br/>
│   │    └── flow_numpy_sub_med/ <br/>
│   │    └── flow_numpy_sub_med_36x64/ <br/>
│   │    └── wasb/ <br/>
│   │    └── tracks_deep_eiou_prune_72/ <br/>
│   │    └── nba (if you use basketball gaol location)/ <br/>
│   │    └── train_ids (limited data) <br/>
│   │── jrdb_par/ <br/>
│   │    └── videos/ <br/>
│   │    └── annotations <br/>
│   │    └── flow_numpy_sub_med_8x63 <br/>
│   │    └── flow_numpy_sub_med_8x63_2gap <br/>
│   │    └── flow_numpy_sub_med_8x63_3gap <br/>
│   │    └── flow_numpy_sub_med_8x63_7gap <br/>
│   │    └── flow_numpy_sub_med_8x63_15gap <br/>


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
Please run the model using the weights obtained by the first stage.
* Volleyball dataset
  
```
bash scripts/finetune_VBD.sh
```

* NBA dataset
  
```
bash scripts/finetune_NBA.sh
```

### JRDB PAR
```
bash scripts/jrdb_train_test_flow_ball.sh
```

### Group Activity Recognition
The model trained on the pretext task can also be used for pretraining in group activity recognition.
Please run the model using the weights by the second stage.
* Volleyball dataset
  
```
bash scripts/WSGAR_VBD.sh
```

* NBA dataset
  
```
bash scripts/WSGAR_NBA.sh
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

## Acknowledgement
This project builds upon many outstanding open-source projects and datasets. We would like to sincerely thank the authors and contributors of the following works:

* [DFWSGAR](https://github.com/dk-kim/DFWSGAR) and [GAFL](https://github.com/chihina/GAFL-CVPR2024) for group activity recgnition/retrieval.
 * [Volleyball dataset](https://github.com/mostafa-saad/deep-activity-rec), [MP-GCN](https://github.com/mgiant/MP-GCN) and [NBA dataset](https://ruiyan1995.github.io/SAM.html) for datasets.
* [DINOv3](https://github.com/facebookresearch/dinov3) for strong image feature extractor.
* [LaMa](https://github.com/advimman/lama) for inpainting.
* [RAFT](https://github.com/princeton-vl/RAFT) for optical flow estimation.
* [WASB](https://github.com/nttcom/WASB-SBDT) for ball tracking.
