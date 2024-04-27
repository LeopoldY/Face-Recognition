# Face Recognition Based on ArcFace

## Usages

### usage of training

Before training, you need to prepare the dataset and the datalist. The dataset should be organized as follows:

```bash
DATASET_ROOT
├── ID1
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── ID2
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

then you can use `generate_datalist.py` under the `utils` folder to generate the datalist, the datalist should be organized as follows:

```bash
label1 PATH/TO/IMAGE1
label1 PATH/TO/IMAGE2
...
```
After preparing the dataset and the datalist, you can start training with the following steps:

1. Install the required packages

2. Alter the `config.yml` file to your needs, the arguments are as follows:

```yaml
dataset_root: PATH/TO/DATASET/ROOT
datalist: PATH/TO/DATALIST
dataset: DATA_NAME
num_workers: 4 # number of workers in DataLoader
pin_memory: True # use pin_memory in DataLoader if your memory is big enough

device: gpu # cpu, gpu
backbone: iresnet18 # backbone network for face feature extraction

# training parameters
num_epochs: 50 # number of epochs
batch_size: 100
lr: 0.1
lr_gamma: 0.1 # decay rate of learning rate
step_size: 5 # decay learning rate every step_size epochs
weight_decay: 5e-4 

metric: arcface # metric function
loss: focal_loss # focal_loss, cross_entropy
gamma: 2 # parameter for focal loss
optimizer: sgd # sgd, adam

resume: True # False, True
resume_file: CHECKPOINT_FILE # checkpoint file to resume training

save_all_states: True # False, True
save_dir: PATH/TO/SAVE_DIR # directory to save checkpoints and training models

show_progress: True # False, True
```

3. Run the training script

```bash
python train.py
```

### usage of evaluation



