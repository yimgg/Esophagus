# Esophagus Segmentation Tasks

## Train Process
The main training file is:
```
train.py
```
Moreover, if you want to use multi-card parallel training, you can adjust the relevant parameters and execute:
```
train.sh
```

## Datasets
### There are the Esophagus Segmentation Tasks:

### CVC-ClinicDB
1、The dataloader file is 
```
src \ CVCLoder.py
```
2、Please rename the Ground Truth folder in the CVC-ClinicDB decompressed data to GroundTruth, that is, remove the spaces. and fill the root path in the following files:
```
config.yml
```

### Other Task
Please add it according to the above format.\
Especially when adding additional dataloader files, do not modify src\CVCLoder.py




