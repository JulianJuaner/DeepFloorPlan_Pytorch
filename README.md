# 5210-DeepFloorPlan
This is a pytorch implementation for the floor plan segmentation on the r2v dataset.

As well as a simple 3D mesh modeling script with the ModelNet dataset.

if you need to train the model, 
you need to replace path in the bash script / config file to the your directory \
ModelNet Dataset and r2v/jp should be placed in the `./data`.
## TRAIN
```bash sourcecode/exp/02/train.sh```
## INFERENCE (TEST)
```bash sourcecode/exp/02/inference.sh```
## MESH RECONSTRUCTION
```bash sourcecode/exp/02/mesh.sh```