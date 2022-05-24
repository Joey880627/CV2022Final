# CV2022Final - Pupil Tracking

## Getting started
```bash
# Copy dataset/ into this directory
# (Using python 3.8.13)
pip3 install -r requirements.txt

cd script/
# Use model.pth to perform prediction
python3 eval.py

# Train your own model
python3 main.py
```

## Current results
+ **model1.pth**
	+ Overall weighted IoU: 0.7649
	+ Average true negative rate: 0.8996
	+ Benchmark score: 0.8053

================================
Overall weighted IoU: 0.9813
Average true negative rate: 0.9478
Benchmark score: 0.9712
S1 results
================================
Overall weighted IoU: 0.6586
Average true negative rate: 0.9928
Benchmark score: 0.7589
S2 results
================================
Overall weighted IoU: 0.9332
Average true negative rate: 0.8969
Benchmark score: 0.9223
S3 results
================================

## TODO
1. Let the model predict confidence score
2. Deal with imbalance data (e.g. focal loss, dice loss, etc.)
3. Add CV preprocessed mask into input

## Files
+ ***script/main.py***
	+ Training segmentation model
+ ***script/dataset.py***
	+ Wrapping customized PyTorch dataset
+ ***script/model.py***
	+ Get model from segmentation_models_pytorch
+ ***script/loss.py***
	+ Implement 2D cross entropy loss
+ ***script/predict.py***
	+ Predict function for eval.py
+ ***script/model1.pth***
	+ Model weights trained by S1 data