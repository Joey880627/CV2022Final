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
+ **model4.pth**
	+ Overall weighted IoU: 0.7756
	+ Average true negative rate: 0.8841
	+ Benchmark score: 0.8082
+ **model12.pth**
	+ Overall weighted IoU: 0.7745
	+ Average true negative rate: 0.9966
	+ Benchmark score: 0.8411
+ **model14.pth**
	+ Overall weighted IoU: 0.9340
	+ Average true negative rate: 0.9703
	+ Benchmark score: 0.9449

## TODO
1. Let the model predict confidence score
2. Deal with imbalance data (e.g. focal loss, dice loss, etc.)
3. Add CV preprocessed mask into input
4. Add image augmentation (albumentation library)

## Files
+ ***script/main.py***
	+ Training segmentation model
+ ***script/dataset.py***
	+ Wrapping customized PyTorch dataset
+ ***script/model.py***
	+ Get model from segmentation_models_pytorch
+ ***script/loss.py***
	+ Implement loss functions and evaluation metrics
+ ***script/predict.py***
	+ Predict function for eval.py
+ ***script/modelXYZ.pth***
	+ Model weights trained by data SX, SY, SZ