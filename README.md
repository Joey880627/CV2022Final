# CV2022Final - Pupil Tracking

## Getting started
```bash
# (Using python 3.8.13)
pip3 install -r requirements.txt

cd script/
# Use model.pth to perform prediction
python3 eval.py

# Train your own model
python3 main.py
```

## Files
- ***script/main.py***
	Training segmentation model
- ***script/dataset.py***
	Wrapping customized PyTorch dataset
- ***script/model.py***
	Get model from segmentation_models_pytorch
- ***script/loss.py***
	Implement 2D cross entropy loss
- ***script/predict.py***
	Predict function for eval.py
- ***script/model1.pth***
	Model weights trained by S1 data