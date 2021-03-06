import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from model import get_model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
THRESHOLD = 0.005

model_path="model1.pth"
model = get_model().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

@torch.no_grad()
def predict(image):
    # image = np.asarray(Image.open(image_name))
    # image = np.expand_dims(image, 2)
    if len(image.shape)==2:
        image = np.expand_dims(image, 2)
    elif image.shape[2]==3:
        image = image[..., :1]
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),])
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    out = model(image)
    _, pred = torch.max(out.data, 1)
    pred = pred[0].cpu().numpy()
    area = pred.sum() / np.prod(pred.shape)
    conf = float(area > THRESHOLD)
    return pred, conf
