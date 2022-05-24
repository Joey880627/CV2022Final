import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from model import get_model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
THRESHOLD = 0.005

class Predictor:
    def __init__(self, model_path="model14.pth"):
        self.model = get_model().to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    @torch.no_grad()
    def predict(self, image_name):
        image = np.asarray(Image.open(image_name).convert('RGB'))
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
        image = transform(image).to(device)
        image = image.unsqueeze(0)
        out = self.model(image)
        _, pred = torch.max(out.data, 1)
        pred = pred[0].cpu().numpy()
        area = pred.sum() / np.prod(pred.shape)
        conf = float(area > THRESHOLD)
        return pred, conf
