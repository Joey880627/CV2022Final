import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from model import get_model
from dataset import data_preprocess
import cv2
import os


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
THRESHOLD = 500/(640*480)

model_path="resnet18.pth"
model = get_model().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

print(model_path)

@torch.no_grad()
def predict(image):
    # image = np.asarray(Image.open(image_name))
    # image = np.expand_dims(image, 2)
    if len(image.shape)==2:
        image = np.expand_dims(image, 2)
    elif image.shape[2]==3:
        image = image[..., :1]
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = data_preprocess(image)
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),])
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    out = model(image)
    _, pred = torch.max(out.data, 1)
    pred = pred[0].cpu().numpy()
    pred = pred*255
    pred = reserve_largest_component(pred)
    area = np.count_nonzero(pred) / np.prod(pred.shape)
    conf = float(area > THRESHOLD)
    return pred, conf

def reserve_largest_component(image):
    h,w = image.shape
    prediction = np.zeros(shape=(h,w))

    ret, image = cv2.threshold(image.astype(np.float32), 125, 255, cv2.THRESH_BINARY)#27
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image.astype(np.int8), connectivity=8)

    # stats[:,x] = [x, y, width, height, area]
    sizes = stats[:, -1]
    sizes_rank = np.argsort(sizes)
    if len(sizes_rank) >= 2:
        prediction[output == sizes_rank[-2]] = 255
    return prediction

def generate_output_file(label,conf,action_number,image_number):
    folder_name = f'./S5_solution/{action_number:02d}/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    filename = f'{image_number}.png'
    cv2.imwrite(os.path.join(folder_name,filename),label)


    conf_file_path = os.path.join(folder_name,'conf.txt')

    mode = 'a' if os.path.exists(conf_file_path) else 'w'
    with open(conf_file_path, mode) as f:
        f.write(f'{conf}\n')


