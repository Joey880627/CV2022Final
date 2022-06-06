from re import L
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

dataset_path = '../dataset/public'
subjects = ['S1', 'S2', 'S3', 'S4']

def get_data(dataset_path, subjects):
    images = [] # Raw images
    labels = [] # Masks of pupil
    for subject in subjects:
        for action_number in range(26):
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            if not os.path.exists(os.path.join(image_folder, '0.png')):
                print(f'Labels are not available for {image_folder}')
                continue
            for idx in range(nr_image):
                image_name = os.path.join(image_folder, f'{idx}.jpg')
                label_name = os.path.join(image_folder, f'{idx}.png')
                images.append(image_name)
                labels.append(label_name)
    return images, labels


def data_preprocess(image):
    hist = cv2.equalizeHist(image)
    # canny = cv2.Canny(hist, 15, 150)
    # blur = cv2.blur(hist, (5, 5))
    # laplacian = cv2.Laplacian(blur,cv2.CV_64F)
    _, threshold = cv2.threshold(hist, 27, 255, cv2.THRESH_BINARY_INV)
    image = np.stack([hist,threshold],axis = 2)
    return image.astype(np.float32)

class PupilDataset(Dataset):
    def __init__(self, images, labels, mode="train"):
        self.images = images
        self.labels = labels
        self.mode = mode
        # self.transform = transforms.Compose([
        #                 transforms.ToTensor(),
        #                 transforms.Normalize([0.5], [0.5]),])
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5,0.5], [0.5,0.5]),])
    def __getitem__(self, item):
        image = np.asarray(Image.open(self.images[item]))
        # image = np.expand_dims(image, 2)
        image = data_preprocess(image)

        label = np.asarray(Image.open(self.labels[item]).convert('RGB'))

        label = (label.sum(axis=-1) > 0).astype(np.int64)
        label_validity = int(np.sum(label.flatten()) > 0)
        image = self.transform(image)
        return image, label, label_validity

    def __len__(self):
        return len(self.images)

def get_dataset(dataset_path, subjects, split_ratio = 0.8):
    images, labels = get_data(dataset_path, subjects)


    info = np.stack( (np.array(images), np.array(labels)) ,axis=1)
    N = info.shape[0]
    np.random.shuffle(info)
    x = int(N*split_ratio) 
    
    all_images, all_labels = info[:,0].tolist(), info[:,1].tolist()


    train_image = all_images[:x]
    val_image = all_images[x:]

    train_label = all_labels[:x] 
    val_label = all_labels[x:]

    print(f"Training data: {len(train_image)}, validation data: {len(val_image)}")

    return PupilDataset(train_image, train_label),PupilDataset(val_image, val_label)

if __name__ == "__main__":
    # images, labels = get_data(dataset_path, subjects)
    # dataset = PupilDataset(images, labels)
    dataset, val_set = get_dataset(dataset_path, subjects, split_ratio=0.95)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    ones = 0
    """for image, label, label_validity in dataloader:
        print(image.shape, label.shape, label_validity.shape)
        print(image.max(), image.min())
        break
        print(label_validity)
        lv.append(label_validity.numpy())"""
    print(len(dataset))
    from tqdm import tqdm
    for image, label, label_validity in tqdm(dataloader):
        ones += label_validity.sum().item()
    print(ones)