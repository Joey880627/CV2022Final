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
    # labels_validity = [] # 1 if eye open, else 0
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
                """image = cv2.imread(image_name)
                label = cv2.imread(label_name)
                images.append(image)
                labels.append(label)
                if np.sum(label.flatten()) > 0:
                    labels_validity.append(1.0)
                else:  # empty ground truth label
                    labels_validity.append(0.0)"""
    return images, labels

class PupilDataset(Dataset):
    def __init__(self, images, labels, mode="train"):
        self.images = images
        self.labels = labels
        self.mode = mode
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),])
    def __getitem__(self, item):
        image = np.asarray(Image.open(self.images[item]))
        image = np.expand_dims(image, 2)
        label = np.asarray(Image.open(self.labels[item]).convert('RGB'))

        label = (label.sum(axis=-1) > 0).astype(np.int64)
        # label = np.expand_dims(label, 0)
        label_validity = float(np.sum(label.flatten()) > 0)
        image = self.transform(image)
        return image, label, label_validity

    def __len__(self):
        return len(self.images)

def get_dataset(dataset_path, subjects):
    images, labels = get_data(dataset_path, subjects)
    return PupilDataset(images, labels)

if __name__ == "__main__":
    images, labels = get_data(dataset_path, subjects)
    dataset = PupilDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    for image, label, label_validity in dataloader:
        print(image.shape, label.shape, label_validity.shape)
        print(image.max(), image.min())