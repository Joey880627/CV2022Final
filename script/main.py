import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import get_dataset
from model import get_model
from loss import cross_entropy2d

dataset_path = '../dataset/public'
train_subjects = ['S1']
valid_subjects = ['S2']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 8
lr = 0.001
epochs = 3

if __name__ == "__main__":
    train_dataset = get_dataset(dataset_path, train_subjects)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = get_dataset(dataset_path, valid_subjects)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_valid_acc = 0.0
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        train_total = 0.0
        model.train()
        for step, (image, label, label_validity) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            label_validity = label_validity.to(device)
            bsz = image.shape[0]

            out = model(image)
            optimizer.zero_grad()
            loss = cross_entropy2d(out, label)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out.data, 1)
            acc = pred.eq(label.data).cpu().sum() / np.prod(label.shape)
            train_loss += loss.item() * bsz
            train_acc += acc.item() * bsz
            train_total += bsz
            print(f"Epoch ({epoch}/{epochs}) Step ({step}/{len(train_loader)})  Train loss: {loss.item()}  Train acc: {acc.item()}")
        
        valid_loss = 0.0
        valid_acc = 0.0
        valid_total = 0.0
        model.eval()
        for step, (image, label, label_validity) in enumerate(valid_loader):
            image = image.to(device)
            label = label.to(device)
            label_validity = label_validity.to(device)
            bsz = image.shape[0]

            out = model(image)
            loss = cross_entropy2d(out, label)
            _, pred = torch.max(out.data, 1)
            acc = pred.eq(label.data).cpu().sum() / np.prod(label.shape)
            valid_loss += loss.item() * bsz
            valid_acc += acc.item() * bsz
            valid_total += bsz
            print(f"Epoch ({epoch}/{epochs}) Step ({step}/{len(valid_loader)})  Valid loss: {loss.item()}  Valid acc: {acc.item()}")


        train_loss = train_loss / train_total
        train_acc = train_acc / train_total
        valid_loss = valid_loss / valid_total
        valid_acc = valid_acc / valid_total
        print(f"Epoch {epoch}/{epochs}  Train loss: {train_loss}  Train acc: {train_acc}  Valid loss: {valid_loss}  Valid acc: {valid_acc}")
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            print("Saving model")
            torch.save(model.state_dict(), "model1.pth")