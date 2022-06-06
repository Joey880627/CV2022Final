import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import get_dataset
from model import get_model
from loss import *

dataset_path = '../dataset/public'
train_subjects = ['S1', 'S2','S3', 'S4']
valid_subjects = ['S1']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 8
lr = 0.001
epochs = 5
loss_type = "ce" # ("ce", "weighted_ce", "dice", "iou")

pretrain = False
model_path="model1_.pth"

save_model_path = "resnet18_1234_2channel_classifier.pth"

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  

if __name__ == "__main__":
    train_dataset,valid_dataset = get_dataset(dataset_path, train_subjects,split_ratio=0.95)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # valid_dataset,_ = get_dataset(dataset_path, valid_subjects,split_ratio=0.05)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    model = get_model().to(device)
    if pretrain == True:
        model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if loss_type == "iou":
        iouLoss = JaccardLoss(mode= "multiclass", classes=[1])
    cls_criterion = nn.CrossEntropyLoss(weight=torch.Tensor([10., 1.])).to(device)
    best_valid_iou = 0.0
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        train_iou = 0.0
        train_total = 0.0
        model.train()
        for step, (image, label, label_validity) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            label_validity = label_validity.to(device)
            bsz = image.shape[0]

            out, cls_out = model(image)
            optimizer.zero_grad()
            if loss_type == "ce":
                seg_loss = cross_entropy2d(out, label)
            elif loss_type == "weighted_ce":
                seg_loss = cross_entropy2d(out, label, weight=torch.Tensor([1, 10]))
            elif loss_type == "dice":
                seg_loss = dice_loss(out, label)
            elif loss_type == "iou":
                seg_loss = iouLoss(out, label)
            cls_loss = cls_criterion(cls_out, label_validity)
            loss = seg_loss + cls_loss
            loss.backward()
            optimizer.step()
            _, seg_pred = torch.max(out.data, 1)
            _, pred = torch.max(cls_out.data, 1)
            acc = pred.eq(label_validity.data).cpu().sum() / np.prod(label_validity.shape)
            iou = binary_iou(seg_pred, label)
            train_loss += loss.item() * bsz
            train_acc += acc.item() * bsz
            train_iou += iou.item() * bsz
            train_total += bsz
            print(f"Epoch ({epoch+1}/{epochs}) Step ({step+1}/{len(train_loader)})  Train loss: {loss.item()}  Train acc: {acc.item()}  Train iou: {iou.item()}", end='\r')
        
        valid_loss = 0.0
        valid_acc = 0.0
        valid_iou = 0.0
        valid_total = 0.0
        model.eval()
        with torch.no_grad():
            for step, (image, label, label_validity) in enumerate(valid_loader):
                image = image.to(device)
                label = label.to(device)
                label_validity = label_validity.to(device)
                bsz = image.shape[0]
                out, cls_out = model(image)
                if loss_type == "ce":
                    seg_loss = cross_entropy2d(out, label)
                elif loss_type == "weighted_ce":
                    seg_loss = cross_entropy2d(out, label, weight=torch.Tensor([1, 10]))
                elif loss_type == "dice":
                    seg_loss = dice_loss(out, label)
                elif loss_type == "iou":
                    seg_loss = iouLoss(out, label)
                cls_loss = cls_criterion(cls_out, label_validity)
                loss = seg_loss + cls_loss
                _, seg_pred = torch.max(out.data, 1)
                _, pred = torch.max(cls_out.data, 1)
                acc = pred.eq(label_validity.data).cpu().sum() / np.prod(label_validity.shape)
                iou = binary_iou(seg_pred, label)
                valid_loss += loss.item() * bsz
                valid_acc += acc.item() * bsz
                valid_iou += iou.item() * bsz
                valid_total += bsz
                print(f"Epoch ({epoch+1}/{epochs}) Step ({step+1}/{len(valid_loader)})  Valid loss: {loss.item()}  Valid acc: {acc.item()}  Valid iou: {iou.item()}", end='\r')

        train_loss = train_loss / train_total
        train_acc = train_acc / train_total
        train_iou = train_iou / train_total
        valid_loss = valid_loss / valid_total
        valid_acc = valid_acc / valid_total
        valid_iou = valid_iou / valid_total
        print(f"Epoch ({epoch+1}/{epochs})  Train loss: {train_loss}  Train acc: {train_acc}  Train iou: {train_iou}  Valid loss: {valid_loss}  Valid acc: {valid_acc}  Valid iou: {valid_iou}")
        if valid_iou > best_valid_iou:
            best_valid_iou = valid_iou
            print("Saving model")
            torch.save(model.state_dict(), save_model_path)
        '''print("Saving model")
        torch.save(model.state_dict(), save_model_path)'''