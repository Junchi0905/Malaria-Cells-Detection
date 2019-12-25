import numpy as np
import torch
import torch.nn as nn
import cv2
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DROPOUT = 0.5

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, (11, 11), padding=5)
        self.convnorm1 = nn.BatchNorm2d(18)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(18, 36, (11, 11), padding=5)
        self.convnorm2 = nn.BatchNorm2d(36)
        self.pool2 = nn.AvgPool2d((2, 2))
        self.linear1 = nn.Linear(270000, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(400, 7)
        self.act = torch.relu
        self.act2 = torch.sigmoid


    def forward(self, x):

        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        return self.act2(self.linear2(x))

model = CNN().to(device)
model.load_state_dict(torch.load('model_junchi0905.pt'))
model.eval()

def predict(x):
    prediction = []
    for img_path in x:

        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (400, 300))
        resized_img = np.moveaxis(resized_img, -1, 0)
        arr = np.array(resized_img)
        arr = arr / 255.0
        arr = torch.from_numpy(arr).float().unsqueeze(0).to(device)
        y_pred = model(arr)
        y_pred = y_pred.round()
        y_pred = y_pred.cpu().detach()
        prediction.append(y_pred)

    prediction = torch.cat(prediction)
    return prediction
