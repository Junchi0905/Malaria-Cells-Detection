from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import random
import os
import numpy as np
import tensorflow as tf
import cv2
import torch
import torch.nn as nn
from torchvision import datasets

LR = 5e-2
N_EPOCHS = 15
BATCH_SIZE = 20
DROPOUT = 0.5

#read the data from the folder

SEED=42

imgs = []
labels = []
label1=[0 for n in range(7)]
dictionary ={"red blood cell":0, "difficult":1, "gametocyte":2, "trophozoite":3,"ring":4, "schizont":5, "leukocyte":6}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for file in os.scandir('train'):

    if file.name.endswith('png'):

        img = cv2.imread(os.path.join('train', file.name))

        resized_img = cv2.resize(img,(400,300))
        resized_img=np.moveaxis(resized_img, -1, 0)
        imgs.append(list(resized_img))
        txt = file.name[:-4]+'.txt'
        a = open(os.path.join('train', txt))
        label = a.read()
        label = str(label).split('\n')
        label1 = [0 for n in range(7)]
        for i in label:
            if i in dictionary.keys():
                label1[dictionary[i]] = 1
        labels.append(label1)
        a.close()
x = np.array(imgs)
y = np.array(labels)




x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2)
x_train, x_test = x_train/255, x_test/255

x_train, y_train =torch.from_numpy(x_train).float().to(device), torch.from_numpy(y_train).float().to(device)
x_train.requires_grad = True

x_test, y_test =torch.from_numpy(x_test).float().to(device), torch.from_numpy(y_test).float().to(device)


# %% -------------------------------------- CNN Class ------------------------------------------------------------------

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
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.BCELoss()
#start training the model
print("Starting training loop...")
for epoch in range(N_EPOCHS):
    loss_train=0
    model.train()
    for batch in range(len(x_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_test_pred = model(x_test)
        loss = criterion(y_test_pred, y_test)
        loss_test = loss.item()

    print("Epoch {} | Train Loss {:.5f},  Test Loss {:.5f}".format(
        epoch, loss_train / BATCH_SIZE, loss_test))

torch.save(model.state_dict(), "model_junchi0905.pt")
model = CNN().to(device)

model.load_state_dict(torch.load("model_junchi0905.pt"))
model.eval()

