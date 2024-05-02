# -*- coding: utf-8 -*-
"""《Must Have 텐초의 파이토치 딥러닝 특강》 ex4_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jmv1QhTPcQVYyw-Qo5UMRSBwSqESCM0C

# 사전 학습된 모델 준비
"""

import torch
import torch.nn as nn

from torchvision.models.vgg import vgg16

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = vgg16(pretrained=True) # ❶ vgg16 모델 객체 생성
fc = nn.Sequential( # ❷ 분류층의 정의
       nn.Linear(512 * 7 * 7, 4096),
       nn.ReLU(),
       nn.Dropout(), #❷ 드롭아웃층 정의
       nn.Linear(4096, 4096),
       nn.ReLU(),
       nn.Dropout(),
       nn.Linear(4096, 10),
   )

model.classifier = fc # ➍ VGG의 classifier를 덮어씀
model.to(device)

"""# 데이터 전처리와 증강"""

import tqdm

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam

transforms = Compose([
   Resize(224),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   ToTensor(),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

"""# 데이터로더 정의"""

training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

"""# 학습 루프 정의"""

lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(5):
   iterator = tqdm.tqdm(train_loader) # ➊ 학습 로그 출력
   for data, label in iterator:
       optim.zero_grad()

       preds = model(data.to(device)) # 모델의 예측값 출력

       loss = nn.CrossEntropyLoss()(preds, label.to(device))
       loss.backward()
       optim.step()
     
       # ❷ tqdm이 출력할 문자열
       iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "CIFAR_pretrained.pth") # 모델 저장

model.load_state_dict(torch.load("CIFAR_pretrained.pth", map_location=device))

num_corr = 0

with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       preds = output.data.max(1)[1]
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")