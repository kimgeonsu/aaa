import torch
import torchvision.models as models
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor


class CnnNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=64, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(in_features=64*7*7, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):

        # Building Block 1
        x = self.conv1(x)  # in_img_size=(28,28), in_channels=1,
        # out_channels=256, kernel_size=3, padding=1, out_img_size=(28,28)
        # in_img_size=(28,28), out_channels=256, out_img_size=(28,28)
        x = self.relu(x)
        # in_img_size=(28,28), in_channels=256, kernel_size=2, stride=2
        x = self.pool(x)
        # out_channels=256,out_img_size=(14,14)

        # Building Block 2
        # in_img=(14,14), in_channels=256, out_channels=64, kernel_size=3, stride=1
        x = self.conv2(x)
        # out_img_size=(14,14), out_channels=64
        x = self.relu(x)  # out_img_size=(14,14), out_channels=64
        # in_img_size=(14,14), out_channels=64, kernel_size=2, stride=2
        x = self.pool(x)
        # out_img_size=(7,7), out_channels=64

        # Serialization for 2D image * channels
        x = self.flatten(x)  # in_img_size=(7,7), in_channels=64
        # out_img_size=(3136,)

        # Fully connected layers
        x = self.linear1(x)  # in_features=3136, out_features=256
        x = self.relu(x)  # in_features=256, out_features=256

        # output layer
        x = self.linear2(x)  # in_features=256, out_features=10
        # x = self.softmax(x) #in_features=10, out_features=10
        return x


model = CnnNetwork().to('mps')
model.load_state_dict(torch.load('model.pth'))

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to('mps')
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
