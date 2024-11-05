from AnimalData import AnimalDataSet
from AnimalModel import CNNModel
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize

def train():
    transforms = Compose([
        ToTensor(),
        Resize(size=(32, 32))
    ])
    data_train = AnimalDataSet(data_path="Data-Courses/animals", is_train=True, transformer=transforms)
    data_test = AnimalDataSet(data_path="Data-Courses/animals", is_train=False, transformer=transforms)


    dataloader_train = DataLoader(
        dataset=data_train,
        batch_size=32,
        num_workers=4,
        drop_last=False
    )
    dataloader_test = DataLoader(
        dataset=data_test,
        batch_size=32,
        num_workers=4,
        drop_last=False
    )

    model = CNNModel(num_class=data_train.num_classes())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2)
    num_epochs = 100
    num_iter = int(len(data_train)/32)
    for epoch in range(1, num_epochs+1):
        model.train()
        for iter, (img, label) in enumerate(dataloader_train):
            prediction = model(img)
            loss = criterion(prediction, label)

            print(f"Iter: {iter}/{num_iter} | Loss: {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()






if __name__ == '__main__':
    train()
