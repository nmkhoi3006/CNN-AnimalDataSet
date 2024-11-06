from AnimalData import AnimalDataSet
from AnimalModel import CNNModel
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.metrics import accuracy_score
from tqdm.autonotebook import tqdm

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
        num_workers=2,
        drop_last=False
    )
    dataloader_test = DataLoader(
        dataset=data_test,
        batch_size=32,
        num_workers=2,
        drop_last=False
    )

    model = CNNModel(num_class=data_train.num_classes())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2)
    num_epochs = 100
    num_iter = int(len(data_train)/32)
    for epoch in range(1, num_epochs+1):
        #TRAIN
        progress_bar = tqdm(dataloader_train, colour="blue")
        print("TRAIN")
        model.train()
        for iter, (img, label) in enumerate(progress_bar):
            prediction = model(img)
            loss = criterion(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch: {epoch}/{num_epochs} | Loss: {loss:.4f}")

        #VAL
        all_losses = []
        all_labels = []
        all_predictions = []
        print("VALIDATION")
        model.eval()
        with torch.inference_mode():
            for img, label in (dataloader_test):
                prediction = model(img)
                _, index = torch.max(prediction, dim=1)
                all_predictions.extend(index.tolist())
                all_labels.extend(label.tolist())

                loss = criterion(index, label)
                all_losses.append(loss)

        acc = accuracy_score(all_labels, all_predictions)
        loss = sum(all_losses)/len(all_losses)
        print(f"Accuracy_Validation: {acc} | Loss_Validation{loss}")




if __name__ == '__main__':
    train()
