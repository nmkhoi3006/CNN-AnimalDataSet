import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision.transforms import ToTensor, Resize, Compose


class AnimalDataSet(Dataset):
    def __init__(self, data_path, is_train=True, transformer=None):
        super().__init__()
        if is_train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "test")


        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse",
                    "sheep", "spider", "squirrel"]
        self.img = []
        self.label = []
        for indx, category in enumerate(self.categories):
            category_path = os.path.join(data_path, category)
            for item in os.listdir(category_path):
                img_path = os.path.join(category_path, item)
                self.img.append(img_path)
                self.label.append(indx)

        self.transformer = transformer

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img_path = self.img[item]
        label = self.label[item]
        img = cv2.imread(img_path)
        if self.transformer:
            img = self.transformer(img)
        return img, label

    def class_name(self, index):
        return self.categories[index]

    def num_classes(self):
        return len(self.categories)


if __name__ == '__main__':
    transforms = Compose([
        ToTensor(),
        Resize(size=(200, 200))
    ])

    train_data = AnimalDataSet("Data-Courses/animals", is_train=True, transformer=transforms)
    test_data = AnimalDataSet("Data-Courses/animals", is_train=False, transformer=transforms)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=32,
        shuffle = True,
        num_workers=4
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    print(train_data.num_classes())