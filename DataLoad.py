from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from PIL import Image
import os

# def preprocess():
    
#     folder_path_list = ["Data/mask", "Data/no_mask"]

#     for folder_path in folder_path_list:
#         png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
#         for png_file in png_files:
#             image = Image.open(os.path.join(folder_path, png_file))
#             if image.mode != "RGBA":
#                 image = image.convert("RGBA")
#             image.save(os.path.join(folder_path, png_file))

def create_augmentation_datasets(file_path):
    
    dataset = datasets.ImageFolder(root=file_path, 
                                    transform=transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(60),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
                              )
    return dataset

def split_datasets(dataset):

    train_ratio, val_ratio = [0.7, 0.15]

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, validation_dataset, test_dataset

def data_load(train_dataset, validation_dataset, test_dataset, batch_size):
    
    train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0
                         )

    val_loader = DataLoader(validation_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0
                         )

    test_loader = DataLoader(test_dataset, 
                         batch_size=batch_size,
                         shuffle=False, 
                         num_workers=0
                        )
    
    return train_loader, val_loader, test_loader