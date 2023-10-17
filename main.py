
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms
# import torchvision.transforms as transforms

# root_path = 'Data'

# '''Data Augmentation'''

# dataset = datasets.ImageFolder(root=root_path, 
#                                     transform=transforms.Compose([
#                                         transforms.Resize((256, 256)),
#                                         transforms.RandomHorizontalFlip(),
#                                         transforms.RandomVerticalFlip(),
#                                         transforms.RandomRotation(60),
#                                         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#                                     ])
#                               )

# classes = dataset.classes

# print(dataset.classes)
# print(dataset.class_to_idx)

# dataset_size = len(dataset)
# train_size = int(dataset_size * 0.7)
# validation_size = int(dataset_size * 0.15)
# test_size = dataset_size - train_size - validation_size

# train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

# print(f"Training Data Size : {len(train_dataset)}")
# print(f"Validation Data Size : {len(validation_dataset)}")
# print(f"Testing Data Size : {len(test_dataset)}")

# # Hyperparameters
# input_size = 256 * 256 * 3
# num_classes = 2
# batch_size = 64
# learning_rate = 1e-3
# epochs = 10

# train_loader = DataLoader(train_dataset, 
#                           batch_size=batch_size,
#                           shuffle=True, 
#                           num_workers=0
#                          )


# val_loader = DataLoader(validation_dataset, 
#                           batch_size=batch_size,
#                           shuffle=True, 
#                           num_workers=0
#                          )

# test_loader = DataLoader(test_dataset, 
#                          batch_size=batch_size,
#                          shuffle=False, 
#                          num_workers=0
#                         )

# class SimpleFCNN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(SimpleFCNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         print(x.shape)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x

# model = SimpleFCNN(input_size, num_classes).cuda()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# from tqdm import tqdm
# import copy

# for epoch in tqdm((range(epochs))):
#     model.train()
#     running_loss = 0.0
    
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
    
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     cur_accuracy = 100 * correct / total
    
#     print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Val Accuracy: {cur_accuracy:.2f}%')
    
#     if epoch == 0:
#         best_model = copy.deepcopy(model)
#         best_accuracy = cur_accuracy

#     if cur_accuracy > best_accuracy:
#         best_model = copy.deepcopy(model)

# torch.save(best_model.state_dict(), 'fully_connected_model.pth')

# best_model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for inputs, labels in test_loader: 
#         outputs = best_model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# test_accuracy = 100 * correct / total
# print(f'Test accuracy: {test_accuracy:.2f}%')

import torch
import torch.nn as nn
import torch.optim as optim
import DataLoad as DL
from Model import SimpleFCNN

root_path = 'Data'

import argparse

import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('AI_Midterm ', add_help=False)
    
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=64, type=int )
    parser.add_argument('--epochs', default=15,type=int)
    
    return parser

img_size = 256*256*3

parser = argparse.ArgumentParser('AI_Midterm ', parents=[get_args_parser()])
args = parser.parse_args(args=[])

wandb.init(project='23_2_AI_Midterm_mask_nomask')
wandb.config.update(args)

num_classes = 2

model = SimpleFCNN(img_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

train_dataset, validation_dataset, test_dataset = DL.split_datasets(DL.create_augmentation_datasets(root_path))
train_loader, val_loader, test_loader = DL.data_load(train_dataset, validation_dataset, test_dataset, args.batch_size)

model=model.cuda()
epochs = args.epochs

from tqdm.auto import tqdm
import copy

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    wandb.log({"Trian Loss": running_loss / len(train_loader)})
              
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    cur_accuracy = 100 * correct / total
    
    print(f'Epoch {epoch + 1}, Trian Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Val Accuracy: {cur_accuracy:.2f}%')
    
    if epoch == 0:
        best_model = copy.deepcopy(model)
        best_accuracy = cur_accuracy

    if cur_accuracy > best_accuracy:
        best_model = copy.deepcopy(model)

torch.save(best_model.state_dict(), 'fully_connected_model.pth')

best_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
     
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test accuracy: {test_accuracy:.2f}%')
