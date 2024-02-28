import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import  models
from torchvision.transforms import v2 as T
from model import classifier, RoI_Head
from tqdm import tqdm
from dataset import dataset
import wandb

lr = 0.001
momentum = 0.9
input_size = [800, 1333]

wandb.init(
    # set the wandb project where this run will be logged
    project="classify",
    
    # track hyperparameters and run metadata
    config={
    "optimizer": "SGD", 
    "learning_rate": lr,
    "momentum": 0.9,
    "input_size": input_size,
    "architecture": "resnet34",
    "epochs": 10,
    }
)

# Define your data transformations
transform_train = T.Compose([
    T.Resize(input_size),
    T.RandomVerticalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = T.Compose([
    T.Resize(input_size),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def collate_fn(batch):
    return tuple(zip(*batch))

# 设置随机种子以确保可重复性
torch.manual_seed(0)

# Load your dataset
root = '/opt/data/private/zsf/Railway/part2/out_harmonization'
annFile = '/opt/data/private/zsf/Railway/part2/after_aug_(0, 3554).json'

train_dataset = dataset.CocoDatasetBbox(root, annFile, transform=transform_train)
# 计算训练集和测试集的大小
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
# 随机划分数据集
train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
test_dataset.transform = transform_test
# print(train_dataset) # {'anomaly': 0, 'normal': 1}

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
# print(train_loader)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Define the ResNet-34 model
backbone = models.resnet34(pretrained=True)
num_classes = 2
backbone = nn.Sequential(*list(backbone.children())[:-2])
backbone.out_channels = 512

# backbone = models.detection.backbone_utils.resnet_fpn_backbone(backbone_name='resnet34', weights=models.ResNet34_Weights.DEFAULT, trainable_layers=3)

roi_head = RoI_Head(None, None, None, out_channels=backbone.out_channels, num_classes=2)
model = classifier(backbone, roi_head)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs = torch.stack(inputs).to(device)
        
        optimizer.zero_grad()
        loss = model(inputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
    torch.save(model.state_dict(), f'{epoch}_res34_finetune.pth')
    
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
    wandb.log({"acc": accuracy * 100, "loss": running_loss / len(train_loader)})
    
torch.save(model.state_dict(), 'final_res34_finetune.pth')
# # Evaluation
# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)

#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)

#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = correct / total
# print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
