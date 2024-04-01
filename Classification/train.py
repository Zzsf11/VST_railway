import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import  models
from torchvision.transforms import v2 as T
from model import classifier, RoI_Head
from tqdm import tqdm
from dataset import dataset
import wandb
import random

def does_overlap(new_box, existing_boxes):
    new_x1, new_y1, new_x2, new_y2 = new_box
    for existing_box in existing_boxes:
        ex_x1, ex_y1, ex_x2, ex_y2 = existing_box
        # Check if there is an overlap
        if not (new_x2 <= ex_x1 or new_x1 >= ex_x2 or new_y2 <= ex_y1 or new_y1 >= ex_y2):
            return True
    return False

def generate_boxes(shape, area, aspect_ratio, existing_boxes):
    height, width = shape
    boxes = []

    # Calculate box dimensions based on the specified area and aspect ratio
    box_height = int((area / aspect_ratio) ** 0.5)
    box_width = int(box_height * aspect_ratio)

    while len(boxes) != 2:
        # Randomly position the new box within the available space
        x1 = random.randint(0, width - box_width)
        y1 = random.randint(0, height - box_height)
        x2 = x1 + box_width
        y2 = y1 + box_height
        new_box = (x1, y1, x2, y2)

        # Check if the new box overlaps with any existing box
        if not does_overlap(new_box, existing_boxes):
            boxes.append(new_box)
            existing_boxes.append(new_box)

    return boxes

if __name__ == "__main__":
    lr = 0.0001
    momentum = 0.9
    input_size = [800, 1333]

    wandb.init(
        # set the wandb project where this run will be logged
        project="classify",
        name="classifier_fpn_pretrain",
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
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Define the ResNet-34 model
    backbone = models.resnet34(pretrained=True)
    # backbone = models.resnet34(pretrained=False)
    num_classes = 2
    # backbone = nn.Sequential(*list(backbone.children())[:-2])
    # backbone.out_channels = 512

    backbone = models.detection.backbone_utils.resnet_fpn_backbone(backbone_name='resnet34', weights=models.ResNet34_Weights.DEFAULT, trainable_layers=3)

    roi_head = RoI_Head(None, None, None, out_channels=backbone.out_channels, num_classes=2)
    model = classifier(backbone, roi_head)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # pth_path = '/opt/data/private/zsf/VST_railway/Classification/checkpoints/with_pretrain/final_res34_finetune.pth'
    # state_dict = torch.load(pth_path)
    # model.load_state_dict(state_dict)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.mode= 'train'
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
        model.mode= 'test'
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs = torch.stack(inputs).to(device)
                bboxes = [ann['bbox'] for ann in labels[0]]
                bboxes_neg = generate_boxes(inputs.shape[-2:],2000,1, bboxes)
                bboxes_neg = bboxes_neg + generate_boxes(inputs.shape[-2:],1000,1, bboxes)
                
                # 合并 bboxes 和 bboxes_neg
                merged_bboxes = bboxes + bboxes_neg

                # 创建标签列表，bboxes 的位置为 1，bboxes_neg 的位置为 0
                labels = [1] * len(bboxes) + [0] * len(bboxes_neg)
                
                outputs = model(inputs, merged_bboxes)
                class_preds = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                total += len(labels)
                labels_tensor = torch.tensor(labels).to(device)
                correct += (predicted == labels_tensor).sum().item()

            accuracy = correct / total
        wandb.log({"acc": accuracy * 100, "loss": running_loss / len(train_loader)})
        print("acc:",  accuracy * 100)
        
    torch.save(model.state_dict(), 'final_res34_finetune.pth')



