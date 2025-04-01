import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import argparse
import os, time, copy

parser = argparse.ArgumentParser(description='COVID-19 Detection using DenseNet-121')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--dataset_path', type=str, default='/content/drive/MyDrive/ColabNotebooks/data/', help='Dataset directory')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading') 
args = parser.parse_args()

# Data Transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = args.dataset_path
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True) for x in ['train', 'val']}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DenseNet-121 Model
model_conv = models.densenet121(pretrained=True)
num_ftrs = model_conv.classifier.in_features
model_conv.classifier = nn.Linear(num_ftrs, 2)  # 2 classes: COVID, Non-COVID
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=args.learning_rate, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# Training Function
def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f'{phase} Accuracy: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts)
    return model

# saving the Model
if __name__ == "__main__":
    model_conv = train_model(model_conv, criterion, optimizer_conv, scheduler, args.epochs)
    torch.save(model_conv, '/content/drive/MyDrive/ColabNotebooks/covid_denseNet_epoch%d.pt'%args.epochs)
