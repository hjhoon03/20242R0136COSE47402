import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt

class FilteredCIFAR10(Dataset):
    def __init__(self, original_dataset, selected_classes):
        self.selected_classes = selected_classes
        self.dataset = original_dataset
        self.indices = [
            i for i, (_, label) in enumerate(self.dataset) if label in self.selected_classes
        ]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, label = self.dataset[original_idx]
        # 레이블을 새롭게 정렬 (예: 0, 1, 2로 매핑)
        new_label = self.selected_classes.index(label)
        return image, new_label
    

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 통계
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 통계
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def model_initializing(num_classes):
    model = models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features  # 기존 fully connected layer의 입력 크기
    model.fc = nn.Linear(num_ftrs, num_classes)  # CIFAR-10 클래스 개수로 수정

    return model


def load_dataset(selected_classes):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    original_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    original_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_dataset = FilteredCIFAR10(original_train_dataset, selected_classes)
    test_dataset = FilteredCIFAR10(original_test_dataset, selected_classes)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    return train_loader, test_loader


if __name__ == '__main__':
    num_classes = (2, 5, 10)
    resnets = [None] * len(num_classes)
    epochs = 5
    results = [[None] * epochs for _ in range(len(num_classes))]

    for i, n in enumerate(num_classes):
        resnets[i] = model_initializing(n)
        train_loader, test_loader = load_dataset(range(n))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(resnets[i].parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        resnets[i] = resnets[i].to(device)

        for epoch in range(epochs):
            train_loss, train_acc = train(resnets[i], train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(resnets[i], test_loader, criterion, device)

            results[i][epoch] = (train_loss, train_acc, val_loss, val_acc)

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        torch.save(resnets[i].state_dict(), f'./models/fine_tuned_resnet_{n}_classes.pth')

    titles = ["Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"]

    for i, t in enumerate(titles):
        x_values = list(range(1, epochs + 1))
        plt.plot(x_values, list(map(lambda x: x[i], results[0])), linestyle='-', color='blue', label=f'{num_classes[0]}_classes', marker='o')
        plt.plot(x_values, list(map(lambda x: x[i], results[1])), linestyle='--', color='orange', label=f'{num_classes[1]}_classes', marker='s')
        plt.plot(x_values, list(map(lambda x: x[i], results[2])), linestyle='-.', color='green', label=f'{num_classes[2]}_classes', marker='^')

        plt.title(t)
        plt.xticks(ticks=x_values)
        plt.xlabel('num_epochs')
        plt.legend()

        plt.savefig(f'./graphs/{t}.pdf')
        plt.close()
    