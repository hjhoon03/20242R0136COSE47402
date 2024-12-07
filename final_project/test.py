import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import convnext_tiny

import matplotlib.pyplot as plt

def test_model(model, dataloader, classes):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy of the model on selected classes {classes}: {accuracy:.2f}%")

    return accuracy


selected_classes = [0, 1]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

selected_indices = [i for i, (_, label) in enumerate(test_dataset) if label in selected_classes]
test_dataset_filtered = Subset(test_dataset, selected_indices)

test_loader = DataLoader(test_dataset_filtered, batch_size=64, shuffle=False)

num_classes = (2, 5, 10)
resnets = [None] * len(num_classes)
for i, n in enumerate(num_classes):
    resnets[i] = models.resnet18(num_classes=n)
    state_dict = torch.load(f"./models/fine_tuned_resnet_{n}_classes.pth", weights_only=True)
    resnets[i].load_state_dict(state_dict)
    resnets[i].eval()

convnext_model = convnext_tiny(pretrained=True)
convnext_model.classifier[2] = nn.Linear(convnext_model.classifier[2].in_features, len(selected_classes))
convnext_model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

results = [None] * (len(num_classes) + 1)

for i, n in enumerate(num_classes):
    resnets[i].to(device)
    print(f"Testing Fine-tuned resnet ({n} classes) Model:")
    results[i] = test_model(resnets[i], test_loader, selected_classes)

print("Testing ConvNeXT Model:")
convnext_model.to(device)
results[-1] = test_model(convnext_model, test_loader, selected_classes)

model_names = [f"ResNet({n})" for n in num_classes] + ["ConvNeXT"]

plt.title("Test Accuracy")
plt.bar(model_names, results, color=['red', 'blue', 'green', 'purple'], edgecolor='black', hatch=['/', '\\', '|', '-'])
plt.xlabel("Models")

plt.savefig(f'./graphs/test_accuracy.pdf')
