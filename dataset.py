from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_ROOT = './data'
CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
default_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_cifar10_loader(batch_size=64, train=True, num_workers=2):
    dataset = datasets.CIFAR10(root=DATASET_ROOT, train=train, transform=default_transform)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)