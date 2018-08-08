from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
default_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_cifar10_loader(batch_size=64, train=True, num_workers=2, data_root='./data'):
    dataset = datasets.CIFAR10(root=data_root, train=train, transform=default_transform, download=True)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
