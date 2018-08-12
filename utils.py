import torchvision
import matplotlib.pyplot as plt
import numpy as np


def reshape_for_removing_batch_size(tensor):
    batch_size = tensor.shape[0]
    if batch_size != 1:
        raise ValueError("batch_size should be 1")

    return tensor.view(tensor.shape[1:])


def min_max_normalize(tensor):
    tensor = tensor - tensor.min()
    return tensor / tensor.max()


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == '__main__':
    from dataset import get_cifar10_loader, CLASSES

    trainloader = get_cifar10_loader(batch_size=4)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % CLASSES[labels[j]] for j in range(4)))
