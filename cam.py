import os
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

from models import ConvNetWithGAP
from dataset import CLASSES, get_cifar10_loader, IMG_SIZE
from utils import reshape_for_removing_batch_size, min_max_normalize

TEST_JPG = 'result/test.jpg'

conv_activations = []


def store_activations(module, input, output):
    conv_activations.append(output)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNetWithGAP.get_pretrained().to(device)
model.conv5.register_forward_hook(store_activations)
fc_weights, fc_bias = list(model.fc.parameters())

loader = get_cifar10_loader(batch_size=1, train=False, num_workers=1)


def create_cam(export_root='result'):
    if not os.path.exists(export_root):
        os.mkdir(export_root)

    image_tensor, label = next(iter(loader))
    save_test_image(image_tensor)
    logits = model(image_tensor)
    predicted_idx = logits.max(1, keepdim=True)[1]
    cam = generate_cam(conv_activations.pop(), fc_weights, predicted_idx)
    test_img = cv2.imread(TEST_JPG)
    height, width, _ = test_img.shape
    heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + test_img * 0.5
    cv2.imwrite('result/{}_cam.jpg'.format(CLASSES[label]), result)


def save_test_image(image_tensor):
    denormalize = transforms.Compose([
        transforms.Normalize([0, 0, 0], [2., 2., 2.]),
        transforms.Normalize([-0.5, -0.5, -0.5], [1., 1., 1.])
    ])
    transform = transforms.Compose([
        denormalize, transforms.ToPILImage()
    ])
    pil_image = transform(image_tensor[0])
    pil_image.save(TEST_JPG)


def generate_cam(activation, fc_weights, class_idx):
    activation = reshape_for_removing_batch_size(activation)
    class_idx = reshape_for_removing_batch_size(class_idx)

    channel, height, width = activation.shape
    cam = torch.zeros([height, width], dtype=torch.float)
    fc_weight = fc_weights[class_idx]
    fc_weight = fc_weight.view(fc_weight.shape[1])
    for c in range(channel):
        cam += fc_weight[c] * activation[c]

    normalized_cam = min_max_normalize(cam)
    scaled_cam = np.uint8((normalized_cam * 255).int().numpy())
    return cv2.resize(scaled_cam, (IMG_SIZE, IMG_SIZE))
