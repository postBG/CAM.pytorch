import torch
import torch.nn as nn

from dataset import CLASSES


class ConvNetWithGAP(nn.Module):
    def __init__(self):
        super(ConvNetWithGAP, self).__init__()
        # 3 * 128 * 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        # 32 * 128 * 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        # 64 * 128 * 128
        self.pool2 = nn.MaxPool2d(2, stride=2)

        # 64 * 64 * 64
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        # 128 * 64 * 64
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # 256 * 64 * 64
        self.pool4 = nn.MaxPool2d(2, stride=2)

        # 256 * 32 * 32
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, len(CLASSES), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(len(CLASSES)),
            nn.LeakyReLU(0.2)
        )

        # 10 * 32 * 32
        self.gap = nn.AvgPool2d(32)

        # 10 * 1 * 1
        self.fc = nn.Linear(10, 10)
        self.init_weights()

    def forward(self, images):
        out = self.conv1(images)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool4(out)
        out = self.conv5(out)
        out = self.gap(out)
        logits = self.fc(out.view(-1, 10))

        return logits

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_pretrained():
        model = ConvNetWithGAP()
        model.load_state_dict(torch.load("output/checkpoint.pth.tar",
                                         map_location="cuda" if torch.cuda.is_available() else "cpu"))
        return model
