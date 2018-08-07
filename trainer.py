import shutil

import torch
import torch.nn.functional as F
import torch.optim as optim


def save_checkpoint(state, is_best=False, filename='output/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class Trainer(object):
    def __init__(self, model, device, loader_factory, epoch=100, lr=0.0001, batch_size=64):
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size

        self.loader_factory = loader_factory

        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.min_loss = 999.

    def train_one_epoch(self, epoch):
        self.model.train()

        train_data_loader = self.loader_factory(batch_size=self.batch_size, train=True)
        for batch_idx, (images, labels) in enumerate(train_data_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            self.optimizer.step()

            if batch_idx % 300 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(images),
                    len(train_data_loader.dataset),
                    100. * batch_idx / len(train_data_loader.dataset),
                    loss.item()))

        if epoch % 5 == 0:
            loss = self.test_model()
            if loss < self.min_loss:
                self.min_loss = loss
                save_checkpoint(self.model.state_dict(), is_best=True)

    def test_model(self):
        test_data_loader = self.loader_factory(batch_size=self.batch_size, train=False)

        with torch.no_grad():
            self.model.eval()
            test_loss = 0
            correct = 0
            for images, labels in test_data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                test_loss += F.cross_entropy(logits, labels)
                pred = logits.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()

            test_loss /= len(test_data_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss,
                correct,
                len(test_data_loader.dataset),
                100. * correct / len(test_data_loader.dataset)))

            return test_loss
