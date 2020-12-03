import torch
import torch.nn as nn
from torchvision.models import densenet121
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch.optim import Adam
import torchvision.transforms as transforms


class DataSet(Dataset):
    def __init__(self):
        super().__init__()
        self.label = torch.from_numpy(np.load('label.dat', allow_pickle=True)).view(-1, 361).long()

    def __getitem__(self, item):
        trans = transforms.ToTensor()
        pic = Image.open('pic\\' + str(item) + '.jpg')
        return trans(pic), self.label[item]

    def __len__(self):
        return self.label.shape[0]


class F(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(1000, 19*19*3)

    def forward(self, x):
        return self.p(x).view(-1, 19*19, 3)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            densenet121(pretrained=True),
            F(),
            # nn.Softmax(2)
        )

    def forward(self, x):
        return self.sequential(x)


def train(epoch, load=None):
    if load is not None:
        dic = torch.load('discriminate_module\\dic_' + str(load) + '.pth')
        net.load_state_dict(dic['module'])
        opt.load_state_dict(dic['opt'])
    else:
        load = 0
    loss_f = torch.nn.CrossEntropyLoss()
    for e in range(load, load + epoch):
        for batch, (data_x, data_y) in enumerate(dataloader):
            predict = net(data_x.to('cuda:0'))
            loss = loss_f(predict.view(-1, 3), data_y.to('cuda:0').view(-1))
            loss.backward()
            opt.step()
            opt.zero_grad()
            print('epoch = ' + str(e) + ' batch = ' + str(batch) + ' loss = ' + str(loss.item()))
        dic = {
            'module': net.state_dict(),
            'opt': opt.state_dict()
        }
        torch.save(dic, 'discriminate_module\\dic_' + str(e) + '.pth')


if __name__ == '__main__':
    dataset = DataSet()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    net = Net().to('cuda:0')
    opt = Adam(net.parameters())
    train(50, load=None)

