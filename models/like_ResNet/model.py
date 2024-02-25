import torch.nn as nn
import numpy as np
class ResBlock1(nn.Module):
    def __init__(self, count_ch):
        super().__init__()
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.drop_out = nn.Dropout(0.35)
        self.batch_norm1 = nn.BatchNorm2d(count_ch)
        self.conv1 = nn.Conv2d(in_channels=count_ch, out_channels=count_ch, kernel_size=(3, 3), padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=count_ch, out_channels=count_ch, kernel_size=(3, 3), padding=1, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        #out = self.batch_norm1(out)
        out = self.drop_out(out)
        out = self.act(out)
        out = self.conv2(out)
        #out = self.batch_norm1(out)
        out = self.drop_out(out)
        return self.act(x + out)

class TransitionBlock1(nn.Module):
    def __init__(self, count_ch):
        super().__init__()
        self.act = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(count_ch)
        self.drop_out = nn.Dropout(0.35)
        self.conv1 = nn.Conv2d(in_channels=int(count_ch / 2), out_channels=count_ch, kernel_size=(3, 3), padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=count_ch, out_channels=count_ch, kernel_size=(3, 3), padding=1, stride=1)
        self.transition_conv = nn.Conv2d(in_channels=int(count_ch / 2) , out_channels=count_ch, kernel_size=(1,1), padding=0, stride=2)
    def forward(self, x):
        out = self.conv1(x)
        #out = self.batch_norm1(out)
        out = self.drop_out(out)
        out = self.act(out)
        out = self.conv2(out)
        #out = self.batch_norm1(out)
        out = self.drop_out(out)
        return self.act(self.transition_conv(x) + out)

class TruckBlock(nn.Module):
    def __init__(self, type_block, type_transition_block, count_blocks, count_ch, use_transition_block = False):
        super().__init__()
        self.truck = []

        if use_transition_block:
            self.truck += [globals()[type_transition_block](count_ch)]
            count_blocks -= 1

        for i in range(count_blocks):
            self.truck += [globals()[type_block](count_ch)]
        self.truck = nn.Sequential(*self.truck).to('cuda')
    def forward(self, x):
        return self.truck(x)

class EmotionClassificationResNet(nn.Module):
    def __init__(self, num_classes, conf):
        super().__init__()
        self.conf = conf
        self.conv0 = nn.Conv2d(1, conf.ch, kernel_size=(3, 3), padding=1, stride=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.groups_block = []
        self.groups_block.append(
        TruckBlock(conf.type_block, conf.type_transition_block, conf.count_block_in_group, conf.ch * (2 ** 0)))
        for i in range(1, conf.count_group_block):
            self.groups_block.append(TruckBlock(
                conf.type_block, conf.type_transition_block, conf.count_block_in_group, conf.ch * (2 ** i), use_transition_block=True)
            )

        self.groups_block = nn.Sequential(*self.groups_block)
        self.flat = nn.Flatten()
        #self.chen = conf.ch * 2 ** (conf.count_group_block-1)
        #self.on_y = np.around(conf.n_mels / 2 ** conf.count_group_block)
        #self.on_x = np.around(np.around(conf.segment_size / conf.hop_length) / 2 ** conf.count_group_block)
        in_linear =  int(conf.ch * 2 ** (conf.count_group_block-1) *
                      np.around(conf.n_mels / 2 ** conf.count_group_block) *
                      np.around(np.around(conf.segment_size / conf.hop_length) / 2 ** conf.count_group_block))
        self.line = nn.Linear(in_linear, num_classes)
        self.soft_max = nn.Softmax()
    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)
        out = self.pool(out)
        for i in self.groups_block:
            out = i(out)
        out = self.flat(out)
        out = self.line(out)
        return self.soft_max(out)

