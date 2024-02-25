import torch.nn as nn


class Block1(nn.Module):
    def __init__(self, conf, input_ch, out_ch):
        super().__init__()
        self.count_layers_in_group = conf.count_layers_in_group

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(conf.dropout)

        assert self.count_layers_in_group >= 2, "The number of layers in the yt block may be less than 2"

        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Conv2d(input_ch, out_ch, (3, 3), stride=1, padding=1),
            self.act,
            self.dropout
        ))
        for i in range(self.count_layers_in_group - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, (3,3), stride=1, padding=1),
                self.act,
                self.dropout
            ))
        self.layers = nn.Sequential(*self.layers)
    def forward(self, x):
        out = self.layers(x)
        out = self.pool(out)
        return out


class EmotionClassificationModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.ch = conf.ch
        self.count_blocks = conf.count_blocks
        self.count_class = conf.count_class

        self.blocks = []
        self.blocks.append(Block1(conf, 1, self.ch))

        for i in range(self.count_blocks - 1):
            self.blocks.append(Block1(conf, self.ch * 2 ** i, self.ch * 2 ** (i+1)))

        self.blocks = nn.Sequential(*self.blocks)

        self.out_chen = self.ch * 2 ** (self.count_blocks - 1)
        self.out_x = conf.segment_size // conf.hop_length // 2 ** self.count_blocks
        self.out_y = conf.n_mels // 2 ** self.count_blocks

        self.flat1 = nn.Flatten()
        self.lin1 = nn.Linear(self.out_chen * self.out_x * self.out_y, self.count_class)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        out = self.blocks(x)
        out = self.flat1(out)
        out = self.lin1(out)
        out = self.softmax(out)
        return out
