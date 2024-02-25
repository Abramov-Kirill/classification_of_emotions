import torch.nn as nn

class EmotionClassificationModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.count_class = conf.count_class
        self.count_layers = conf.count_layers
        self.ch = conf.ch

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

        self.blocks = []
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(1, self.ch, (3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25)
            ))
        for i in range(self.count_layers-1):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(self.ch * (2 ** i), self.ch * (2 ** (i +1)), (3, 3), stride=1, padding=1),
                    self.act,
                    self.pool,
                    self.dropout
                ))

        self.blocks = nn.Sequential(*self.blocks)

        self.out_chen = self.ch * 2 ** (self.count_layers - 1)
        self.out_x = conf.segment_size // conf.hop_length // 2 ** self.count_layers
        self.out_y = conf.n_mels // 2 ** self.count_layers

        self.in_linear = self.out_chen * self.out_x * self.out_y
        self.flat1 = nn.Flatten()
        self.lin1 = nn.Linear(self.in_linear, self.count_class)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        out = self.blocks(x)
        out = self.flat1(out)
        out = self.lin1(out)
        x = self.softmax(out)
        return x