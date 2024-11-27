class Model_AUROC():
    def __init__(self, learn_rate, gamma):
        self.network = ResNet18()
        self.network.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.network = self.network.cuda()
        self.loss_func = AUCMLoss()
        self.opt = PESG(self.network,
                        loss_fn=self.loss_func,
                        lr=learn_rate,
                        momentum=0.9,
                        margin=1.0,
                        epoch_decay=gamma,
                        weight_decay=0,
                        mode='adam')

class Model_AUPRC():
    def __init__(self, learn_rate, gamma, pos_len):
        self.network = ResNet18()
        self.network.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.network = self.network.cuda()
        self.loss_func = APLoss(pos_len=pos_len, margin=0.6, gamma=gamma)
        self.opt = SOAP(self.network.parameters(), lr=learn_rate, mode='adam', weight_decay=0.0)

class Model_Default():
    def __init__(self, learn_rate, gamma):
        self.network = ResNet18()
        self.network.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.network = self.network.cuda()
        self.loss_func = CrossEntropyLoss()
        self.opt = Adam(self.network.parameters(),
                        lr=learn_rate,
                        weight_decay=0)