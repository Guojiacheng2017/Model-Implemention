import torch.nn as nn

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=5,
                      stride=1,
                      padding=0),
            nn.Tanh(),  # original LeNet-5 use tanh() as activation function rather than ReLU()
            nn.MaxPool2d(kernel_size=2) # originally use averagepooling here we use maxpooling
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(400, 120) # 400 is based on 28 * 28 input image, if the img size is not known,
        # we can set 400 as imgsize*imgsize*out_channels/(4^2) -> because of maxpooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act = nn.Tanh()
        self.out = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.out(self.fc3(self.act(self.fc2(self.act(self.fc1(x))))))
        return output
