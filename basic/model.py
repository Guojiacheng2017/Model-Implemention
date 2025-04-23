import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output    # return x for visualization

class ConvNet(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)  # nn.Sequential()
        self.relu = nn.ReLU()
        self.padding = nn.ZeroPad2d(1)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 4)
        self.flat = nn.Flatten(1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        covlayer1 = self.max_pool(self.relu(self.conv1(self.padding(x))))  # size * size -> size/2 * size/2
        covlayer2 = self.max_pool(self.relu(self.conv2(self.padding(covlayer1))))  # size/2 * size/2 -> size/4 * size/4
        #         x = torch.flatten(covlayer2, 1)
        #         covlayer2.reshape(covlayer2.shape[0], -1)
        covlayer2 = self.flat(covlayer2)
        #         print(covlayer2.shape)
        x = self.relu(self.fc1(covlayer2))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
            nn.MaxPool2d(kernel_size=2, stride=2) # originally use averagepooling here we use maxpooling
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,   # 16 => 120 since layer three is not useful in MNIST case
                      kernel_size=5,
                      stride=1,
                      padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=120,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(480, 84) # 400 is based on 28 * 28 input image, if the img size is not known,
        # we can set 400 as imgsize*imgsize*out_channels/(4^2) -> because of maxpooling
        self.fc2 = nn.Linear(84, 10)
        self.act = nn.Tanh()
        # self.out = nn.Softmax()

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        # x = self.act(self.fc1(x))
        # x = self.act(self.fc2(x))
        # rst = self.out(self.fc3(x))
        # print(x.shape)
        rst = self.fc2(self.act(self.fc1(x)))
        return rst


# _____________________ ResNet _____________________ #

class Basicblock(nn.Module):
    expansion: int = 1
    def __init__(self, in_planes, planes, stride):
        super(Basicblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != planes:
            # if the stride != 1, that means the image will downsample and get smaller image, which we cannot cancanate original image directly.
            # similar to the in_planes, that means the
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        out = self.block(x) + self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, in_planes, planes, stride):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=planes, out_channels=(self.expansion * planes), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d((self.expansion * planes)),
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != (self.expansion * planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=(self.expansion * planes), kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d((self.expansion * planes)),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        # print(self.expansion)
        # print("bottleneck: ", self.bottleneck(x).shape)
        # print("shortcut: ", self.shortcut(x).shape)
        out = self.bottleneck(x) + self.shortcut(x)
        # print(out.shape)
        out = self.relu(out)
        return out

# not include any dilation (暂不考虑扩张卷积)
class ResNet(nn.Module):
    def __init__(self, block, in_channel, num_blocks, num_classes):
        super(ResNet, self).__init__()
        # self.in_channel = in_channel    # used in function makelayers
        self.in_plane = 64
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=self.in_plane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # ==> (64 * 64) * 64 channels

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # ==> (64 * 64) * 64 channels
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # ==> (32 * 32) * 128 channels
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # ==> (16 * 16) * 256 channels
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # ==> (8 * 8) * 512 channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # feature size is normed in (1, 1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)  # (feature size ^ 2 * 512 channels) x (categories)

        # if type(block) == Basicblock:
        #     print('Basic Block')
        #     self.layer1 = self._make_layer(Basicblock, 64, num_blocks[0], stride=1)      # ==> (64 * 64) * 64 channels
        #     self.layer2 = self._make_layer(Basicblock, 128, num_blocks[1], stride=2)     # ==> (32 * 32) * 128 channels
        #     self.layer3 = self._make_layer(Basicblock, 256, num_blocks[2], stride=2)     # ==> (16 * 16) * 256 channels
        #     self.layer4 = self._make_layer(Basicblock, 512, num_blocks[3], stride=2)     # ==> (8 * 8) * 512 channels
        # elif type(block) == Bottleneck:
        #     print('Bottleneck')
        #     self.layer1 = self._make_layer(Bottleneck, 64, num_blocks[0], stride=1)      # ==> (64 * 64) * 256 channels
        #     self.layer2 = self._make_layer(Bottleneck, 128, num_blocks[1], stride=2)     # ==> (32 * 32) * 512 channels
        #     self.layer3 = self._make_layer(Bottleneck, 256, num_blocks[2], stride=2)     # ==> (16 * 16) * 1024 channels
        #     self.layer4 = self._make_layer(Bottleneck, 512, num_blocks[3], stride=2)     # ==> (8 * 8) * 2048 channels
        # the reason that channels is 64 after first layer is stride is 1, layer2 or 3 or 4 stride is 2

    def _make_layer(self, block, plane, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)   # list all stride for each layer
        layers = []
        for stride in strides:
            layers.append(block(self.in_plane, plane, stride))
            self.in_plane = plane * block.expansion
        return nn.Sequential(*layers)   # why we use the address of the list?

    def forward(self, x):
        # print(x.shape)
        out = self.maxpool(self.relu(self.bn(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out







# _____________________ Generative Adversarial Nets for MNIST _____________________ #

class Generator_MNIST(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator_MNIST, self).__init__()
        self.img_shape = img_shape

        def block(in_channel, out_channel, normalize=True):
            layers = [nn.Linear(in_channel, out_channel)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_channel, 0.8))  # why BatchNorm1d? not BatchNorm2d?
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        # print(type(self.img_shape), *self.img_shape, img.size(0))
        img = img.reshape(img.shape[0], *self.img_shape)  # 64 * 1 * 28 * 28
        return img


class Discriminator_MNIST(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator_MNIST, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape), 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), # inplace = True => saves memory during both training and testing
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.reshape(img.shape[0], -1)
        out = self.model(img_flat)
        return out


# _____________________ Generative Adversarial Nets _____________________ #

class Generator(nn.Module):
    def __init__(self, latent_size, gen_fea, channels, num_gpu):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(latent_size, gen_fea * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_fea * 8),
            nn.ReLU(inplace=True),
            # state size, (gen_fea * 8) x 4 x 4
            nn.ConvTranspose2d(gen_fea*8, gen_fea*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_fea * 4),
            nn.ReLU(inplace=True),
            # state size, (gen_fea * 4) x 8 x 8
            nn.ConvTranspose2d(gen_fea*4, gen_fea*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_fea * 2),
            nn.ReLU(inplace=True),
            # state size, (gen_fea * 2) x 16 x 16
            nn.ConvTranspose2d(gen_fea*2, gen_fea, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_fea),
            nn.ReLU(True),
            # state size, (gen_fea) x 32 x 32
            nn.ConvTranspose2d(gen_fea, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size, (channels) x 64 x 64
        )

    def forward(self, input):
        return self.backbone(input)


class Discriminator(nn.Module):
    def __init__(self, gen_fea, channels, num_gpu): # latent_size is not using in this model
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.backbone = nn.Sequential(
            # input image size, channels x 64 x 64
            nn.Conv2d(channels, gen_fea, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size, (gen_fea) x 32 x 32
            nn.Conv2d(gen_fea, gen_fea*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size, (gen_fea*2) x 16 x 16
            nn.Conv2d(gen_fea*2, gen_fea*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size, (gen_fea*4) x 8 x 8
            nn.Conv2d(gen_fea*4, gen_fea*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size, (gen_fea*8) x 4 x 4
            nn.Conv2d(gen_fea*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.backbone(input)


# _____________________ U-net _____________________ #






# ===================== 2024-08-14 ===================== #

# --------------------- Faster-RCNN --------------------- #

class RPN(nn.Module):
    