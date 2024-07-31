from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Sequential, Linear, Sigmoid, AvgPool2d,Flatten, LeakyReLU, ModuleList, AdaptiveAvgPool2d

# Implement the ResNet class
class ResNet(Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.layers = [
            Conv2d(3, 64, 7, 2),
            BatchNorm2d(64),
            LeakyReLU(),
            MaxPool2d(3, 2),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 512, 2),
            AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            Linear(512, 2),
            Sigmoid(),
            ]

        self.layers = ModuleList(self.layers)

    def forward(self, input_tensor):
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        return x
    

class ResidualBlock(Module):
    '''
    IMplementation of residual block
    '''
    def __init__(self, input_channels, output_channels, stride):
        # invoking super constructor
        super().__init__()
        self.identity_downsample = Sequential(
            Conv2d(input_channels, output_channels, kernel_size=1, stride=stride),
            BatchNorm2d(output_channels))
        
        self.layers =[
            Conv2d(input_channels, output_channels, kernel_size = 3, stride = stride, padding = 1),
            BatchNorm2d(output_channels),
            LeakyReLU(),
            Conv2d(output_channels, output_channels, kernel_size = 3, padding = 1),
            BatchNorm2d(output_channels),
            LeakyReLU()]

        self.layers = ModuleList(self.layers)

    def forward(self, input_tensor):
        x = input_tensor
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i == 4:
                input_tensor = self.identity_downsample(input_tensor)
                x += input_tensor
        return x

