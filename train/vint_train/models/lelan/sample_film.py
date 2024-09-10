import torch
import torch.nn as nn
from torchvision import models

class FiLM(nn.Module):
    def __init__(self, in_channels, condition_dim):
        super(FiLM, self).__init__()
        self.gamma_fc = nn.Linear(condition_dim, in_channels)
        self.beta_fc = nn.Linear(condition_dim, in_channels)

    def forward(self, x, condition):
        gamma = self.gamma_fc(condition).unsqueeze(2).unsqueeze(3)
        beta = self.beta_fc(condition).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta
        
class FiLMResNet(models.ResNet):
    def __init__(self, condition_dim, *args, **kwargs):
        super(FiLMResNet, self).__init__(*args, **kwargs)
        self.condition_dim = condition_dim
        self.film_layers = nn.ModuleList([FiLM(self.inplanes, condition_dim)])
        
        for block in self.layer1:
            block.film = FiLM(block.conv1.out_channels, condition_dim)
        for block in self.layer2:
            block.film = FiLM(block.conv1.out_channels, condition_dim)
        for block in self.layer3:
            block.film = FiLM(block.conv1.out_channels, condition_dim)
        for block in self.layer4:
            block.film = FiLM(block.conv1.out_channels, condition_dim)

    def _forward_impl(self, x, condition):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.film_layers[0](x, condition)
        
        x = self.layer2(x)
        x = self.film_layers[1](x, condition)
        
        x = self.layer3(x)
        x = self.film_layers[2](x, condition)
        
        x = self.layer4(x)
        x = self.film_layers[3](x, condition)
        
        x = self.avgpool(x)
        xc = torch.flatten(x, 1)
        x = self.fc(x)

        return xc

    def forward(self, x, condition):
        return self._forward_impl(x, condition)        
      
def film_resnet18(condition_dim, pretrained=True):
    model = FiLMResNet(condition_dim, block=models.resnet.BasicBlock, layers=[2, 2, 2, 2])
    if pretrained:
        state_dict = models.resnet18(pretrained=True).state_dict()
        model.load_state_dict(state_dict, strict=False)
    return model

condition_dim = 10  # Example conditioning dimension
model = film_resnet18(condition_dim, pretrained=True)      
