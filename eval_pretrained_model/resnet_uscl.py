import torch.nn as nn
import torchvision.models as models

class ResNetUSCL(nn.Module):
    ''' The ResNet feature extractor + projection head + classifier for USCL '''

    def __init__(self, base_model, out_dim, pretrained=False):
        super(ResNetUSCL, self).__init__()

        self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained),
                            "resnet50": models.resnet50(pretrained=pretrained)}
        if pretrained:
            print('\nModel parameters loaded.\n')
        else:
            print('\nRandom initialize model parameters.\n')
        
        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # discard the last fc layer

        # projection MLP
        self.linear = nn.Linear(num_ftrs, out_dim)

        # classifier
        num_classes = 12
        self.fc = nn.Linear(out_dim, num_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.linear(h)

        return x
