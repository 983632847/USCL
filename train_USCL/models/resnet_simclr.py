import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .model_resnet import resnet18_cbam, resnet50_cbam


class ResNetSimCLR(nn.Module):
    ''' The ResNet feature extractor + projection head for SimCLR '''

    def __init__(self, base_model, out_dim, pretrained=False):
        super(ResNetSimCLR, self).__init__()

        use_CBAM = False  # # use CBAM or not, att_type="CBAM" or None
        if use_CBAM:
            self.resnet_dict = {"resnet18": resnet18_cbam(pretrained=pretrained),
                                "resnet50": resnet50_cbam(pretrained=pretrained)}
        else:
            self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained),
                                "resnet50": models.resnet50(pretrained=pretrained)}


        if pretrained:
            print('\nImageNet pretrained parameters loaded.\n')
        else:
            print('\nRandom initialize model parameters.\n')
        
        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1]) # discard the last fc layer

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        #########################################################
        num_classes = 2
        self.fc = nn.Linear(out_dim, num_classes)

        ## Mixup
        self.fc1 = nn.Linear(num_ftrs, num_ftrs)
        self.fc2 = nn.Linear(num_ftrs, num_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    # def forward(self, x):
    #     h = self.features(x)
    #
    #     h = h.squeeze()
    #
    #     x = self.l1(h)
    #     x = F.relu(x)
    #     x = self.l2(x)
    #     return h, x # the feature vector, the output

    def forward(self, x):
        h = self.features(x)
        h1 = h.squeeze()  # feature before project g()=h1

        x = self.l1(h1)
        x = F.relu(x)
        x = self.l2(x)

        # 1.classification: feature is before project g()
        # c = h1
        # # c = self.avgpool(c)
        # c = c.view(c.size(0), -1)
        # c = self.fc1(c)
        # c = F.relu(c)
        # c = self.fc2(c)

        ## 2.classification: feature is the output of project g()
        c = x
        c = c.view(c.size(0), -1)
        c = self.fc(c)
        return h1, x, c # the feature vector, the output
