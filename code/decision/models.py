import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import vgg16, vgg19
from torch.nn import init


class ResNetModel(nn.Module):

    def __init__(self, pretrained=False):

        super(ResNetModel, self).__init__()

        self.model = resnet18(pretrained)
        self.hidden_features = 1024
        self.out_features = 128
        self.dropout_rate = 0.7

        # fc layer fine tuning
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, self.hidden_features),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_features, self.out_features)
        )

        # self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        # fc layer initialization
        '''
        if pretrained:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    # m.weight.data.normal_()
                    nn.init.xavier_normal_(m.weight)
                    # m.weight.data.normal_()
                    m.bias.data.zero_()

            for m in model.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.normal(m.weight.data)
                    nn.init.xavier_normal(m.weight.data)
                    nn.init.kaiming_normal(m.weight.data)
                    m.bias.data.fill_(0)
                elif isinstance(m,nn.Linear):
                    m.weight.data.normal_()
            '''

    # feature normalization
    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.model.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.model.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    # returns face embedding(embedding_size)
    def forward(self, x):

        x = self.model(x)

        '''
        features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        features = features * alpha
        return features        
        '''
        return x


class VggModel(nn.Module):

    def __init__(self, pretrained=False):

        super(VggModel, self).__init__()

        self.model = vgg16(pretrained)
        self.hidden_features = 1024
        self.out_features = 128
        self.dropout_rate = 0.7

        # fc layer fine tuning
        self.model.classifier = nn.Linear(in_features=25088, out_features=128, bias=True)

        # self.model.classifier[6] = nn.Linear(4096, self.out_features)
        # self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        # fc layer initialization
        '''
        if pretrained:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    # m.weight.data.normal_()
                    nn.init.xavier_normal_(m.weight)
                    # m.weight.data.normal_()
                    m.bias.data.zero_()

            for m in model.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.normal(m.weight.data)
                    nn.init.xavier_normal(m.weight.data)
                    nn.init.kaiming_normal(m.weight.data)
                    m.bias.data.fill_(0)
                elif isinstance(m,nn.Linear):
                    m.weight.data.normal_()
            '''

    # feature normalization
    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.model.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.model.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    # returns face embedding(embedding_size)
    def forward(self, x):

        x = self.model(x)

        '''
        features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        features = features * alpha
        return features        
        '''
        return x


class DeModel(nn.Module):

    def __init__(self):

        super(DeModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=1, bias=False),
            nn.Sigmoid()
        )

    # returns face embedding(embedding_size)
    def forward(self, x):

        x = self.model(x)

        return x


# test model
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeModel().to(device)
    data_input = torch.randn([4, 256]).to(device)
    output = model(data_input)
    print(output.size(),output.shape)


