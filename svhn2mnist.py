import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse


class Feature_base(nn.Module):
    def __init__(self):
        super(Feature_base, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = x.view(x.size(0), 8192)
        # x = F.relu(self.bn1_fc(self.fc1(x)))
        # x = F.dropout(x, training=self.training)
        return x


class Feature_disentangle(nn.Module):
    def __init__(self,in_feature,out_feature1,out_feature2):
        super(Feature_disentangle, self).__init__()
        self.fc1 = nn.Linear(in_feature, out_feature1)
        self.bn1_fc = nn.BatchNorm1d(out_feature1)
        self.fc2 = nn.Linear(out_feature1, out_feature2)
        self.bn2_fc = nn.BatchNorm1d(out_feature2)
    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x


class Feature_discriminator(nn.Module):
    def __init__(self,in_features,out_features):
        super(Feature_discriminator, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc2 = nn.Linear(out_features, 2)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return x


class Reconstructor(nn.Module):
    def __init__(self,in_features,out_features):
        super(Reconstructor, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
    def forward(self,x):
        x = self.fc(x)
        return x


class Mine(nn.Module):
    def __init__(self,in_features,out_features):
        super(Mine, self).__init__()
        self.fc1_x = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc1_y = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc2 = nn.Linear(in_features=out_features,out_features=1)
    def forward(self, x,y):
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


class Predictor(nn.Module):
    def __init__(self, in_features,out_features,prob=0.5):
        super(Predictor, self).__init__()
        self.fc3 = nn.Linear(in_features=in_features, out_features=out_features)
        self.bn_fc3 = nn.BatchNorm1d(out_features)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        # x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.relu(self.bn_fc3(self.fc3(x)))
        return x
