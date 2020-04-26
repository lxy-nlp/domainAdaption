import torch
import torch.nn as nn

# from layers.classifier import classify
from layers.distance import wasserstein


class WasserAdaption(nn.Module):
    def __init__(self, feature_extractor, hidden_size, num_classes,keep_dropout):
        super(WasserAdaption, self).__init__()
        self.input_size = hidden_size*2
        self.num_classes = num_classes
        self.keep_dropout = keep_dropout
        self.feature_extractor = feature_extractor
        self.fc1 = nn.Linear(self.input_size, self.input_size)
        self.fc2 = nn.Linear(self.input_size, self.num_classes)
        self.drop = nn.Dropout(keep_dropout)
        self.activation = nn.ReLU(inplace=True)

    def classifier(self, lstm_out):
        tmp_out = self.drop(lstm_out)
        tmp_out = self.activation(self.fc1(tmp_out))
        tmp_out = self.drop(tmp_out)
        logits = self.fc2(tmp_out)
        return logits

    def get_distance(self, feature1, features2):
        distance = wasserstein(feature1, features2)
        return distance

    def forward(self, label_features, unlabel_features=None, mode='train'):
        # feature_unlabel = self.feature_extractor(unlabel_data)
        # distance = wasserstein_distance(feature_label, feature_unlabel)
        distance = 0.0
        if mode == 'train':
            distance = self.get_distance(label_features, unlabel_features)
        else:
            distance = 0.0
        logtis = self.classifier(label_features)
        return logtis, distance

