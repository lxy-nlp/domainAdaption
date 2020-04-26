'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# 分类器

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes, keep_dropout):
        super(Classifier, self).__init__()
        self.fc_out = nn.Sequential(
            nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_size * 2, self.num_classes)
        )

    def forward(self,lstm_out):
        logits = self.fc_out(lstm_out)
        return logits


def classify(lstm_out, input_size,num_classes,keep_dropout):
    fc1 = nn.Linear(input_size,input_size)
    fc2 = nn.Linear(input_size,num_classes)
    drop = nn.Dropout(keep_dropout)
    activation = nn.ReLU(inplace=True)
    out = activation(fc1(lstm_out))
    out = drop(out)
    logits = fc2(out)
    return logits
'''

