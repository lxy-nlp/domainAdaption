import torch
import scipy
from scipy.stats import wasserstein_distance

# feature1->labelled feature2->unlabelled

def wasserstein(feature1, feature2,length):
    feature1 = feature1.cpu().detach().numpy()
    feature2 = feature2.cpu().detach().numpy()
    distance = 0.0
    for row in range(length):
        distance += wasserstein_distance(feature1[row], feature2[row])
    return torch.tensor(distance/len(feature1))
