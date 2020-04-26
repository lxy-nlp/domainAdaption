import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from layers.bilstm import BiLSTM
from model import WasserAdaption
from train import train
from utils.dataload import load_data, loader_set

# the parameters of preprocess
dataset_label = 'cd'
dataset_unlabel = 'clt'
embedding_size = 200
max_length = 200
label_embedding, labeltrainset, labeldevset, labeltestset, wrd_dict = load_data(dataset_label, embedding_size,max_length)
unlabel_embedding, unlabeltrainset, unlabeldevset, unlabeltestset, _ = load_data(dataset_unlabel, embedding_size, max_length)

# the parameters of bilstm
vocab_size = len(wrd_dict)
pretrained_embeddings = label_embedding
hidden_size = 200
keep_dropout = 0.1
num_classes = 5

# init model
feature_extractor = BiLSTM(vocab_size, embedding_size, hidden_size,num_classes, pretrained_embeddings,keep_dropout)
WasserAda = WasserAdaption(feature_extractor,hidden_size,num_classes,keep_dropout)
learning_rate = 1e-3

# data_loader
batch_size = 50
trainloader_label = loader_set(labeltrainset, batch_size)
devloader_label = loader_set(labeldevset, batch_size)
testloader_label = loader_set(labeltestset, batch_size)
trainloader_unlabel = loader_set(unlabeltrainset, batch_size)
devloader_unlabel = loader_set(unlabeldevset, batch_size)
testloader_unlabel = loader_set(unlabeltestset, batch_size)
epoches = 30


# label_traindata, unlabel_traindata, unlabel_devdata, unlabel_testdata, epoches, model,learnig_rate
train(trainloader_label, trainloader_unlabel, devloader_unlabel,testloader_unlabel, epoches, WasserAda, learning_rate)
print("suc")




