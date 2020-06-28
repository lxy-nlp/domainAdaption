import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from layers.bilstm import BiLSTM
from layers.critic import domain_critic
from ml import Mutual
from adapt import WasserAdaption
from train import train
from utils.dataload import load_data, loader_set

parameters = {
    'dataset_label': 'elc',
    'dataset_unlabel': 'cd',
    'embedding_size': 200,
    'max_length': 40,
    'max_seq_length': 5,
    'max_word_length':100,
    'hidden_size': 300,
    'keep_dropout': 0.2,
    'num_classes': 5,
    'learning_rate': 0.0001,
    'batch_size': 100,
    'epoches': 10,
}
# the parameters of preprocess
label_embedding, labeltrainset, labeldevset, labeltestset, wrd_dict = load_data(parameters['dataset_label'], parameters['embedding_size'], parameters['max_length'])
unlabel_embedding, unlabeltrainset, unlabeldevset, unlabeltestset, _ = load_data(parameters['dataset_unlabel'], parameters['embedding_size'], parameters['max_length'])

parameters['vocab_size'] = len(wrd_dict)
parameters['pretrained_embeddings'] = label_embedding

# init model

# data_loader
trainloader_label = loader_set(labeltrainset, parameters['batch_size'])
devloader_label = loader_set(labeldevset, parameters['batch_size'])
testloader_label = loader_set(labeltestset, parameters['batch_size'])
trainloader_unlabel = loader_set(unlabeltrainset, parameters['batch_size'])
devloader_unlabel = loader_set(unlabeldevset, parameters['batch_size'])
testloader_unlabel = loader_set(unlabeltestset, parameters['batch_size'])
# model = Mutual(parameters)

model = WasserAdaption(parameters)
critic = domain_critic(parameters)
# label_traindata, unlabel_traindata, unlabel_devdata, unlabel_testdata, epoches, model,learnig_rate
train(trainloader_label, trainloader_unlabel, devloader_unlabel, testloader_unlabel, parameters['epoches'], model, critic, parameters['learning_rate'])
print("suc")

# deep mutual learning
model1 = WasserAdaption(parameters)
critic1 = domain_critic(parameters)
model2 = WasserAdaption(parameters)
critic2 = domain_critic(parameters)
model = (model1, model2)
# train(trainloader_label, trainloader_unlabel, devloader_unlabel, testloader_unlabel, parameters['epoches'], model, critic, parameters['learning_rate'])




