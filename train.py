import torch
import torch.nn as nn
from datetime import datetime

from evaluate import test
from layers.cosine import cosine
from layers.mmd import mmd
from layers.wassert import wasserstein

import scipy.stats # 计算ＫＬ散度

def train(label_traindata, unlabel_traindata, unlabel_devdata, unlabel_testdata, epoches, model, critic,learnig_rate):
    '''
    得到lstm提取的特征
    计算二者的推土机距离
    在有标签的数据集上进行分类，得到分类的交叉熵损失
    将推土机距离和交叉熵损失相加作为总的损失进行反向传播
    利用集成的思想　加入互学习 后续可能会加上，目前不确定
    确定迭代次数
    
    :return: 
    '''

    criterion = nn.CrossEntropyLoss() #  定义交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learnig_rate)  # 设置学习率
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=learnig_rate)  # 设置域识别器的学习率
    loss_history = []
    # 开始训练
    print("{},Start training".format(
        datetime.now().strftime('%02y/%02m%02d %H:%M:%S')))
    loss = 0
    bestdevacc = 0.0
    besttestacc = 0.0
    for epoch in range(epoches):
        for (batch_labelx,batch_labely), (batch_unlabelx,batch_unlabely) in zip(label_traindata, unlabel_traindata):
            # print("feature ok")
            '''
            #  deep mutual learning
            logits1, logits2, distance = model(batch_labelx, batch_unlabelx, mode)
            # 将　logits　从张量转换成 list　的形式
            logits_array1 = logits1.cpu().detach().numpy()
            logits_array2 = logits2.cpu().detach().numpy()
            kl = computewa(logits_array1, logits_array2)
            # 模型转换成　深度互学习的表现形式
            # 损失函数由　两个分类器的交叉熵损失函数　特征提取器的分布差异　分类器之间的kl散度　组成
            loss = criterion(logits1, batch_labely-1) + criterion(logits2, batch_labely-1) + 0.6*distance + 0.1*kl
            '''
            # 需要创建　域识别器的标签　源域-0 目标域-1
            batch_domains = torch.zeros_like(batch_labely)
            batch_domaint = torch.ones_like(batch_unlabely)
            feature_label, attnweight = model.feature_extractor(batch_labelx)
            feature_unlabel,_ = model.feature_extractor(batch_unlabelx)
            length = compare(feature_label,feature_unlabel)
            discrepancy = 0.5 * wasserstein(feature_label, feature_unlabel,length) + 0.2 * cosine(feature_label, feature_unlabel,length)+ 0.3 * mmd(feature_label[:length], feature_unlabel[:length])   # 两个分布之间的推土机距离
            logits_critics, logits_critict = critic(feature_label, feature_unlabel)
            loss_critic = criterion(logits_critics, batch_domains)+criterion(logits_critict, batch_domaint)
            # logits_class, logits_domain, distance = model(feature_label, feature_unlabel, mode)  # distance is discrepancy between source and target
            logits_class= model(feature_label)   # 情感分类器的分类结果
            loss_class = criterion(logits_class, batch_labely-1)    # loss for classifier
            loss_classes = loss_class + 0.8 * discrepancy + 0.1*torch.norm(attnweight, p=2) # 将注意力权重作为惩罚项
            optimizer.zero_grad()
            loss_classes.backward(retain_graph=True)
            # print("compute loss")
            # 使用梯度削减策略
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.15)
            optimizer.step()
            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()
            total_loss = loss_critic + loss_classes
        print('Epoch:', '%03d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_loss))
        loss_history.append(loss)
        print("test start")
        devacc = test(unlabel_devdata, model)
        testacc = test(unlabel_testdata, model)
        if devacc > bestdevacc:
            torch.save(model, 'bestdev.pth')
        if testacc >besttestacc:
            torch.save(model, 'besttest.pth')


def compare(feature1,feature2):
    feature1 = feature1.cpu().detach().numpy()
    feature2 = feature2.cpu().detach().numpy()
    if len(feature1) == len(feature2) or len(feature1) < len(feature2):
        length = len(feature1)
    else:
        length = len(feature2)
    return length


'''
def train(label_traindata, unlabel_traindata, unlabel_devdata, unlabel_testdata, epoches, model, critic, learnig_rate):

    得到lstm提取的特征
    计算二者的推土机距离
    在有标签的数据集上进行分类，得到分类的交叉熵损失
    将推土机距离和交叉熵损失相加作为总的损失进行反向传播
    利用集成的思想　加入互学习 后续可能会加上，目前不确定
    确定迭代次数
    :return:

    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learnig_rate)  # 设置学习率
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=learnig_rate)  # 设置域识别器的学习率
    loss_history = []
    # 开始训练
    print("{},Start training".format(
        datetime.now().strftime('%02y/%02m%02d %H:%M:%S')))
    loss = 0
    bestdevacc = 0.0
    besttestacc = 0.0
    for epoch in range(epoches):
        for (batch_labelx, batch_labely), (batch_unlabelx, batch_unlabely) in zip(label_traindata, unlabel_traindata):
            # print("feature ok")
            
            #  deep mutual learning
            logits1, logits2, distance = model(batch_labelx, batch_unlabelx, mode)
            # 将　logits　从张量转换成 list　的形式
            logits_array1 = logits1.cpu().detach().numpy()
            logits_array2 = logits2.cpu().detach().numpy()
            kl = computewa(logits_array1, logits_array2)
            # 模型转换成　深度互学习的表现形式
            # 损失函数由　两个分类器的交叉熵损失函数　特征提取器的分布差异　分类器之间的kl散度　组成
            loss = criterion(logits1, batch_labely-1) + criterion(logits2, batch_labely-1) + 0.6*distance + 0.1*kl

            # 需要创建　域识别器的标签　源域-0 目标域-1
            batch_domains = torch.zeros_like(batch_labely)
            batch_domaint = torch.ones_like(batch_unlabely)
            feature_label_1 = model[0].feature_extractor(batch_labelx)
            feature_unlabel_1 = model[0].feature_extractor(batch_unlabelx)
            feature_label_2 = model[1].feature_extractor(batch_labelx)
            feature_unlabel_2 = model[1].feature_extractor(batch_unlabelx)
            length = compare(feature_label_1, feature_unlabel_1)
            discrepancy_1 = 0.5 * wasserstein(feature_label_1, feature_unlabel_1,length) + 0.2 * cosine(
                feature_label_1, feature_unlabel_1,length) + 0.3 * mmd(feature_label_1[:length], feature_unlabel_1[:length])  # 两个分布之间的推土机距离

            discrepancy_２ = 0.5 * wasserstein(feature_label_2, feature_unlabel_2,length) + 0.2 * cosine(
                feature_label_2, feature_unlabel_2,length) + 0.3 * mmd(feature_label_2[:length], feature_unlabel_2[:length])

            logits_critics, logits_critict = critic(feature_label_1, feature_unlabel_1)

            loss_critic = criterion(logits_critics, batch_domains) + criterion(logits_critict, batch_domaint)
            # logits_class, logits_domain, distance = model(feature_label, feature_unlabel, mode)  # distance is discrepancy between source and target
            logits_class_1 = model[0](feature_label_1)  # 情感分类器的分类结果
            loss_class_1 = criterion(logits_class_1, batch_labely - 1)  # loss for classifier
            loss_classes_1 = loss_class_1 + 0.8 * discrepancy_1
            logits_class_2 = model[1](feature_label_2)
            loss_classes_2 = criterion(logits_class_2, batch_labely-1)
            optimizer.zero_grad()
            loss_classes.backward(retain_graph=True)
            # print("compute loss")
            # 使用梯度削减策略
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.15)
            optimizer.step()
            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()
            total_loss = loss_critic + loss_classes
        print('Epoch:', '%03d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_loss))
        loss_history.append(loss)
        print("test start")
        devacc = test(unlabel_devdata, model)
        testacc = test(unlabel_testdata, model)
        if devacc > bestdevacc:
            torch.save(model, 'bestdev.pth')
        if testacc > besttestacc:
            torch.save(model, 'besttest.pth')





'''








