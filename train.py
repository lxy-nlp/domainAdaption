import torch
import torch.nn as nn
from datetime import datetime

from evaluate import test


def train(label_traindata, unlabel_traindata, unlabel_devdata, unlabel_testdata, epoches, model, learnig_rate):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learnig_rate) # 设置学习率
    loss_history = []
    # 开始训练
    print("{},Start training".format(
        datetime.now().strftime('%02y/%02m%02d %H:%M:%S')))
    loss = 0
    bestdevacc = 0.0
    besttestacc = 0.0
    for epoch in range(epoches):
        for (batch_labelx,batch_labely), (batch_unlabelx,batch_unlabely) in zip(label_traindata, unlabel_traindata):
            features_label = model.feature_extractor(batch_labelx)
            features_unlabel = model.feature_extractor(batch_unlabelx)
            # print("feature ok")
            mode = 'train'
            logits,distance = model(features_label,features_unlabel,mode)
            loss = criterion(logits, batch_labely-1)+0.6*distance
            # print("compute loss")
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
        print('Epoch:', '%03d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss_history.append(loss)
        mode = 'test'
        print("test start")
        devacc = test(unlabel_devdata, model, mode)
        testacc = test(unlabel_testdata, model, mode)
        if devacc > bestdevacc:
            torch.save(model, 'bestdev.pth')
        if testacc >besttestacc:
            torch.save(model,'besttest.pth')










