import torch


def test(test_data, model):
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (batch_x, batch_y) in enumerate(test_data):
            feature, _ = model.feature_extractor(batch_x)
            logits = model(feature)
            _, prediction = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (prediction == batch_y-1).sum().item()
        testacc = correct/total
        print("Accuracy:{}".format(testacc))
        return testacc


def predict(input_data, model):
    with torch.no_grad():
        feature = model.feature_extractor(input_data)
        out = model(feature)
        _, prediction = torch.max(out.data, 1)
        return prediction.item()


'''
def test(test_data, model, mode='test'):
    with torch.no_grad():
        correct1 = 0
        correct2 = 0
        correct = 0
        total = 0
        testacc = 0.0
        for step, (batch_x, batch_y) in enumerate(test_data):
            logits1,logits2,distance = model(batch_x,None,mode)
            _, prediction1 = torch.max(logits1.data, 1)
            _,prediction2 = torch.max(logits2.data,1)
            total += batch_y.size(0)
            correct1 += (prediction1 == batch_y-1).sum().item()
            correct2 += (prediction2 == batch_y-1).sum().item()
            if correct1 >= correct2:
                correct = correct1
            else:
                correct = correct2
            testacc = correct/total
        print("Accuracy:{}".format(testacc))
        return testacc

'''

