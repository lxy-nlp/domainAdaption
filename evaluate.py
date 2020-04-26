import torch


def test(test_data, model, mode='test'):
    with torch.no_grad():
        correct = 0
        total = 0
        testacc = 0.0
        for step, (batch_x, batch_y) in enumerate(test_data):
            feature = model.feature_extractor(batch_x)
            logits,distance = model(feature,None,mode)
            _, prediction = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (prediction == batch_y-1).sum().item()
            testacc = correct/total
        print("Accuracy:{}".format(testacc))
        return testacc


def predict(input_data,model):
    with torch.no_grad():
        feature = model.feature_extractor(input_data)
        out = model(feature)
        _, prediction = torch.max(out.data, 1)
        return prediction.item()

