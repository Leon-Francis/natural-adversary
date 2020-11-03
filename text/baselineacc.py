import torch
from models import Baseline_LSTM
from utils import SNLIDataset, collate_snli

baseline_model = Baseline_LSTM(100, 300, 10, gpu=True)
baseline_model.load_state_dict(torch.load('./models/baseline/model_lstm.pt'))

corpus_train = SNLIDataset(train=True,
                           vocab_size=11004,
                           path="./data/classifier")
corpus_test = SNLIDataset(train=False,
                          vocab_size=11004,
                          path="./data/classifier")
trainloader = torch.utils.data.DataLoader(corpus_train,
                                          batch_size=32,
                                          collate_fn=collate_snli,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(corpus_test,
                                         batch_size=32,
                                         collate_fn=collate_snli,
                                         shuffle=False)


def evaluate_model():
    baseline_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for premise, hypothesis, target, _, _, _ in testloader:

            if True:
                premise = premise.cuda()
                hypothesis = hypothesis.cuda()
                target = target.cuda()

            prob_distrib = baseline_model((premise, hypothesis))
            correct += (prob_distrib.argmax(
                dim=1) == target).float().sum().item()
            total += premise.size(0)
        acc = correct / float(total)
        print("Accuracy:{0}".format(acc))
    baseline_model.train()
    return acc


if __name__ == "__main__":
    evaluate_model()