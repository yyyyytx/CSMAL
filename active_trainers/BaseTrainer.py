import torch
from torch.utils.data import DataLoader


class BaseActiveTrainer():
    def __init__(self, device):
        self.device = device

    def base_model_accuracy(self, net, test_loader):
        net.eval()
        correct_num = 0
        for Xs, ys in test_loader:
            Xs = Xs.to(self.device)
            ys = ys.to(self.device)

            with torch.set_grad_enabled(False):
                y_hats, _ = net(Xs)
                _, preds = torch.max(y_hats, 1)

            correct_num += torch.sum(preds == ys.data)
        return (float(correct_num) / float(len(test_loader.dataset)))

    def predict_entropy(self, net, dataset, idx):
        net.eval()
        subdataset = torch.utils.data.Subset(dataset, idx)
        data_iter = DataLoader(dataset=subdataset,
                               batch_size=self.config['test_batch_size'],
                               shuffle=False,
                               num_workers=4)
        predicts = []
        for Xs, ys in data_iter:
            Xs = Xs.to(self.device)
            ys = ys.to(self.device)
            with torch.set_grad_enabled(False):
                y_hats,_ = net(Xs)
            predicts.append(y_hats)
        predicts = torch.cat(predicts, dim=0)
        predicts = torch.softmax(predicts, dim=1)
        entropy = (- predicts * torch.log2(predicts)).sum(dim=1)
        return entropy

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
        if type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)

