from torch import optim
import torch.nn.functional as F

from translation.data.mydataset import data_iter
from translation.model.lstm import LSTM


def main():
    train_iter, val_iter, test_iter, weight_matrix = data_iter(train_path, valid_path, test_path, TEXT, LABEL)

    model = LSTM(weight_matrix)
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    loss_funtion = F.cross_entropy

    for epoch, batch in enumerate(train_iter):
        optimizer.zero_grad()
        predicted = model(batch.comment_text)

        loss = loss_funtion(predicted, batch.toxic)
        loss.backward()
        optimizer.step()
        print(loss)


if __name__ == '__main__':
    from torchtext import data

    train_path = "./data/emotion_class/train_one_label.csv"
    valid_path = "./data/emotion_class/valid_one_label.csv"
    test_path = "./data/emotion_class/test.csv"

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
    LABEL = data.Field(sequential=False, use_vocab=False)

    main()