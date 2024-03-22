from torch import nn
from torchtext import data

tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False)


class LSTM(nn.Module):
    def __init__(self, weight_matrix):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 300)  # embedding之后shape: torch.size([200, 8, 300])

        # 若使用预训练的词向量, 需在此处指定预训练的权重
        # self.word_embeddings.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1)  # torch.Size([200, 8, 128])
        self.decoder = nn.Linear(128, 2)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds)[0]  # lstm_out: 200*8*128
        final = lstm_out[-1]  # 8*128
        y = self.decoder(final)  # 8*2
        return y
