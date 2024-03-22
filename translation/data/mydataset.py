import pandas as pd
from torchtext import data
import random
import numpy as np
from torchtext.data import BucketIterator, Iterator
from tqdm import tqdm


class MyDataset(data.Dataset):
    def __init__(self, csv_path, text_field, label_field, test=False, aug=False, **kwargs):
        # 数据处理操作格式
        fields = [("id", None),  # we won't be needing the id, so we pass in None as the field
                  ("comment_text", text_field), ("toxic", label_field)]

        csv_data = pd.read_csv(csv_path)
        examples = []
        if test:
            # 如果为测试集, 则不加载标签
            for text in tqdm(csv_data["comment_text"]):
                examples.append(data.Example.fromlist([None, text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data['comment_text'], csv_data['toxic'])):
                # 数据增强
                if aug:
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                examples.append(data.Example.fromlist([None, text, label], fields))

        # 上面是一些预处理操作，此处调用super调用父类构造方法，产生标准Dataset
        super(MyDataset, self).__init__(examples=examples, fields=fields)

    @staticmethod
    def shuffle(text):
        # 序列随机排序
        text = np.random.permutation(text.strip().split())
        return " ".join(text)

    @staticmethod
    def dropout(text, p=0.5):
        # 随机删除一些文本
        text = text.strip().split()
        len_ = len(text)
        index = np.random.choice(len_, int(len_ * p))
        for i in index:
            text[i] = ""
        return " ".join(text)


def data_iter(train_path, valid_path, test_path, TEXT, LABEL):
    train = MyDataset(train_path, text_field=TEXT, label_field=LABEL, test=False, aug=1)
    valid = MyDataset(valid_path, text_field=TEXT, label_field=LABEL, test=False, aug=1)
    # 因为test没有label，需要指定label_field为None
    test = MyDataset(test_path, text_field=TEXT, label_field=None, test=True, aug=1)

    TEXT.build_vocab(train)
    weight_matrix = TEXT.vocab.vectors
    train_iter, val_iter = BucketIterator.splits(
        (train, valid),  # 构建数据所需的数据集
        batch_size=(8, 8),
        # 如果使用gpu, 此处将-1更换为GPU的编码
        device=-1,
        # the bucketIterator needs to be told what function it should use to group the data.
        sort_key=lambda x: len(x.comment_text),
        sort_within_batch=False,
        repeat=False
    )
    test_iter = Iterator(test, batch_size=8, device=-1, sort=False, sort_within_batch=False, repeat=False)
    return train_iter, val_iter, test_iter, weight_matrix


if __name__ == '__main__':
    train_path = "./emotion_class/train_one_label.csv"
    valid_path = "./emotion_class/valid_one_label.csv"
    test_path = "./emotion_class/test.csv"

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
    LABEL = data.Field(sequential=False, use_vocab=False)

    train_iter, val_iter, test_iter, weight_matrix = data_iter(train_path, valid_path, test_path, TEXT, LABEL)
    print(f"train_iter size: {train_iter.batch_size}")
