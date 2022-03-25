import pandas as pd
import numpy as np
import xlrd

allData = pd.read_excel('all_data_0108.xlsx', header = 0)

trainSet = allData.iloc[:5000]
testSet = allData.iloc[5000:]

trainText = trainSet['text'].values.tolist()
trainHeadline = trainSet['headline'].values.tolist()

testIndex = testSet['index']
testText = testSet['text'].values.tolist()
testOriHeadline = testSet['headline']

trainTuple = []
for i in range(len(trainText)):
    trainTuple.append((trainText[i], trainHeadline[i]))

from headliner.trainer import Trainer
from headliner.model.transformer_summarizer import TransformerSummarizer

def read_data_iteratively():
    return ((trainText[i], trainHeadline[i]) for i in range(len(trainText)))

class DataIterator:
    def __iter__(self):
        return read_data_iteratively()

data_iter = DataIterator()

summarizer = TransformerSummarizer(embedding_size = 500, max_prediction_len = 20)
trainer = Trainer(batch_size = 100, steps_per_epoch = 300)
trainer.train(summarizer, data_iter, num_epochs = 5)

generatedHeadline = []
for i in range(len(testText)):
    generatedHeadline.append(summarizer.predict(testText[i]))
d = {"index":testIndex, "originalHeadline":testOriHeadline, "generatedHeadline":generatedHeadline}
result = pd.DataFrame(data = d)
result.to_excel('result_0108_large_epoch5.xlsx',encoding='utf-8', index = None)
