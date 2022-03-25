import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from simpletransformers.classification import ClassificationModel
import pandas as pd
import numpy
import logging
import xlrd
from sklearn.metrics import f1_score

trainSet = pd.read_excel('trainData.xlsx', header = 0)
testSet = pd.read_excel('testData.xlsx', header = 0)
testData = testSet['headline']

model = ClassificationModel('bert', 'bert-large-cased', num_labels = 2, args = {'num_train_epochs': 1, 'reprocess_input_data': True, 'overwrite_output_dir': True})
model.train_model(trainSet)
torch.save(model, 'classificationModel.pt')
result, model_outputs, wrong_predictions = model.eval_model(testSet)
print(result)

predictions, raw_outputs = model.predict(testData.values)
d = {'Headline':testData, 'Predict_Classification':predictions, 'Origincal_Classfication': testSet['label']}
df = pd.DataFrame(data = d)
df.to_excel('classifyResult_t5_ori.xlsx', encoding = 'utf-8', index = None)
print(f1_score(testSet['label'], predictions, average='macro'))
