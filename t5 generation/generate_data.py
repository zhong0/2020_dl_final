# -*- coding: UTF-8 -*-

import json
import random

import pandas


def generate_data():
    fetch_docs = None
    with open('/home/albeli/workspace/NCCU/DeepLearningBasis/final/fetch_docs.json', 'r') as reader:
        fetch_docs = json.load(reader)

    new_fetch_docs = list()
    new_fetch_docs.extend(fetch_docs)
    for idx, doc in enumerate(fetch_docs):
        text = doc.get('text')
        if text:
            new_text = text.replace('\t', '').replace('\n', ' ').strip()
            new_text_length = len(new_text.split())
            new_fetch_docs[idx]['text'] = new_text
            new_fetch_docs[idx]['text_length'] = new_text_length

    with open('/home/albeli/workspace/NCCU/DeepLearningBasis/final/new_fetch_docs.json', 'w') as writer:
        json.dump(new_fetch_docs, writer)

    with open('/home/albeli/workspace/NCCU/DeepLearningBasis/final/new_fetch_docs.json', 'r') as reader:
        fetch_docs = json.load(reader)

    filtered_docs = list(filter(lambda item: item.get(
        'text_length') <= 155 and item.get('text_length') > 0, fetch_docs))
    print(len(filtered_docs))

    data = list(map(lambda item: [item.get('start'), item.get(
        'text'), item.get('headline')], filtered_docs))
    sample_data = random.sample(data, 10000)
    train_data = sample_data[:5000]
    test_data = sample_data[5000:]

    columns = ['index', 'text', 'headline']
    df = pandas.DataFrame(data, columns=columns)
    df.to_excel('raw_data_0108.xlsx', index=0)
    df = pandas.DataFrame(train_data, columns=columns)
    df.to_excel('train_data_0108.xlsx', index=0)
    df = pandas.DataFrame(test_data, columns=columns)
    df.to_excel('test_data_0108.xlsx', index=0)


if __name__ == "__main__":
    generate_data()
