#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
import logging
import os
import time
import traceback

import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':
    FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    with open('News_Category_Dataset_v3.json', 'r') as reader:
        data = json.load(reader)

    header = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36',
        'Connection': 'close'
    }

    with open('stopped_record.txt', 'r') as reader:
        last_record = int(reader.read().strip())

    docs = list()
    fetch_docs = list()
    for index, item in enumerate(data, 1):
        try:
            res = requests.get(item.get('link'), headers=header)
            if not res.ok:
                raise Exception('Response is not ok!')
            soup = BeautifulSoup(res.text, 'lxml')
            paragraphs = list(
                map(lambda tag: tag.text.strip(), soup.select('article p')))
            doc = os.linesep.join(paragraphs)
            doc_length = len(doc.split())
            to_save_data = {
                'start': index,
                'text': doc,
                'text_length': doc_length
            }
            to_save_data.update(item)
            docs.append(to_save_data)
            if len(docs) % 50 == 0:
                with open('fetch_docs.json', 'r', encoding='utf-8') as reader:
                    fetch_docs.extend(json.load(reader))
                with open('fetch_docs.json', 'w', encoding='utf-8') as writer:
                    fetch_docs.extend(docs)
                    json.dump(fetch_docs, writer)
                logging.info(
                    'Write docs in fetch_docs.json, current total data: {}'.format(len(fetch_docs)))
                docs.clear()
                fetch_docs.clear()
            time.sleep(0.1)
        except:
            logging.info('Error at {}, {}'.format(
                index, traceback.format_exc()))
