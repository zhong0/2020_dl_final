# -*- coding: UTF-8 -*-


import difflib
import json
import logging
import os
from datetime import datetime

import pandas
import tensorflow as tf
from transformers import T5Config, T5Tokenizer, TFT5ForConditionalGeneration


def get_t5_x_format(input_string):
    return 'headline: {}'.format(input_string)


if __name__ == "__main__":
    FORMAT = '%(asctime)s [%(module)s] [%(levelname)s]: %(message)s'
    log_file_name = '{}.log'.format(
        datetime.strftime(datetime.now(), '%Y-%m-%d'))
    logging.basicConfig(level=logging.INFO, filename=log_file_name,
                        filemode='a', format=FORMAT)
    T5_PRE_TRAINED = 't5-base'
    MAX_SEQ_LENGTH = 512
    LABES_MAX_SEQ_LENGTH = 128
    BATCH_SIZE = 100
    tokenizer = T5Tokenizer.from_pretrained(T5_PRE_TRAINED)

    data = pandas.read_excel('test_data_0108.xlsx')
    indice = data['index'].values
    test_data = list(map(get_t5_x_format, data['text'].values))
    raw_text = data['text'].values
    test_labels = data['headline'].values

    encoded_x = tokenizer.batch_encode_plus(
        test_data, return_tensors='tf', padding=True, pad_to_multiple_of=MAX_SEQ_LENGTH, return_token_type_ids=True)

    inputs_tensors = encoded_x.get('input_ids')
    attention_masks_tensors = encoded_x.get('attention_mask')
    token_types_ids_tensors = encoded_x.get('token_type_ids')
    input_tensors_set = {
        'input_ids': inputs_tensors,
        'attention_mask': attention_masks_tensors,
        'token_type_ids': token_types_ids_tensors,
        'labels': test_labels,
        'raw_text': raw_text
    }
    datasets = tf.data.Dataset.from_tensor_slices(input_tensors_set)
    datasets_total_length = int(
        round(datasets.cardinality().numpy() / BATCH_SIZE, 0)) + 1
    datasets = datasets.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    datasets = datasets.batch(BATCH_SIZE)

    current_model = 'model-5'
    config = T5Config.from_json_file('{}/config.json'.format(current_model))
    model = TFT5ForConditionalGeneration.from_pretrained(
        '{}/tf_model.h5'.format(current_model), config=config)
    model.summary()

    all_results = list()
    all_results_to_excel = list()
    for batch_index, batch_inputs in enumerate(datasets, 1):
        prediction = model.generate(input_ids=batch_inputs.get(
            'input_ids'), attention_mask=batch_inputs.get('attention_mask'))
        decoded_predictions = list(map(lambda item: item.replace(tokenizer.eos_token, '').replace(
            tokenizer.pad_token, '').strip(), tokenizer.batch_decode(prediction)))
        result_pairs = list(
            zip(batch_inputs.get('labels').numpy(), decoded_predictions))
        all_results.extend(result_pairs)

        to_excel_result_pairs = list(
            zip(batch_inputs.get('raw_text').numpy(), decoded_predictions))
        all_results_to_excel.extend(to_excel_result_pairs)

    encoding = 'utf-8'
    to_compute_results = list()
    all_complete_results = list()
    to_excel_results = list()
    for (index, (ground_truth, prediction), results_to_excel) in zip(indice, all_results, all_results_to_excel):
        ground_truth = ground_truth.decode(encoding)
        raw_text = results_to_excel[0].decode(encoding)
        matcher = difflib.SequenceMatcher(
            None, ground_truth.upper(), prediction.upper())
        ratio = matcher.ratio()
        all_complete_results.append(
            '\t'.join([str(index), ground_truth, prediction, str(ratio)]) + os.linesep)
        to_compute_results.append(
            (ground_truth, prediction, round(ratio, 4)))
        to_excel_results.append([index, raw_text, prediction])

    # with open('{}/prediction_results.tsv'.format(current_model), 'w') as writer:
    #     writer.write(
    #         '\t'.join(['index', 'ground_truth', 'prediction', 'ratio']) + os.linesep)
    #     writer.writelines(all_complete_results)

    results_to_count = dict()
    for ground_truth, prediction, ratio in to_compute_results:
        if results_to_count.get(ratio):
            results_to_count[ratio] += 1
        else:
            results_to_count[ratio] = 1
    with open('{}/similarity_dist.tsv'.format(current_model), 'w') as writer:
        results_pairs = list(results_to_count.items())
        results_pairs.sort(key=lambda item: item[0], reverse=True)
        to_write_results = list(
            map(lambda item: '{}\t{}\n'.format(item[0], item[1]), results_pairs))
        writer.writelines(to_write_results)

    df = pandas.DataFrame(to_excel_results, columns=[
                          'index', 'text', 'prediction'])
    df.to_excel('prediction.xlsx', index=0)
