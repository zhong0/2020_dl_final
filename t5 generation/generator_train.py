# -*- coding: UTF-8 -*-


import json
import logging
import os
from datetime import datetime

import tensorflow as tf
from transformers import T5Config, T5Tokenizer, TFT5ForConditionalGeneration
import pandas


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
    BATCH_SIZE = 4
    EPOCHS = 5
    tokenizer = T5Tokenizer.from_pretrained(T5_PRE_TRAINED)

    data = pandas.read_excel('train_data_0108.xlsx')
    indice = data['index'].values
    train_data = list(map(get_t5_x_format, data['text'].values))
    train_labels = data['headline'].values
    encoded_x = tokenizer.batch_encode_plus(
        train_data, return_tensors='tf', padding=True, pad_to_multiple_of=MAX_SEQ_LENGTH, return_token_type_ids=True)
    encoded_y = tokenizer.batch_encode_plus(
        train_labels)

    inputs_tensors = encoded_x.get('input_ids')
    records = list(map(lambda item: len(item), inputs_tensors))
    attention_masks_tensors = encoded_x.get('attention_mask')
    token_types_ids_tensors = encoded_x.get('token_type_ids')
    decoder_attention_masks_tensors = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_y.get('attention_mask'), padding='post', maxlen=LABES_MAX_SEQ_LENGTH)
    decoder_inputs_tensors = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_y.get('input_ids'), padding='post', maxlen=LABES_MAX_SEQ_LENGTH, value=-100)
    input_tensors_set = {
        'input_ids': inputs_tensors,
        'attention_mask': attention_masks_tensors,
        'token_type_ids': token_types_ids_tensors,
        'decoder_attention_mask': decoder_attention_masks_tensors,
        'labels': decoder_inputs_tensors
    }

    train_datasets = tf.data.Dataset.from_tensor_slices(input_tensors_set)
    datasets_total_length = int(
        round(train_datasets.cardinality().numpy() / BATCH_SIZE, 0)) + 1
    train_datasets = train_datasets.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    train_datasets = train_datasets.batch(BATCH_SIZE)

    config = T5Config.from_pretrained(T5_PRE_TRAINED)
    model = TFT5ForConditionalGeneration.from_pretrained(
        T5_PRE_TRAINED, config=config)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-5, epsilon=1e-8, clipnorm=1.0)
    model.summary()

    for epoch in range(1, EPOCHS + 1):
        for step, batch_train_inputs in enumerate(train_datasets, 1):
            with tf.GradientTape() as tape:
                outputs = model(batch_train_inputs, return_dict=True)
                loss = tf.reduce_mean(outputs.get('loss'))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables))

            logging.info('Epoch: {}/{}, Step: {}/{}, Progress: {:.4%}, Loss: {}'.format(
                epoch,
                EPOCHS,
                step,
                datasets_total_length,
                step / datasets_total_length,
                loss))
        folder_path = '/home/albeli/workspace/NCCU/DeepLearningBasis/final/model-{}'.format(
            epoch)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        model.save_pretrained(folder_path)
