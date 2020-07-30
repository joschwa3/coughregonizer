import tensorflow as tf
import os

sound_class = "/m/01b_21"
path = r'/Users/kylegood/Desktop/MastersProgram/data/audioset_v1_embeddings/bal_train/'

def get_files(directory_of_files):
    per_class_datasets = [tf.data.TFRecordDataset(tf.data.Dataset.list_files(directory_of_files)) for directory_of_files in sound_class]
    return per_class_datasets

def read_tfrecord(serialized_example):
    feature_description = \
    {
        'video_id': tf.io.FixedLenFeature([], tf.string),
        'start_time_seconds': tf.io.FixedLenFeature([], tf.int64),
        'end_time_seconds': tf.io.FixedLenFeature([], tf.float32),
        'labels': tf.io.FixedLenFeature([], tf.int64),
        'audio_embedding': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)

    video_id = example['video_id']
    start_time_seconds = example['start_time_seconds']
    end_time_seconds = example['end_time_seconds']
    labels = example['labels']
    audio_embedding = tf.io.parse_tensor(example['audio_embedding'], out_type = tf.float64)

    return video_id, start_time_seconds, end_time_seconds, labels, audio_embedding

read_tfrecord = get_files(path)
parsed_dataset = tfrecord_dataset.map(read_tfrecord)

