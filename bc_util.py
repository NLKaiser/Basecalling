import numpy as np
import tensorflow as tf

import re
import parasail
from collections import defaultdict

import csv

split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")

def _parse_function(proto):
    # Define your feature description dictionary
    feature_description = {
        'chunk': tf.io.FixedLenFeature([5000], tf.float32),  # Assuming chunk has a fixed size of 5000
        'reference_indices': tf.io.VarLenFeature(tf.int64),  # Sparse tensor indices
        'reference_values': tf.io.VarLenFeature(tf.float32),  # Sparse tensor values
        'reference_dense_shape': tf.io.FixedLenFeature([1], tf.int64),  # Expecting dense_shape to be [1] for 1D
        'reference_length': tf.io.FixedLenFeature([], tf.int64),  # Reference length
    }
    
    # Parse the input tf.Example proto using the dictionary
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    
    # Extract indices and reshape them to [num_non_zero_elements, 1] for sparse tensor
    reference_indices = tf.reshape(parsed_features['reference_indices'].values, [-1, 1])
    
    # Reconstruct the sparse tensor for reference
    reference = tf.SparseTensor(
        indices=tf.cast(reference_indices, tf.int64),  # Reshaped indices for the sparse tensor
        values=tf.cast(parsed_features['reference_values'].values, tf.int32),  # Sparse tensor values
        dense_shape=tf.cast(parsed_features['reference_dense_shape'], tf.int64)  # Dense shape of the tensor
    )
    
    #reference = tf.sparse.to_dense(reference)
    
    # Convert chunk and reference_length as they are
    chunk = tf.cast(parsed_features['chunk'], tf.float32)
    # For some reason the target_lengths in the train dataset are offset by 7!
    reference_length = tf.cast(parsed_features['reference_length'], tf.int16) + 7
    
    return chunk, reference, reference_length

def load_tfrecords(tfrecord_file_path, batch_size=32, shuffle=False, repeat=False):
    # Create a dataset from the TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_file_path)
    
    if shuffle:
        # Shuffle the dataset
        dataset = dataset.shuffle(2048)
    
    if repeat:
        # Repeat dataset indefinitely
        dataset = dataset.repeat()
    
    # Map the parsing function over the dataset
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Return individual elements for unpacking
    def unpack_data(chunks, targets, reference_length):
        return chunks, targets, reference_length

    dataset = dataset.map(unpack_data)  # This will yield the three tensors separately
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Prefetch data to improve performance
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset


class CosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, final_lr, decay_steps):
        super(CosineDecaySchedule, self).__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_steps = decay_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        cosine_decay = 0.5 * (tf.cos(np.pi * step / decay_steps) + 1.0)
        decayed_lr = (self.initial_lr - self.final_lr) * cosine_decay + self.final_lr
        return decayed_lr

def get_learning_rate(initial_lr=1e-4, final_lr=1e-6, decay_steps=100, epochs=None, steps_per_epoch=None):
    if (epochs != None and steps_per_epoch != None):
        decay_steps = epochs * steps_per_epoch
    return CosineDecaySchedule(initial_lr, final_lr, decay_steps)
    
class LinearDecayScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, final_learning_rate, total_epochs, steps_per_epoch):
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.total_steps = total_epochs * steps_per_epoch  # Total number of steps across all epochs

    def __call__(self, step):
        # Cast the current step as float
        step = tf.cast(step, tf.float32)

        # Linear decay formula based on the step
        decayed_learning_rate = self.initial_learning_rate - \
            (step / self.total_steps) * (self.initial_learning_rate - self.final_learning_rate)

        return decayed_learning_rate

def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded.tolist() if e)

def parasail_to_sam(result, seq):
    """
    Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.
    """
    cigstr = result.cigar.decode.decode()
    first = re.search(split_cigar, cigstr)

    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query

    clip = '' if cliplen == 0 else '{}S'.format(cliplen)
    if first_op == 'I':
        pre = '{}S'.format(int(first_count) + cliplen)
    elif first_op == 'D':
        pre = clip
        rstart = int(first_count)
    else:
        pre = '{}{}'.format(clip, prefix)

    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = '{}S'.format(end_clip) if end_clip > 0 else ''
    new_cigstr = ''.join((pre, mid, suf))
    return rstart, new_cigstr

def accuracy(ref, seq, balanced=False, min_coverage=0.0):
    """
    Calculate the accuracy between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    counts = defaultdict(int)

    q_coverage = len(alignment.traceback.query) / len(seq)
    r_coverage = len(alignment.traceback.ref) / len(ref)

    if r_coverage < min_coverage:
        return 0.0

    _, cigar = parasail_to_sam(alignment, seq)

    for count, op  in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    if balanced:
        accuracy = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
    else:
        accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    return accuracy * 100

class ReverseLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ReverseLayer, self).__init__(**kwargs)
        self.axis = axis  # The axis to reverse along, default is the last axis

    def call(self, inputs):
        # Reverse the input tensor along the specified axis
        return tf.reverse(inputs, axis=[self.axis])

    def get_config(self):
        config = super(ReverseLayer, self).get_config()
        config.update({"axis": self.axis})
        return config

class CastToFloat32(tf.keras.layers.Layer):
    def __init__(self):
        super(CastToFloat32, self).__init__()

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

def model_summary_to_string(model):
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)

class CustomCSVLogger:
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames

        # Open the file in write mode to initialize with the header
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()  # Write the header

    def __call__(self, data):
        # Open the file in append mode to write a new row
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(data)
