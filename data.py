import tensorflow as tf
import math

def _parse_function(proto):
    # Define your feature description dictionary
    feature_description = {
        'chunk': tf.io.FixedLenFeature([5000], tf.float32),  # Assuming chunk has a fixed size of 5000
        'reference_indices': tf.io.VarLenFeature(tf.int64),  # Sparse tensor indices
        'reference_values': tf.io.VarLenFeature(tf.int64),  # Sparse tensor values
        'reference_dense_shape': tf.io.FixedLenFeature([1], tf.int64),  # Expecting dense_shape to be [1] for 1D
        'reference_length': tf.io.FixedLenFeature([], tf.int64),  # Reference length
    }
    
    # Parse the input tf.Example proto using the dictionary
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    
    # Extract indices and reshape them to [num_non_zero_elements, 1] for sparse tensor
    indices = tf.reshape(parsed_features['reference_indices'].values, [-1, 1])
    indices = tf.cast(indices, tf.int64)
    values = tf.cast(parsed_features['reference_values'].values, tf.int32)
    dense_shape = tf.cast(parsed_features['reference_dense_shape'], tf.int64)
    
    # Reconstruct the sparse tensor for reference
    reference = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=dense_shape
    )
    
    reference = tf.sparse.to_dense(reference)
    
    # Convert chunk and reference_length as they are
    chunk = tf.cast(parsed_features['chunk'], tf.float32)
    # The target_lengths in the train dataset are offset by 7
    length = tf.minimum(parsed_features['reference_length'], 500)
    reference_length = tf.cast(length, tf.int32)
    
    return chunk, reference, reference_length

def load_tfrecords(tfrecord_file_path, batch_size=32, shuffle=False, repeat=False):
    # Create a dataset from the TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_file_path)
    
    # Map the parsing function over the dataset
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if shuffle:
        # Shuffle the dataset
        dataset = dataset.shuffle(2048)
    
    if repeat:
        # Repeat dataset indefinitely
        dataset = dataset.repeat()
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Prefetch data to improve performance
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

# Naive reduction of training data set time steps
def reduce_data(chunks, targets, target_lengths, factor):
    # Convert targets from sparse to dense once at the beginning
    targets = tf.sparse.to_dense(targets)
    
    # Reduce the chunk size based on factor
    chunks = chunks[:, :tf.cast(tf.math.ceil(chunks.shape[1] / factor), dtype=tf.int32)]
    
    # Compute the reduced target lengths and cast to int
    target_lengths = tf.cast(tf.math.ceil(target_lengths / factor), dtype=tf.int32)
    
    # Use a mask to trim each row in `targets` according to `target_lengths`
    max_seq_length = tf.shape(targets)[1]  # Actual max length in targets (e.g., 500 in this case)
    mask = tf.sequence_mask(target_lengths, max_seq_length)  # Ensure mask has same width as targets

    # Apply the mask to reduce `targets` length based on `target_lengths`
    targets_new = tf.ragged.boolean_mask(targets, mask)  # Ragged tensor allows variable lengths per row
    
    # Convert the ragged tensor to a dense tensor, padding as needed to the correct shape
    target_shape = tf.cast(tf.math.ceil(tf.shape(targets)[1] / factor), dtype=tf.int32)
    targets_dense = targets_new.to_tensor(default_value=0, shape=[tf.shape(targets)[0], target_shape])
    
    # Convert back to sparse only once at the end
    targets = tf.sparse.from_dense(tf.cast(targets_dense, dtype=tf.int32))
    
    return chunks, targets, target_lengths
