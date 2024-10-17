import tensorflow as tf

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
    
    # Convert chunk and reference_length
    chunk = tf.cast(parsed_features['chunk'], tf.float32)
    # For some reason the target_lengths in the train dataset are offset by 7!
    length = tf.minimum(parsed_features['reference_length'] + 7, 500)
    reference_length = tf.cast(length, tf.int32)
    
    return chunk, reference, reference_length

def load_tfrecords(tfrecord_file_path, batch_size=32, shuffle=False, repeat=False):
    # Create a dataset from the TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_file_path)
    
    if shuffle:
        # Shuffle the dataset
        dataset = dataset.shuffle(4096)
    
    if repeat:
        # Repeat dataset indefinitely
        dataset = dataset.repeat()
    
    # Map the parsing function over the dataset
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Return individual elements for unpacking
    #def unpack_data(chunks, targets, reference_length):
    #    return chunks, targets, reference_length

    #dataset = dataset.map(unpack_data)  # This will yield the three tensors
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Prefetch data to improve performance
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

