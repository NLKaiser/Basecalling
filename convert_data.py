import numpy as np
import tensorflow as tf

def transform_target(target, state_len=5, n_base=4, alphabet_len=5, sparse=True):
    target = tf.cast(target, dtype=tf.int32)
    target = target-1
    index = tf.where(target == -1)
    if tf.size(index) > 0:
        target = target[:tf.reduce_min(index)]
    n = tf.size(target) - (state_len - 1)
    t_new = tf.zeros([n], dtype=tf.int32)
    for i in range(state_len):
        window = target[i:n+i] * (n_base ** (state_len - i - 1))
        t_new += window
    t_new *= alphabet_len
    target_move = t_new[1:] + target[:n-1] + 1
    pad_length = 500 - tf.shape(target_move)[0]
    target_move = tf.pad(target_move, paddings=[[0, pad_length]], constant_values=0)
    if sparse:
        target_move = tf.sparse.from_dense(target_move)
    return target_move

def serialize_sparse_example(chunk, sparse_reference, reference_length):
    # Convert sparse tensor to components
    indices = sparse_reference.indices.numpy().astype(np.int64)
    values = sparse_reference.values.numpy()
    dense_shape = sparse_reference.dense_shape.numpy().astype(np.int64)
    
    # Ensure dense_shape is a list with one value for 1D tensors
    assert len(dense_shape) == 1, "Dense shape should be a list with one value for 1D tensor."
    
    # Flatten indices for serialization
    flattened_indices = indices.flatten()
    
    feature = {
        'chunk': tf.train.Feature(float_list=tf.train.FloatList(value=chunk)),
        'reference_indices': tf.train.Feature(int64_list=tf.train.Int64List(value=flattened_indices)),
        'reference_values': tf.train.Feature(int64_list=tf.train.Int64List(value=values)),
        'reference_dense_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=dense_shape)),
        'reference_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[reference_length])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def convert_npys_to_tfrecords(chunks_file_path, references_file_path, reference_lengths_file_path, output_tfrecord_path, transform):
    # Load the numpy data
    chunks = np.load(chunks_file_path)
    references = np.load(references_file_path)
    reference_lengths = np.load(reference_lengths_file_path)
    
    # Create a TFRecord writer
    with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
        for i in range(len(chunks)):
            # Sensor readings
            chunk = chunks[i]
            
            # The reference sequences, padded with 0
            reference = references[i]
            # Convert dense reference to sparse tensor
            if transform:
                sparse_reference = transform_target(reference, state_len=5, n_base=4, alphabet_len=5, sparse=True)
            else:
                sparse_reference = tf.sparse.from_dense(reference)
            
            # Lengths of the reference sequences
            reference_length = reference_lengths[i]
            
            # Serialize the example
            example = serialize_sparse_example(chunk, sparse_reference, reference_length)
            
            # Write the serialized example to the TFRecord file
            writer.write(example)

def convert_dataset(path, transform=False):
    convert_npys_to_tfrecords(path + "chunks.npy", path + "references.npy", path + "reference_lengths.npy", "train_data.tfrecord", transform)
    path = path + "validation/"
    convert_npys_to_tfrecords(path + "chunks.npy", path + "references.npy", path + "reference_lengths.npy", "valid_data.tfrecord", transform)

path = "PATH_TO_NUMPY_DATASET/example_data_dna_r10.4.1_v0/"
convert_dataset(path, transform=True)
