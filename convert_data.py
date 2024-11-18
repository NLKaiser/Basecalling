"""
Script to convert a numpy training dataset to tfrecords.
Initial data structure:
- example_data_dna_r10.4.1_v0
  - chunks.npy (1.000.000, 5.000) # Sensor measurements
  - references.npy (1.000.000, 500) # Reference sequence encoded (1-4) with zero padding
  - reference_lengths.npy (1.000.000) # Length of the reference sequence
  - validation
    - chunks.npy (50.000, 5.000)
    - references.npy (50.000, 500)
    - reference_lengths.npy (50.000)
"""

import numpy as np
import tensorflow as tf

def serialize_sparse_example(chunk, sparse_reference, reference_length):
    # Sparse tensor components
    indices = sparse_reference.indices.numpy().astype(np.int64)
    values = sparse_reference.values.numpy()
    dense_shape = sparse_reference.dense_shape.numpy().astype(np.int64)
    
    # Ensure dense_shape is a list with one value for 1D tensors
    assert len(dense_shape) == 1, "Dense shape should be a list with one value for 1D tensor."
    
    # Flatten indices for serialization
    flattened_indices = indices.flatten()
    
	# Define one entry
    feature = {
        'chunk': tf.train.Feature(float_list=tf.train.FloatList(value=chunk)),
        'reference_indices': tf.train.Feature(int64_list=tf.train.Int64List(value=flattened_indices)),
        'reference_values': tf.train.Feature(float_list=tf.train.FloatList(value=values)),
        'reference_dense_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=dense_shape)),
        'reference_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[reference_length])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def convert_npys_to_tfrecords(chunks_file_path, references_file_path, reference_lengths_file_path, output_tfrecord_path):
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
            sparse_reference = tf.sparse.from_dense(reference)
            
            # Lengths of the reference sequences
            reference_length = reference_lengths[i]
            
            # Serialize the example
            example = serialize_sparse_example(chunk, sparse_reference, reference_length)
            
            # Write the serialized example to the TFRecord file
            writer.write(example)

def convert_dataset(path):
    convert_npys_to_tfrecords(path + "chunks.npy", path + "references.npy", path + "reference_lengths.npy", "train_data.tfrecord")
    path = path + "validation/"
    convert_npys_to_tfrecords(path + "chunks.npy", path + "references.npy", path + "reference_lengths.npy", "valid_data.tfrecord")

path = "PATH_TO_NUMPY_DATASET/example_data_dna_r10.4.1_v0/"
convert_dataset(path)
