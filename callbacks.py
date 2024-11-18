"""
Custom callbacks for training.
"""

import bc_util

import LRU_tf as lru

import numpy as np
import tensorflow as tf

import json
import csv

# Calculate and print different training metrics suc as accuracy
class Metrics:
    
    def __init__(self, alphabet, model_summary):
        self.alphabet = alphabet
        self.model_summary = model_summary
        
        self.mean_accuracy = 0
        self.median_accuracy = 0
        
        self.max_mean_accuracy = 0
    
    def on_epoch_end(self, prediction, targets):
        val_targets = tf.sparse.to_dense(targets, default_value=0)
        val_targets = val_targets.numpy()
        
        # Decode the predictions
        decoded_sequences = self.decode_predictions(prediction)
        
        # val_targets are numbers from 1 to 4, padded with 0
        refs = [bc_util.decode_ref(self.remove_trailing(target, 0), self.alphabet) for target in val_targets]
        # decoded_sequences are numbers from 1 to 4, padded with -1
        seqs = [bc_util.decode_ref(self.remove_trailing(target, -1), self.alphabet) for target in decoded_sequences]
        
		# Only calculate accuracy if the seq length is at least half that of the ref
        accs = [bc_util.accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, seqs)]
        
        # Print the model stats to see the number of parameters
        print(self.model_summary)
        # Print some of the decoded arrays together with the expected output
        #for i in range(1):    #(max(1, min(2, int(len(decoded_sequences)/8)))):
        #    seq = self.remove_trailing(decoded_sequences[i], -1)
        #    valid = self.remove_trailing(val_targets[i], 0)
        #    print("Original:", valid)
        #    print("Original length:", len(valid))
        #    print("Prediction:", seq)
        #    print("Prediction length:", len(seq))
        # Max accuracy during training
        self.max_mean_accuracy = max(self.max_mean_accuracy, np.mean(accs))
        # Print training statistics
        print(f"Max mean accuracy over training: {self.max_mean_accuracy:.2f}")
        print(f"mean_acc = {np.mean(accs):.2f}")
        print(f"median_acc = {np.median(accs):.2f}")
        print("Accuracies:", [f"{acc:.2f}" for acc in accs])
		# GPU number in case training is done with multiple consoles on multiple GPUs
        print("This is GPU 1!")
        self.mean_accuracy = np.mean(accs)
        self.median_accuracy = np.median(accs)
    
    def decode_predictions(self, logits):
        # Get the length of each sequence in the batch
        input_lengths = np.ones(logits.shape[0]) * logits.shape[1]
        
        # Decode the predictions
        decoded_sequences = tf.keras.ops.ctc_decode(logits, sequence_lengths=input_lengths, strategy="beam_search", beam_width=32)
        decoded_sequences = tf.cast(decoded_sequences[0][0], tf.int32)
        
        # Convert the tensor sequences to numpy arrays
        decoded_arrays = [seq.numpy() for seq in decoded_sequences]
        return decoded_arrays
    
    def remove_trailing(self, arr, num):
        indices = np.where(arr == num)[0]
        if indices.size == 0:
            # No num found, return the array as is
            return arr
        else:
            # Slice up to the first num
            return arr[:indices[0]]

# Save a models weights and reload them
class ModelReset:
    def __init__(self, model, reset_counter=0):
        model.save_weights("model.weights.h5", overwrite=True)
        self.reset_counter = reset_counter
    
    def save(self, model):
        model.save_weights("model.weights.h5", overwrite=True)
    
    def load(self, model):
        model.load_weights("model.weights.h5")
        self.reset_counter += 1

# Log nu_log and theta_log of the LRU layers
class LRULogger:
    def __call__(self, model):
        layers_ = {}
        c = 0
        for layer in model.base_model.layers:
            if isinstance(layer, lru.LRU_Block):
                nu_fw = layer.lru_fw.nu_log.numpy().tolist()
                nu_rv = layer.lru_rv.nu_log.numpy().tolist()
                theta_fw = layer.lru_fw.theta_log.numpy().tolist()
                theta_rv = layer.lru_rv.theta_log.numpy().tolist()
                layers_[c] = {"nu_fw":nu_fw, "nu_rv":nu_rv, "theta_fw":theta_fw, "theta_rv":theta_rv}
                c += 1
        return layers_

# Write different training metrics to a csv file
class CSVLogger:
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames

        # Open the file in write mode to initialize with the header
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames, delimiter=';')
            writer.writeheader()  # Write the header

    def __call__(self, data):
        # Open the file in append mode to write a new row
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames, delimiter=';')
            writer.writerow(data)
