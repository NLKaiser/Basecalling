import bc_util

import LRU_tf as lru

import numpy as np
import tensorflow as tf

import json
import csv

class Metrics:
    
    def __init__(self, model_summary):
        self.alphabet = {1: "A", 2: "C", 3: "G", 4: "T"}
        self.model_summary = model_summary
        
        self.accuracy_original = [0]
        self.accuracy_global = [0]
        self.accuracy_pairwise = [0]
        
        self.max_mean_accuracy = 0
    
    def batch_accuracy(self, decoded, targets, original=True, global_=True, pairwise=True):
        val_targets = targets.numpy()
        
        # val_targets are numbers from 1 to 4, padded with 0
        refs = ["".join(self.alphabet[i] for i in target if i in self.alphabet)
        for target in val_targets]
        
        if original:
            accs_original = [bc_util.accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, decoded)]
            self.accuracy_original = accs_original
        if global_:
            accs_global = [bc_util.accuracy_global(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, decoded)]
            self.accuracy_global = accs_global
        if pairwise:
            accs_pairwise = [bc_util.accuracy_pairwise(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, decoded)]
            self.accuracy_pairwise = accs_pairwise
    
    def print_statistics(self, type_, stats, summary=True, original=True, global_=True, pairwise=True):
        # Print the model stats to see the number of parameters
        if summary:
            print(self.model_summary)
        # Training statistics
        if type_ == "training":
            print("Trainings statistics:")
            print(f"Loss: {stats['loss']:.2f}")
        # Validation statistics
        if type_ == "validation":
            print("Validation statistics:")
            print(f"Loss: {stats['loss']:.2f}")
            if original:
                print(f"Mean accuracy original: {stats['mean_accuracy_original']:.2f}")
                print(f"Median accuracy original: {stats['median_accuracy_original']:.2f}")
            if global_:
                print(f"Mean accuracy global: {stats['mean_accuracy_global']:.2f}")
                print(f"Median accuracy global: {stats['median_accuracy_global']:.2f}")
            if pairwise:
                print(f"Mean accuracy pairwise: {stats['mean_accuracy_pairwise']:.2f}")
                print(f"Median accuracy pairwise: {stats['median_accuracy_pairwise']:.2f}")
    
    # Return one prediction and reference globally aligned and as it is done in bonito
    def alignment_comparison(self, prediction, reference):
        #reference = tf.sparse.to_dense(reference, default_value=0)
        reference = reference.numpy()[0]
        reference = "".join(self.alphabet[i] for i in reference if i in self.alphabet)
        prediction = prediction[0]
        alignment_original = bc_util.alignment_local(prediction, reference)
        alignment_global = bc_util.alignment_global(prediction, reference)
        try:
            return {"pred_original":alignment_original.traceback.query, "ref_original":alignment_original.traceback.ref,
                    "pred_global":alignment_global.traceback.query, "ref_global":alignment_global.traceback.ref}
        except:
            return {"pred_original":"-", "ref_original":"-",
                    "pred_global":"-", "ref_global":"-"}

class ModelReset:
    def __init__(self, model, reset_counter=0):
        model.save_weights("model.weights.h5", overwrite=True)
        self.reset_counter = reset_counter
    
    def save(self, model):
        model.save_weights("model.weights.h5", overwrite=True)
    
    def load(self, model):
        model.load_weights("model.weights.h5")
        self.reset_counter += 1

class LRULogger:
    def __call__(self, model):
        layers_ = {}
        c = 0
        for layer in model.model_architecture.layers:
            if isinstance(layer, lru.LRU_Block):
                nu_fw = layer.lru_fw.nu_log.numpy().tolist()
                try:
                    nu_rv = layer.lru_rv.nu_log.numpy().tolist()
                except:
                    nu_rv = nu_fw
                theta_fw = layer.lru_fw.theta_log.numpy().tolist()
                try:
                    theta_rv = layer.lru_rv.theta_log.numpy().tolist()
                except:
                    theta_rv = theta_fw
                layers_[c] = {"nu_fw":nu_fw, "nu_rv":nu_rv, "theta_fw":theta_fw, "theta_rv":theta_rv}
                c += 1
        return layers_

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
