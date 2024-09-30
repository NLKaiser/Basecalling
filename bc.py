import bc_util

import LRU_tf as lru

#from tf_seq2seq_losses import classic_ctc_loss

import numpy as np
import tensorflow as tf
#import keras

#import logging
#import warnings

# Suppress warnings
#logger = tf.get_logger()
#logger.setLevel(logging.ERROR)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#warnings.filterwarnings("ignore")

#tf.debugging.set_log_device_placement(True)

# Print available GPUs
print("GPU:", tf.config.list_physical_devices('GPU'))

# Fully print arrays with lengths up to 514
np.set_printoptions(threshold=514)

# Necessary for LRU and CTC compatibility
tf.config.run_functions_eagerly(True)

# Optimization
tf.config.optimizer.set_jit(True)

# Data directory
directory = "./"
# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 600
STEPS_PER_EPOCH = 1000
CHUNK_LENGTH = 5000
TARGET_LENGTH = 500
ALPHABET = ["N", "A", "C", "G","T"]
NUM_CLASSES = 5  # Including padding (0) and base classes (1-4)

# pi for LRU layer
pi = 3.14

# The model structure
def build_model():
    inputs = tf.keras.Input(shape=(CHUNK_LENGTH, 1))  # Input shape is (5000, 1) for sensor readings
    #x = tf.keras.layers.Conv1D(32, kernel_size=1, padding='same')(inputs)
    #x = tf.keras.layers.Conv1D(16, kernel_size=5, activation='swish', padding='same')(inputs)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Conv1D(16, kernel_size=5, activation='swish', padding='same')(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Conv1D(256, kernel_size=19, strides=1, activation='tanh', padding='same')(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))(x)
    #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))(x)
    #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))(x)
    #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))(x)
    #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))(x)
    #x = tf.keras.layers.Dense(2048)(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -5, 5))(x)
    #x = lru.LRU_Block(N=256, H=256, bidirectional=True, max_phase=2*pi/10)(x)
    #x = lru.LRU_Block(N=256, H=256, bidirectional=True, max_phase=2*pi/10)(x)
    #x = lru.LRU_Block(N=256, H=256, bidirectional=True, max_phase=2*pi/10)(x)
    #x = lru.LRU_Block(N=256, H=256, bidirectional=True, max_phase=2*pi/10)(x)
    #x = lru.LRU_Block(N=256, H=256, bidirectional=True, max_phase=2*pi/10)(x)
    #x = tf.keras.layers.Dense(1024)(x)
    
    x = tf.keras.layers.Conv1D(16, kernel_size=5, activation='swish', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=5, activation='swish', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(1024, kernel_size=19, strides=6, activation='tanh', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = lru.LRU_Block(N=64, H=512, bidirectional=True, max_phase=2*pi)(x)
    x = lru.LRU_Block(N=64, H=512, bidirectional=True, max_phase=2*pi)(x)
    x = lru.LRU_Block(N=64, H=512, bidirectional=True, max_phase=2*pi)(x)
    x = lru.LRU_Block(N=64, H=512, bidirectional=True, max_phase=2*pi)(x)
    x = lru.LRU_Block(N=64, H=512, bidirectional=True, max_phase=2*pi)(x)
    
    # Output layer - logits for each time step
    classes = tf.keras.layers.Dense(NUM_CLASSES)(x)
    #logits = tf.keras.layers.Softmax()(classes)
    
    model = tf.keras.Model(inputs, classes)
    return model

# Compile and train the model
class CTCModel(tf.keras.Model):
    def __init__(self):
        super(CTCModel, self).__init__()
        self.base_model = build_model()
        # Gradient accumulation
        self.gradient_accumulators = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.trainable_variables]
        self.accumulation_counter = tf.Variable(0, trainable=False)
        self.accumulation_steps = 1
    
    def call(self, inputs):
        return self.base_model(inputs)
    
    @tf.function
    def train_step(self, data):
        chunks, targets, target_lengths = data
        # For some reason the target_lengths in the train dataset are offset by 7
        target_lengths = target_lengths + 7
        
        #targets = tf.sparse.to_dense(targets)
        
        with tf.GradientTape() as tape:
            logits = self(chunks, training=True)
            input_lengths = tf.shape(logits)[1] * tf.ones([tf.shape(chunks)[0]], dtype=tf.int32)
            logits = tf.transpose(logits, perm=[1, 0, 2])
            
            # Compute CTC loss
            loss = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, logits=logits, label_length=target_lengths, logit_length=input_lengths, logits_time_major=True, blank_index=0))
            
            #loss = tf.reduce_mean(classic_ctc_loss(labels=targets, logits=logits, label_length=target_lengths, logit_length=input_lengths, blank_index=0,))
            
            #loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=input_lengths, time_major=True))
        
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Accumulate gradients
        for i, grad in enumerate(gradients):
            self.gradient_accumulators[i].assign_add(grad)

        # Increment accumulation counter
        self.accumulation_counter.assign_add(1)
        
        # Apply gradients if accumulation steps reached
        if tf.equal(self.accumulation_counter, self.accumulation_steps):
            # Normalize accumulated gradients
            for i, grad_accum in enumerate(self.gradient_accumulators):
                self.gradient_accumulators[i].assign(grad_accum / self.accumulation_steps)

            # Apply normalized gradients
            self.optimizer.apply_gradients(zip(self.gradient_accumulators, self.trainable_variables))

            # Reset accumulators and counter after applying gradients
            for grad_accum in self.gradient_accumulators:
                grad_accum.assign(tf.zeros_like(grad_accum))
            self.accumulation_counter.assign(0)
        
        return {"loss": loss}
    
    @tf.function
    def test_step(self, data):
        chunks, targets, target_lengths = data
        # For some reason the target_lengths in the train dataset are offset by 7
        target_lengths = target_lengths + 7
        
        #targets = tf.sparse.to_dense(targets)
        
        logits = self(chunks, training=False)
        input_lengths = tf.shape(logits)[1] * tf.ones([tf.shape(chunks)[0]], dtype=tf.int32)
        logits = tf.transpose(logits, perm=[1, 0, 2])
        
        # Compute CTC loss
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, logits=logits, label_length=target_lengths, logit_length=input_lengths, logits_time_major=True, blank_index=0))
        
        # Classic implementation
        #loss = tf.reduce_mean(classic_ctc_loss(labels=targets, logits=logits, label_length=target_lengths, logit_length=input_lengths, blank_index=0,))
        # Implementation from tf 1.
        #loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(labels=targets, logits=logits, sequence_length=input_lengths))
        
        return {"validation_loss": loss}

class CustomMetricsCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, valid_dataset):
        super().__init__()
        self.valid_dataset = valid_dataset.repeat()
        # Create an iterator from the dataset
        self.iterator = iter(valid_dataset)
    
    def on_epoch_end(self, epoch, logs=None):
        val_batch = next(self.iterator)
        val_inputs, val_targets, val_target_lengths = val_batch
        val_targets = tf.sparse.to_dense(val_targets, default_value=0)
        val_targets = val_targets.numpy()
        
        # Get predictions
        logits = self.model.predict(val_inputs, verbose=0)
        
        # Decode the predictions
        decoded_sequences = self.decode_predictions(logits)
        
        # Print some of the decoded arrays
        for i in range(max(1, int(len(decoded_sequences)/8))):
            seq = self.remove_trailing(decoded_sequences[i], -1)
            valid = self.remove_trailing(val_targets[i], 0)
            print("original:", valid)
            print("original length:", len(valid))
            print("prediction:", seq)
            print("prediction length:", len(seq))
        
        # val_targets are numbers from 1 to 4, padded with 0
        refs = [bc_util.decode_ref(self.remove_trailing(target, 0), ALPHABET) for target in val_targets]
        # decoded_sequences are numbers from 1 to 4, padded with -1
        seqs = [bc_util.decode_ref(self.remove_trailing(target, -1), ALPHABET) for target in decoded_sequences]
        
        accs = [bc_util.accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, seqs)]
        
        print(f"Epoch {int(epoch) + 1}:")
        print(f"val_loss = {logs.get('val_validation_loss')}")
        print("mean_acc =", np.mean(accs))
        print("median_acc =", np.median(accs))
        print("Accuracies:", [f"{acc:.5f}" for acc in accs])
        print("This is GPU 1!")
    
    def decode_predictions(self, logits):
        # Get the length of each sequence in the batch
        input_lengths = np.ones(logits.shape[0]) * logits.shape[1]
        
        # Decode the predictions
        #decoded_sequences = tf.keras.backend.ctc_decode(logits, input_length=input_lengths, greedy=False, beam_width=128)[0][0]
        
        decoded_sequences = tf.keras.ops.ctc_decode(logits, sequence_lengths=input_lengths, strategy="beam_search", beam_width=1024)
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


#model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=bc_util.get_learning_rate(epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH), weight_decay=1e-4, clipnorm=1.0))
#model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=bc_util.CosineDecaySchedule(initial_lr=0.005, final_lr=1e-8, decay_steps=EPOCHS*STEPS_PER_EPOCH), momentum=0.9))
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

#optimizer = tf.keras.optimizers.AdamW(learning_rate=tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-8, decay_steps=EPOCHS*STEPS_PER_EPOCH, warmup_target=1e-4, warmup_steps=10000))

optimizer = tf.keras.optimizers.AdamW(learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=8000, t_mul=1.2, m_mul=0.9, alpha=0), weight_decay=0)

#optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-7)

#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-18)

with tf.device('/GPU:0'):
    # Initialize the model and optimizer
    model = CTCModel()
    model.compile(optimizer=optimizer)
    
    model.summary()
    
    # Read in the dataset
    train_dataset = bc_util.load_tfrecords(directory + "train_data.tfrecord", batch_size=BATCH_SIZE, shuffle=True)
    valid_dataset = bc_util.load_tfrecords(directory + "valid_data.tfrecord", batch_size=BATCH_SIZE, shuffle=False)
    
    # Fit the model using the dataset
    model.fit(train_dataset, shuffle=False, validation_data=valid_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=1, callbacks=[CustomMetricsCallback(valid_dataset)])
