import bc_util
import LRU_tf as lru

#from tf_seq2seq_losses import classic_ctc_loss

import time
from tqdm import tqdm

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
# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Optimization
tf.config.run_functions_eagerly(False)
tf.config.optimizer.set_jit(True)

# Fully print arrays with lengths up to 514
np.set_printoptions(threshold=514)

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
    
    #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))(x)
    #x = tf.keras.layers.Dense(2048)(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -5, 5))(x)
    #x = lru.LRU_Block(N=256, H=256, bidirectional=True, max_phase=2*pi/10)(x)
    
    x = tf.keras.layers.Conv1D(16, kernel_size=5, activation='swish', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=5, activation='swish', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=19, strides=6, activation='tanh', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = lru.LRU_Block(N=128, H=512, bidirectional=False, max_phase=2*pi)(x)
    x = bc_util.ReverseLayer(axis=1)(x)
    x = lru.LRU_Block(N=128, H=512, bidirectional=False, max_phase=2*pi)(x)
    x = bc_util.ReverseLayer(axis=1)(x)
    #x = lru.LRU_Block(N=128, H=1024, bidirectional=False, max_phase=2*pi)(x)
    #x = bc_util.ReverseLayer(axis=1)(x)
    #x = lru.LRU_Block(N=128, H=1024, bidirectional=False, max_phase=2*pi)(x)
    #x = bc_util.ReverseLayer(axis=1)(x)
    #x = lru.LRU_Block(N=128, H=1024, bidirectional=False, max_phase=2*pi)(x)
    
    # Output layer - logits for each time step
    classes = tf.keras.layers.Dense(NUM_CLASSES, dtype=tf.float32)(x)
    #logits = tf.keras.layers.Softmax()(classes)
    
    model = tf.keras.Model(inputs, classes)
    return model

# Compile and train the model
class CTCModel(tf.keras.Model):
    def __init__(self):
        super(CTCModel, self).__init__()
        self.base_model = build_model()
        
        # Gradient accumulation
        #self.gradient_accumulators = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.trainable_variables]
        #self.accumulation_counter = tf.Variable(0, trainable=False)
        #self.accumulation_steps = 1
    
    @tf.function
    def call(self, inputs):
        return self.base_model(inputs)
    
    @tf.function(autograph=True, input_signature=[
        tf.TensorSpec(shape=(None, 5000), dtype=tf.float32, name='chunks'),  # Input for chunk
        tf.SparseTensorSpec(shape=(None, None), dtype=tf.int32),  # Adjusted to SparseTensorSpec
        tf.TensorSpec(shape=(None,), dtype=tf.int16, name='target_lengths')  # Length of targets
    ])
    def train_step(self, chunks, targets, target_lengths):
        with tf.GradientTape() as tape:
            logits = self(chunks, training=True)
            input_lengths = tf.shape(logits)[1] * tf.ones([tf.shape(chunks)[0]], dtype=tf.int32)
            logits = tf.transpose(logits, perm=[1, 0, 2])
            
            # Compute CTC loss
            loss = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, logits=logits, label_length=target_lengths, logit_length=input_lengths, logits_time_major=True, blank_index=0))
            #loss = tf.keras.ops.ctc_loss(target=targets, output=logits, target_length=target_lengths, output_length=input_lengths)
            # Classic implementation
            #loss = tf.reduce_mean(classic_ctc_loss(labels=targets, logits=logits, label_length=target_lengths, logit_length=input_lengths, blank_index=0,))
            # Implementation from tf 1.
            #loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=input_lengths, time_major=True))
        
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Accumulate gradients
        #for i, grad in enumerate(gradients):
        #    self.gradient_accumulators[i].assign_add(grad)

        # Increment accumulation counter
        #self.accumulation_counter.assign_add(1)
        
        # Apply gradients if accumulation steps reached
        #if tf.equal(self.accumulation_counter, self.accumulation_steps):
        #    # Normalize accumulated gradients
        #    for i, grad_accum in enumerate(self.gradient_accumulators):
        #        self.gradient_accumulators[i].assign(grad_accum / self.accumulation_steps)

            # Apply normalized gradients
        #    self.optimizer.apply_gradients(zip(self.gradient_accumulators, self.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # Reset accumulators and counter after applying gradients
        #    for grad_accum in self.gradient_accumulators:
        #        grad_accum.assign(tf.zeros_like(grad_accum))
        #    self.accumulation_counter.assign(0)
        
        return {"loss": loss}
    
    @tf.function(autograph=True, input_signature=[
        tf.TensorSpec(shape=(None, 5000), dtype=tf.float32, name='chunks'),  # Input for chunk
        tf.SparseTensorSpec(shape=(None, None), dtype=tf.int32),  # Adjusted to SparseTensorSpec
        tf.TensorSpec(shape=(None,), dtype=tf.int16, name='target_lengths')  # Length of targets
    ])
    def test_step(self, chunks, targets, target_lengths):
        # For some reason the target_lengths in the train dataset are offset by 7
        #target_lengths = target_lengths + 7
        #targets = tf.sparse.to_dense(targets)
        
        logits = self(chunks, training=False)
        input_lengths = tf.shape(logits)[1] * tf.ones([tf.shape(chunks)[0]], dtype=tf.int32)
        logits = tf.transpose(logits, perm=[1, 0, 2])
        
        # Compute CTC loss
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, logits=logits, label_length=target_lengths, logit_length=input_lengths, logits_time_major=True, blank_index=0))
        
        return {"validation_loss": loss}

class CustomMetricsCallback:
    
    def __init__(self, model_summary):
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
        refs = [bc_util.decode_ref(self.remove_trailing(target, 0), ALPHABET) for target in val_targets]
        # decoded_sequences are numbers from 1 to 4, padded with -1
        seqs = [bc_util.decode_ref(self.remove_trailing(target, -1), ALPHABET) for target in decoded_sequences]
        
        accs = [bc_util.accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, seqs)]
        
        print(self.model_summary)
        # Print some of the decoded arrays
        #for i in range(1):    #(max(1, min(2, int(len(decoded_sequences)/8)))):
        #    seq = self.remove_trailing(decoded_sequences[i], -1)
        #    valid = self.remove_trailing(val_targets[i], 0)
        #    print("Original:", valid)
        #    print("Original length:", len(valid))
        #    print("Prediction:", seq)
        #    print("Prediction length:", len(seq))
        self.max_mean_accuracy = max(self.max_mean_accuracy, np.mean(accs))
        print(f"Max mean accuracy over training: {self.max_mean_accuracy:.5f}")
        print(f"mean_acc = {np.mean(accs):.5f}")
        print(f"median_acc = {np.median(accs):.5f}")
        print("Accuracies:", [f"{acc:.5f}" for acc in accs])
        print("This is GPU 1!")
        self.mean_accuracy = np.mean(accs)
        self.median_accuracy = np.median(accs)
    
    def decode_predictions(self, logits):
        # Get the length of each sequence in the batch
        input_lengths = np.ones(logits.shape[0]) * logits.shape[1]
        
        # Decode the predictions
        decoded_sequences = tf.keras.ops.ctc_decode(logits, sequence_lengths=input_lengths, strategy="beam_search", beam_width=128)
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


with tf.device('/GPU:0'):

    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-18)
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=2e-5, decay_steps=EPOCHS*STEPS_PER_EPOCH, alpha=0.001, warmup_target=8e-5, warmup_steps=4000), clipnorm=1, weight_decay=1e-4)
    #optimizer = tf.keras.optimizers.AdamW(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=8e-6, decay_steps=16000, decay_rate=0.95), clipnorm=1, weight_decay=1e-4)
    
    # Initialize the model and optimizer
    model = CTCModel()
    model.compile(optimizer=optimizer)
    
    model_summary = bc_util.model_summary_to_string(model)
    print(model_summary)
    
    # Read in the dataset
    train_dataset = bc_util.load_tfrecords(directory + "train_data.tfrecord", batch_size=BATCH_SIZE, shuffle=True, repeat=True)
    valid_dataset = bc_util.load_tfrecords(directory + "valid_data.tfrecord", batch_size=BATCH_SIZE, shuffle=False, repeat=True)
    
    # Fit the model using the dataset
    #model.fit(train_dataset, shuffle=False, validation_data=valid_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=1, callbacks=[CustomMetricsCallback(model_summary, valid_dataset)])
    
    callback = CustomMetricsCallback(model_summary)
    
    train_loss_results = []
    val_loss_results = []
    val_mean_accuracy = []
    val_median_accuracy = []
    
    # Custom training loop
    train_iter = iter(train_dataset)
    valid_iter = iter(valid_dataset)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = 0.0
        val_loss = 0.0
        with tqdm(total=STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}", unit="step") as pbar:
            for step in range(STEPS_PER_EPOCH):
                start_time = time.time()
                
                train_batch = next(train_iter)
                # Unpack the batch
                chunks, targets, target_lengths = train_batch
                
                # Training step
                train_metrics = model.train_step(chunks, targets, target_lengths)
                loss = train_metrics['loss'].numpy()
                
                # Calculate the elapsed time for this step
                step_time = time.time() - start_time
                
                train_loss += loss
                
                # Update the progress bar with loss and time
                pbar.set_postfix(loss=f"{loss:.2f}", step_time=f"{step_time*1000:.0f} ms")
                pbar.update(1)  # Increment the progress bar by 1 step
                
        # Average the training loss
        train_loss /= STEPS_PER_EPOCH
        train_loss_results.append(train_loss)
        
        print(f"Train Loss: {train_loss:.6f}")
        
        val_batch = next(valid_iter)
        
        val_chunks, val_targets, val_target_lengths = val_batch
        
        # Perform a validation step
        val_metrics = model.test_step(val_chunks, val_targets, val_target_lengths)
        val_loss = val_metrics['validation_loss'].numpy()
        
        val_loss_results.append(val_loss)
        
        print(f"Validation Loss: {val_loss:.6f}")
        
        prediction = model.predict(val_chunks)
        callback.on_epoch_end(prediction, val_targets)
        
        val_mean_accuracy.append(callback.mean_accuracy)
        val_median_accuracy.append(callback.median_accuracy)
        
