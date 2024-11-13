import bc_util
import data
import layers
import schedulers
import callbacks

import LRU_tf as lru

import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf

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

# Mixed precision
#tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Optimization
#tf.config.run_functions_eagerly(False)
tf.config.optimizer.set_jit(True)

# Fully print arrays with lengths up to 514
np.set_printoptions(threshold=514)

# Data directory
directory = "./"
# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 5120
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
    
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    x = tf.keras.layers.Conv1D(16, kernel_size=5, activation='swish', padding='same', use_bias=True)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=5, activation='swish', padding='same', use_bias=True)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
    x = tf.keras.layers.Conv1D(384, kernel_size=19, strides=6, activation='tanh', padding='same', use_bias=True)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
    tf.keras.mixed_precision.set_global_policy('float32')
    
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/100, dropout=0.9)(x)
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/100, dropout=0.9)(x)
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/100, dropout=0.9)(x)
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/100, dropout=0.9)(x)
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/100, dropout=0.9)(x)
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/100, dropout=0.9)(x)
    
    # Output layer - logits for each time step
    classes = tf.keras.layers.Dense(NUM_CLASSES, use_bias=True, dtype=tf.float32)(x)
    classes = layers.ClipLayer(-5, 5)(classes)
    
    model = tf.keras.Model(inputs, classes)
    return model

# Compile and train the model
class CTCModel(tf.keras.Model):
    def __init__(self):
        super(CTCModel, self).__init__()
        self.base_model = build_model()
    
    @tf.function
    def call(self, inputs):
        return self.base_model(inputs)
    
    @tf.function(autograph=True)
    def train_step(self, chunks, targets, target_lengths):
        with tf.GradientTape() as tape:
            logits = self(chunks, training=True)
            # Time dimension of the models output * batch size
            input_lengths = tf.shape(logits)[1] * tf.ones([tf.shape(chunks)[0]], dtype=tf.int32)
            # Logits time major
            logits = tf.transpose(logits, perm=[1, 0, 2])
            
            # Compute CTC loss
            loss = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, logits=logits, label_length=target_lengths, logit_length=input_lengths, logits_time_major=True, blank_index=0))
        
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": loss}
    
    @tf.function(autograph=True)
    def test_step(self, chunks, targets, target_lengths):
        logits = self(chunks, training=False)
        input_lengths = tf.shape(logits)[1] * tf.ones([tf.shape(chunks)[0]], dtype=tf.int32)
        logits = tf.transpose(logits, perm=[1, 0, 2])
        
        # Compute CTC loss
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, logits=logits, label_length=target_lengths, logit_length=input_lengths, logits_time_major=True, blank_index=0))
        
        return {"validation_loss": loss}


with tf.device('/GPU:0'):
    
    scheduler = schedulers.WarmUpCosineDecayWithRestarts(
        warmup_initial=1e-8, warmup_end=1e-5, warmup_steps=400000,
        initial_learning_rate=2e-4, decay_steps=4200000, alpha=0.45, t_mul=1.5, m_mul=0.96)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=scheduler, weight_decay=0.005)
    
    # Initialise the model and optimizer
    model = CTCModel()
    model.compile(optimizer=optimizer)
    
    model_summary = bc_util.model_summary_to_string(model)
    print(model_summary)
    
    # Read in the dataset
    train_dataset = data.load_tfrecords(directory + "train_data.tfrecord", batch_size=BATCH_SIZE, shuffle=True, repeat=True)
    train_iter = iter(train_dataset)
    valid_dataset = data.load_tfrecords(directory + "valid_data.tfrecord", batch_size=BATCH_SIZE, shuffle=False, repeat=True)
    valid_iter = iter(valid_dataset)
    
    # Fit the model using the dataset
    #model.fit(train_dataset, shuffle=False, validation_data=valid_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=1, callbacks=[CustomMetricsCallback(model_summary, valid_dataset)])
    
    # Initialise callbacks
    metrics = callbacks.Metrics(ALPHABET, model_summary)
    model_reset = callbacks.ModelReset(model)
    lru_logger = callbacks.LRULogger()
    lru_values = lru_logger(model)
    csv_logger = callbacks.CSVLogger("training.csv", ["epoch", "train_loss", "val_loss", "val_mean_accuracy", "val_median_accuracy", "learning_rate", "lru_values"])
    csv_logger({"epoch":0, "train_loss":1000, "val_loss":1000, "val_mean_accuracy":0, "val_median_accuracy":0, "learning_rate":0, "lru_values":lru_values})
    
    train_loss_results = []
    val_loss_results = []
    val_mean_accuracy = []
    val_median_accuracy = []
    
    # Custom training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = 0.0
        val_loss = 0.0
        with tqdm(total=STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}", unit="step") as pbar:
            for step in range(STEPS_PER_EPOCH):
                start_time = time.time()
                
                train_batch = next(train_iter)
                # Get one batch
                chunks, targets, target_lengths = train_batch
                
                # Training step
                train_metrics = model.train_step(chunks, targets, target_lengths)
                loss = train_metrics['loss'].numpy()
                
                if np.isnan(loss):
                    print(f"NaN detected in training loss at step {step}, restoring previous weights.")
                    model_reset.load(model)
                    continue
                
                # Calculate the elapsed time for this step
                step_time = time.time() - start_time
                
                train_loss += loss
                
                # Update the progress bar with loss and time
                pbar.set_postfix(loss=f"{loss:.2f}", step_time=f"{step_time*1000:.0f} ms")
                pbar.update(1)  # Increment the progress bar by 1 step
        
        model_reset.save(model)
                
        # Average the training loss
        train_loss /= STEPS_PER_EPOCH
        train_loss_results.append(train_loss)
        
        print(f"Train Loss: {train_loss:.2f}")
        
        val_batch = next(valid_iter)
        
        val_chunks, val_targets, val_target_lengths = val_batch
        
        # Perform a validation step
        val_metrics = model.test_step(val_chunks, val_targets, val_target_lengths)
        val_loss = val_metrics['validation_loss'].numpy()
        
        val_loss_results.append(val_loss)
        
        print(f"Validation Loss: {val_loss:.2f}")
        
        print("Resets:", model_reset.reset_counter)
        
        print(f"Loss: {train_loss:.2f}")
        print(f"Min Loss: {min(train_loss_results):.2f}")
        print("Last 10 Losses:", [f"{l:.2f}" for l in train_loss_results[-10:]])
        
        prediction = model.predict(val_chunks)
        metrics.on_epoch_end(prediction, val_targets)
        
        val_mean_accuracy.append(metrics.mean_accuracy)
        val_median_accuracy.append(metrics.median_accuracy)
        
        lr = scheduler((epoch+1)*STEPS_PER_EPOCH).numpy()
        
        # Log the nu_log and theta_log of each lru layer
        lru_values = lru_logger(model)
        
        csv_logger({"epoch":epoch+1, "train_loss":train_loss, "val_loss":val_loss, "val_mean_accuracy":metrics.mean_accuracy, "val_median_accuracy":metrics.median_accuracy, "learning_rate":lr, "lru_values":lru_values})
        
        print("Learning rate:", lr)
