import bc_util
import data
import layers
import schedulers
import callbacks

import LRU_tf as lru

import CTC
import CTC_HMM
import CTC_2_HMM
import CRF_HMM

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
BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 512
STEPS_PER_EPOCH = int(1000000/BATCH_SIZE)
CHUNK_LENGTH = 5000
TARGET_LENGTH = 500
NUM_CLASSES = 5  # Including padding (0) and base classes (1-4)

MODEL_TYPE = CTC_HMM.HMM(BATCH_SIZE, 500, 834, 0, epsilon=-tf.float32.max)
MODEL_VALID_TYPE = CTC_HMM.HMM(VALID_BATCH_SIZE, 500, 834, 0, epsilon=-tf.float32.max)
#MODEL_TYPE = CTC_2_HMM.HMM(BATCH_SIZE, 500, 834, epsilon=-tf.float32.max)
#MODEL_VALID_TYPE = CTC_2_HMM.HMM(VALID_BATCH_SIZE, 500, 834, epsilon=-tf.float32.max)
#MODEL_TYPE = CTC.CTC()
#MODEL_VALID_TYPE = CTC.CTC()

# pi for LRU layer
pi = 3.14

# The model structure
def build_model_architecture():
    inputs = tf.keras.Input(shape=(None, 1))  # Input shape is (5000, 1) for sensor readings
    
    #tf.keras.mixed_precision.set_global_policy('mixed_float16')
    x = tf.keras.layers.Conv1D(16, kernel_size=5, activation='swish', padding='same', use_bias=True)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=5, activation='swish', padding='same', use_bias=True)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
    x = tf.keras.layers.Conv1D(512, kernel_size=19, strides=6, activation='tanh', padding='same', use_bias=True)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
    #tf.keras.mixed_precision.set_global_policy('float32')
    
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/3, r_min=0.9, r_max=0.98, dropout=0)(x)
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/3, r_min=0.68, r_max=0.75, dropout=0)(x)
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/3, r_min=0.68, r_max=0.75, dropout=0)(x)
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/3, r_min=0.68, r_max=0.75, dropout=0)(x)
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/3, r_min=0.68, r_max=0.75, dropout=0)(x)
    x = lru.LRU_Block(N=256, H=1024, bidirectional=True, max_phase=2*pi/4, r_min=0.68, r_max=0.75, dropout=0)(x)
    
    # Output layer - logits for each time step
    classes = tf.keras.layers.Dense(NUM_CLASSES, use_bias=True, dtype=tf.float32)(x)
    #classes = layers.LinearCRFEncoder(n_base=4, state_len=5, blank_score=-2.0)(x)
    classes = layers.ClipLayer(-5, 5)(classes)
    
    model = tf.keras.Model(inputs, classes)
    return model

def make_transformer_seq_delta(
    num_features: int,        # NUM_CLASSES
    time_steps: int,          # explicitly define T here
    model_dim: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.0
) -> tf.keras.Model:
    """
    Takes input of shape (batch, T, C)
    and produces an output of the same shape: a sequence of per-time-step deltas.
    """
    # 1) fixed-length sequence input
    seq_in = tf.keras.layers.Input(shape=(time_steps, num_features), 
                          name="posterior_sequence")

    # 2) embed into model_dim
    x = tf.keras.layers.Dense(model_dim, name="embed_proj")(seq_in)

    # 3) add learned positional embeddings
    pos_emb = tf.keras.layers.Embedding(input_dim=time_steps,
                               output_dim=model_dim,
                               name="pos_emb")
    # build a position index tensor of shape [1, T]
    positions = tf.range(start=0, limit=time_steps, delta=1)[tf.newaxis, :]
    x = x + pos_emb(positions)

    # 4) stack transformer encoder blocks
    for i in range(num_layers):
        attn = tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=model_dim // num_heads,
                    dropout=dropout,
                    name=f"mha_{i}"
               )(x, x)
        x = tf.keras.layers.LayerNormalization(name=f"ln1_{i}")(x + attn)

        ff = tf.keras.layers.Dense(ff_dim, activation="relu", name=f"ff1_{i}")(x)
        ff = tf.keras.layers.Dense(model_dim, name=f"ff2_{i}")(ff)
        x = tf.keras.layers.LayerNormalization(name=f"ln2_{i}")(x + ff)

    # 5) project back to C *per time step*
    delta_seq = tf.keras.layers.Dense(num_features,
                             activation=None,
                             name="delta_sequence")(x)
    
    delta_seq = layers.ClipLayer(-5, 5)(delta_seq)
    
    return tf.keras.Model(seq_in, delta_seq, name="transformer_seq_delta_head")

class ThreeTransitions(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.transitions = self.add_weight(
            name="transitions",
            shape=(3,),
            initializer=tf.constant_initializer(value=0.0),
            trainable=True
        )

    def call(self, inputs=None):
        # returns a Tensor, not a Variable
        return tf.identity(self.transitions)

# Compile and train the model
class Model(tf.keras.Model):
    def __init__(self, architecture_, type_, valid_type_):
        super(Model, self).__init__()
        self.model_architecture = architecture_
        self.transitions = ThreeTransitions()
        self.MLP = make_transformer_seq_delta(NUM_CLASSES, 834, 256, 8, 256, 4)
        self.model_type = type_
        self.model_valid_type = valid_type_
        self.scale = False
        self.loss_scale = False
        self.loss_scale_factor = 1024
    
    @tf.function
    def call(self, inputs):
        return self.model_architecture(inputs)
    
    @tf.function
    def train_step(self, chunks, targets, target_lengths):
        with tf.GradientTape() as tape:
            logits = self(chunks, training=True)
            # Time dimension of the models output * batch size
            input_lengths = tf.shape(logits)[1] * tf.ones([tf.shape(chunks)[0]], dtype=tf.int32)
            
            # Residual HMM
            logits_posteriors = self.model_type.get_middle_HMM_logit_posteriors(logits, self.transitions())
            logits_posteriors_refined = self.MLP(logits_posteriors)
            logits = (logits + logits_posteriors_refined) / 2
            
            loss = tf.reduce_mean(self.model_type(targets, logits, target_lengths, input_lengths))
            
            if self.loss_scale:
                loss = loss * self.loss_scale_factor
        
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        if self.loss_scale:
            gradients = [grad / self.loss_scale_factor if grad is not None else None for grad in gradients]
            loss = loss / self.loss_scale_factor
        
        # Optional scaling
        if self.scale:
            #to_scale = ["nu_log", "theta_log", "B_re", "B_im", "C_re", "C_im"]
            to_scale = ["nu_log", "theta_log"]
            gradients = [
                grad * 10 if var.name in to_scale else grad
                for grad, var in zip(gradients, self.trainable_variables)
            ]
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": loss}
    
    @tf.function(autograph=True)
    def test_step(self, chunks, targets, target_lengths):
        logits = self(chunks, training=False)
        input_lengths = tf.shape(logits)[1] * tf.ones([tf.shape(chunks)[0]], dtype=tf.int32)
        
        # Compute the loss
        loss = tf.reduce_mean(self.model_valid_type(targets, logits, target_lengths, input_lengths))
        
        return {"validation_loss": loss}
    
    @tf.function
    def predict(self, chunks):
        logits = self(chunks, training=False)
        return logits
    
    def decode(self, chunks):
        logits = self(chunks, training=False)
        return self.model_type.decode(logits)

with tf.device('/GPU:0'):
    
    scheduler = schedulers.WarmUpCosineDecayWithRestarts(
        warmup_initial=8e-5, warmup_end=2e-4, warmup_steps=31250,
        initial_learning_rate=2e-4, decay_steps=10*31250, alpha=0.6, t_mul=0.06, m_mul=0.5)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=scheduler, weight_decay=0.002)
    
    # Initialise the model and optimizer
    architecture = build_model_architecture()
    model = Model(architecture, MODEL_TYPE, MODEL_VALID_TYPE)
    model.build((None,1))
    model.compile(optimizer=optimizer)
    
    # Read in the dataset
    train_dataset = data.load_tfrecords(directory + "train_data.tfrecord", batch_size=BATCH_SIZE, shuffle=True, repeat=True)
    train_iter = iter(train_dataset)
    valid_dataset = data.load_tfrecords(directory + "valid_data.tfrecord", batch_size=VALID_BATCH_SIZE, shuffle=False, repeat=False)
    valid_dataset_length = sum(1 for _ in iter(valid_dataset))
    
    model_summary = bc_util.model_summary_to_string(model)
    print(model_summary)
    
    # Initialise callbacks
    metrics = callbacks.Metrics(model_summary)
    model_reset = callbacks.ModelReset(model)
    lru_logger = callbacks.LRULogger()
    lru_values = lru_logger(model)
    csv_logger = callbacks.CSVLogger("training.csv", ["epoch", "train_loss", "val_loss", "val_mean_accuracy", "val_median_accuracy", "learning_rate", "lru_values", "alignments", "logits", "valid_targets"])
    csv_logger({"epoch":0, "train_loss":0, "val_loss":0, "val_mean_accuracy":0, "val_median_accuracy":0, "learning_rate":0, "lru_values":lru_values, "alignments":"", "logits":"", "valid_targets":""})
    
    train_loss_results = []
    val_loss_results = []
    val_mean_accuracy = []
    val_median_accuracy = []
    
    def validation_step(batch, original, global_, pairwise):
        val_chunks, val_targets, val_target_lengths = batch
        
        # Calculate validation loss
        val_metrics = model.test_step(val_chunks, val_targets, val_target_lengths)
        try:
            val_loss = val_metrics['validation_loss'].numpy()
        except:
            val_loss = 1000
        
        decoded = model.decode(val_chunks)
        # Calculate validation accuracy
        metrics.batch_accuracy(decoded, val_targets, original=original, global_=global_, pairwise=pairwise)
        
        accuracy_original = metrics.accuracy_original
        accuracy_global = metrics.accuracy_global
        accuracy_pairwise = metrics.accuracy_pairwise
        
        return val_loss, accuracy_original, accuracy_global, accuracy_pairwise
        
    def validate_epoch(original, global_, pairwise):
        valid_iter = iter(valid_dataset)
        losses = []
        accuracy_original = []
        accuracy_global = []
        accuracy_pairwise = []
        with tqdm(total=valid_dataset_length, desc="Validation", unit="batch") as pbar:
            for batch in valid_iter:
                start_time = time.time()
                l, ao, ag, ap = validation_step(batch, original, global_, pairwise)
                losses.append(l)
                accuracy_original.extend(ao)
                accuracy_global.extend(ag)
                accuracy_pairwise.extend(ap)
                batch_time = time.time() - start_time
                # Update the progress bar
                pbar.set_postfix(loss=f"{l:.2f}", batch_time=f"{batch_time*1000:.0f} ms")
                pbar.update(1)  # Increment the progress bar by 1 step
        return (np.mean(losses),
                np.mean(accuracy_original), np.median(accuracy_original),
                np.mean(accuracy_global), np.median(accuracy_global),
                np.mean(accuracy_pairwise), np.median(accuracy_pairwise))
    
    # Custom training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = 0.0
        val_loss = 0.0
        model_resetable = True
        with tqdm(total=STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}", unit="step") as pbar:
            for step in range(STEPS_PER_EPOCH):
                start_time = time.time()
                
                train_batch = next(train_iter)
                # Get one batch
                chunks, targets, target_lengths = train_batch
                
                # Training step
                train_metrics = model.train_step(chunks, targets, target_lengths)
                try:
                    loss = train_metrics['loss'].numpy()
                except:
                    loss = 1000
                
                if np.isnan(loss):
                    print(f"NaN detected in training loss at step {step}, restoring previous weights.")
                    model_reset.load(model)
                    model_resetable = False
                    continue
                
                step_time = time.time() - start_time
                
                train_loss += loss
                
                # Update the progress bar with loss and time
                pbar.set_postfix(loss=f"{loss:.2f}", step_time=f"{step_time*1000:.0f} ms")
                pbar.update(1)  # Increment the progress bar by 1 step
        
        # Save the model after each epoch if it has not been resetted that epoch
        if model_resetable:
            model_reset.save(model)
        print("Resets:", model_reset.reset_counter)
        
        # Average the training loss
        train_loss /= STEPS_PER_EPOCH
        train_loss_results.append(train_loss)
        
        # Print training statistics
        metrics.print_statistics("training", {"loss":train_loss}, summary=True)
        print(f"Min Loss: {min(train_loss_results):.2f}")
        print("Last 10 Losses:", [f"{l:.2f}" for l in train_loss_results[-10:]])
        
        # Validation
        val_loss, mean_accuracy_original, median_accuracy_original, mean_accuracy_global, median_accuracy_global, mean_accuracy_pairwise, median_accuracy_pairwise = validate_epoch(original=True, global_=False, pairwise=False)
        val_loss_results.append(val_loss)
        # Print validation statistics
        metrics.print_statistics("validation", {"loss":val_loss, "mean_accuracy_original":mean_accuracy_original, "median_accuracy_original":median_accuracy_original, "mean_accuracy_global":mean_accuracy_global, "median_accuracy_global":median_accuracy_global, "mean_accuracy_pairwise":mean_accuracy_pairwise, "median_accuracy_pairwise":median_accuracy_pairwise}, summary=False, original=True, global_=False, pairwise=False)
        
        # Get four elements from the validation dataset
        validation_elements = bc_util.get_iter_elements(iter(valid_dataset), [0, 256, 1024, 1486])
        
        # Alignment
        alignments_list = []
        for valid_elem in validation_elements:
            val_chunks = valid_elem[0]
            val_targets = valid_elem[1]
            alignments_list.append(metrics.alignment_comparison(model.decode(val_chunks), val_targets))
        alignments_list = str(alignments_list)
        
        # Logits
        logits_list = []
        for valid_elem in validation_elements:
            val_chunks = valid_elem[0]
            logits = model.predict(val_chunks)
            logits = tf.reshape(logits[0], [-1]).numpy()
            logits = ','.join(map(str, logits))
            logits_list.append(logits)
        logits_list = str(logits_list)
        
        target_list = []
        for valid_elem in validation_elements:
            val_targets = valid_elem[1]
            val_targets = val_targets.numpy()[0]
            val_targets = "".join({1: "A", 2: "C", 3: "G", 4: "T"}[i] for i in val_targets if i in {1: "A", 2: "C", 3: "G", 4: "T"})
            target_list.append(val_targets)
        target_list = str(target_list)
        
        # Get the learning rate
        lr = scheduler((epoch+1)*STEPS_PER_EPOCH).numpy()
        # Only needed for custom scheduler
        #scheduler.update(train_loss)
        print("Learning rate:", lr)
        
        # Log the nu_log and theta_log of each lru layer
        lru_values = lru_logger(model)
        
        # Write csv log
        csv_logger({"epoch":epoch+1, "train_loss":train_loss, "val_loss":val_loss, "val_mean_accuracy":mean_accuracy_original, "val_median_accuracy":median_accuracy_original, "learning_rate":lr, "lru_values":lru_values, "alignments":alignments_list, "logits":logits_list, "valid_targets":target_list})
