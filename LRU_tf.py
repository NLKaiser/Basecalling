"""Linear Recurrent Unit
    Special case of a recurrent neural network with complex-valued weights and linear cell.
    The implementation is based on the paper:
    Orvieto et al, "Resurrecting Recurrent Neural Networks for Long Sequences", 2023
    with minor differences in the initialization of the weights and my assumptions on the nonlinear, position-wise layer.

Example:
    import LRU_tf as lru
    
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(None,H)))
    model.all(lru.LRU(N, H)
    model.add(layers.Dense(1, activation='sigmoid'))
"""

__author__ = "Mario Stanke"
__version__ = "0.1"
__license__ = "GPL"

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.initializers import Initializer, GlorotNormal, RandomNormal
from tensorflow.math import log


class DiskRingRadius(Initializer):
    def __init__(self, r_min=0, r_max=1):
        self.r_min = r_min
        self.r_max = r_max

    def __call__(self, shape, dtype=None):
        dtype = tf.float32 if dtype is None else dtype
        u1 = tf.random.uniform(shape=shape, dtype=dtype)
        nu_log = log(-0.5 * log(u1 * (self.r_max**2 - self.r_min**2) + self.r_min**2))        
        return nu_log

    def get_config(self): # to support serialization
        return {'r_min': self.r_min, 'r_max': self.r_max}

class DiskRingPhase(Initializer):
    def __init__(self, max_phase):
        self.max_phase = max_phase
                    
    def __call__(self, shape, dtype=None):
        u2 = tf.random.uniform(shape=shape)
        theta_log = log(self.max_phase * u2)
        return theta_log

    def get_config(self): # to support serialization
        return {'max_phase': self.max_phase}

class LRU(tf.keras.layers.Layer):
    """Layer of Linear Recurrent Unit with complex-valued weights."""
    def __init__(self, N, H, r_min=0, r_max=1, max_phase=6.28/20,
                 return_sequences=False, # sequence dimension in output
                 max_tree_depth=30, # maximum depth of the associative scan tree
                 use_skip_connection=True # use matrix or vector D, could be set to False if used in a residual layer setting to save parameters
                 ):
        super(LRU, self).__init__()
        self.N = N # hidden state dimension
        self.H = H # ouput dimension
        self.H_in = None # input dimension, determined in build
        self.return_sequences = return_sequences
        self.max_tree_depth = max_tree_depth
        self.use_skip_connection = use_skip_connection
        self.drr = DiskRingRadius(r_min=r_min, r_max=r_max)
        self.drp = DiskRingPhase(max_phase=max_phase)

    def build(self, input_shape):
        N = self.N
        H = self.H
        H_in = self.H_in = input_shape[-1] # usually H_in = H when placed in series

        self.B_re = self.add_weight(shape=(N, H_in), initializer=GlorotNormal(),
                                 dtype=tf.float32, trainable=True, name='B_re')
        self.B_im = self.add_weight(shape=(N, H_in), initializer=GlorotNormal(),
                                 dtype=tf.float32, trainable=True, name='B_im')
        self.C_re = self.add_weight(shape=(H, N), initializer=GlorotNormal(),
                                 dtype=tf.float32, trainable=True, name='C_re')
        self.C_im = self.add_weight(shape=(H, N), initializer=GlorotNormal(),
                                 dtype=tf.float32, trainable=True, name='C_im')
        self.nu_log = self.add_weight(shape=(N,), initializer=self.drr,
                                      trainable=True, name='nu_log')                    
        self.theta_log = self.add_weight(shape=(N,), initializer=self.drp,
                                         trainable=True, name='theta_log')
        if self.use_skip_connection:
            if H_in != H: # need a dimension change when 'skipping', formula (1) in the paper
                self.D = self.add_weight(shape=(H_in, H), initializer=RandomNormal(),
                                        dtype=tf.float32, trainable=True, name='D')
            else: # no dimension change, JAX code from paper
                self.D = self.add_weight(shape=(H,), initializer=RandomNormal(),
                                        dtype=tf.float32, trainable=True, name='d')

        super(LRU, self).build(input_shape)

    def call(self, input_sequence):
        """Forward pass of the LRU layer. Output y and input_sequence are of shape (L, H)."""
        L = tf.shape(input_sequence)[-2] # length of the input sequence

        # compute derived parameters
        Lambda = tf.exp(tf.complex(real=-tf.exp(self.nu_log), imag=tf.exp(self.theta_log)))
        gamma = tf.complex(real=tf.sqrt(1 - tf.abs(Lambda)**2), imag=0.)

        # materializing the diagonal of Lambda and projections
        B_norm = tf.complex(real=self.B_re, imag=self.B_im) * tf.expand_dims(gamma, axis=-1)
        C = tf.complex(real=self.C_re, imag=self.C_im)
        
        # Running the LRU
        # computing the inner states x_k
        Bu_elements = tf.matmul(tf.complex(real=input_sequence, imag=0.), B_norm, transpose_b=True)
        Lambda_elements = tf.repeat(tf.expand_dims(Lambda, axis=0), L, axis=0)
        batch_size = tf.shape(input_sequence)[0]
        Lambda_elements = tf.repeat(tf.expand_dims(Lambda_elements, axis=0), batch_size, axis=0)

        # associative scan, parallel on a tree, requires that axes -2 (length) are equal
        # Batch dimension is broadcastable, though.
        def scan_fn(a, b):
            Lambda_power_a, Bu_element_a = a
            Lambda_power_b, Bu_element_b = b
            return Lambda_power_a * Lambda_power_b,  Lambda_power_b * Bu_element_a + Bu_element_b

        """
        # This code requires that the input sequence length is known
        max_Len = 2**self.max_tree_depth # = 1073741824 approx 1G, upper limit for input length
        if (L > max_Len):
            print (f"Error in LRU.call: input sequence too long ({L} > {max_Len}).")
        """
        _, inner_states = tfp.math.scan_associative(scan_fn, (Lambda_elements, Bu_elements),
                                max_num_levels=self.max_tree_depth, axis=-2)
        # output projection
        if not self.return_sequences:
            # ignore all but the last inner state and input token
            inner_states = tf.expand_dims(inner_states[..., -1, :], axis=-2)
            input_sequence = tf.expand_dims(input_sequence[..., -1, :], axis=-2)

        y = tf.math.real(tf.matmul(inner_states, C, transpose_b=True))
        # pointwise multiplication for each position
        if self.use_skip_connection:
            if self.H_in != self.H:
                y = y + tf.matmul(input_sequence, self.D)
            else:
                y = y + tf.math.multiply(input_sequence, tf.expand_dims(self.D, axis=0))
        if not self.return_sequences:
            y = tf.squeeze(y, axis=-2) # reduce the rank by 1

        return y
    
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + (self.H,) if self.return_sequences \
            else input_shape[:-2] + (self.H,)
        return output_shape


class LRU_Block(tf.keras.layers.Layer):
    """ LRU Block layer
    Can be stacked and can replace another RNN such as LSTM or GRU. Optionally, 
         - LRU is bidirectional. In this case the forward and reverse LRU have separate weights and the ouput dimension is doubled. TODO: allow for different input and output dimensions. 
         - with nonlinear, position-wise layer
         - with skip connections
    """
    def __init__(self, N, H, r_min=0, r_max=1, max_phase=6.28/20,
                 max_tree_depth=30, # maximum depth of the associative scan tree
                 bidirectional=False,
                 return_sequences=True,
                 use_skip_connection=False, # use D parameter in LRU layer
                 residual=True, # use skip connection in the LRU_Block
                 use_nonlin = True, # nonlinear, position-wise layer
                 dropout=0.1, # dropout rate
                 use_batch_norm=True # batch normalization
                 ):
        super(LRU_Block, self).__init__()
        self.H = H
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        self.residual = residual
        self.use_nonlin = use_nonlin
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        H_out_fw = H
        if self.bidirectional:
            H_out_rv = H // 2
            H_out_fw = H - H_out_rv # approximately half, after concatenation the output dim is H in any case
            if (H_out_rv == 0):
                print ("Error in bidirectional LRU_Block.",
                       "The output dimension H must be at least 1 and should be even.")
            # reverse LRU has separate weights
            self.lru_rv = LRU(N, H_out_rv, r_min, r_max, max_phase, return_sequences=return_sequences,
                              max_tree_depth=max_tree_depth, use_skip_connection=use_skip_connection)
        # forward LRU
        self.lru_fw = LRU(N, H_out_fw, r_min, r_max, max_phase, return_sequences=return_sequences,
                          max_tree_depth=max_tree_depth, use_skip_connection=use_skip_connection)
        if self.use_batch_norm:
            self.bnorm = tf.keras.layers.BatchNormalization(axis=-1)
        if self.use_nonlin:
            self.denseW = tf.keras.layers.Dense(H, activation='linear')
            self.denseV = tf.keras.layers.Dense(H, activation='sigmoid')
        if self.dropout > 0.0:
            self.dropout_layer = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.lru_fw.build(input_shape)
        if self.bidirectional:
            self.lru_rv.build(input_shape)
        if self.use_batch_norm:
            self.bnorm.build(input_shape)
        if self.use_nonlin:
            idx = -1 if self.return_sequences else -2
            shape = input_shape[:idx] + (self.H,)
            self.denseW.build(shape)
            self.denseV.build(shape)
        if self.dropout > 0.0:
            self.dropout_layer.build(input_shape)
        super(LRU_Block, self).build(input_shape)

    def call(self, input_sequence):
        """Forward pass of the LRU_Block layer. Output y and input_sequence are of shape (L, H)."""
        y = input_sequence
        if self.use_batch_norm:
            y = self.bnorm(y) # batch normalization
        y_fw = self.lru_fw(y)
        if self.bidirectional:
            y_rv = self.lru_rv(tf.reverse(y, axis=[-2]))
            y = tf.concat([y_fw, tf.reverse(y_rv, axis=[-2])], axis=-1)
        else:
            y = y_fw
        if self.use_nonlin:
            # gated linear unit (XW+b) * sigmoid(XV+c)
            W = self.denseW(y)
            V = self.denseV(y)
            y = tf.multiply(W, V)
        if self.dropout > 0:
            y = self.dropout_layer(y)
        if self.residual and self.return_sequences and input_sequence.shape[-1] == self.H:
            # residual connection, only if return_sequences=True, for inner layer in stack
            y = y + input_sequence
        return y

    def compute_output_shape(self, input_shape):
        # the sequence dim is removed if return_sequences=False
        output_shape = input_shape[:-1] + (self.H,) if self.return_sequences else input_shape[:-2] + (self.H,)
        return output_shape



# -------- legacy code -------------
def init_lru_parameters(N, H, r_min=0, r_max=1, max_phase=6.28):
    """Initialize parameters of the LRU layer.
       Tested to be equivalent to the jax version, but not needed for the layer.
    """
    
    # N: state dimension, H: model dimension
    # Initialization of Lambda is complex valued distributed uniformly on ring 
    # between r_min and r_max, with phase in [0, max_phase].
    u1 = tf.random.uniform(shape=(N,))
    u2 = tf.random.uniform(shape=(N,))
    nu_log = tf.math.log(-0.5 * tf.math.log(u1 * (r_max**2 - r_min**2) + r_min**2))
    theta_log = tf.math.log(max_phase * u2) 
  
    # Glorot initialized Input/Output projection matrices
    B_re = tf.random.normal(shape=(N, H)) / tf.sqrt(2.0 * H)
    B_im = tf.random.normal(shape=(N, H)) / tf.sqrt(2.0 * H)
    C_re = tf.random.normal(shape=(H, N)) / tf.sqrt(1.0 * N)
    C_im = tf.random.normal(shape=(H, N)) / tf.sqrt(1.0 * N)
    D = tf.random.normal(shape=(H,))
    
    # Normalization factor
    diag_lambda = tf.exp(tf.complex(real=-tf.exp(nu_log), imag=tf.exp(theta_log)))
    gamma_log = tf.math.log(tf.sqrt(1 - tf.abs(diag_lambda)**2))

    return nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log
