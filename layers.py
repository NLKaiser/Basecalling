"""
Custom layers.
"""

import tensorflow as tf

# Reverse an input along an axis
class ReverseLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ReverseLayer, self).__init__(**kwargs)
        self.axis = axis  # The axis to reverse along, default is the last axis
    
    def call(self, inputs):
        return tf.reverse(inputs, axis=[self.axis])

# Cast the input to a specific datatype
class CastToFloat32(tf.keras.layers.Layer):
    def __init__(self):
        super(CastToFloat32, self).__init__()

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# Clip the inputs values
class ClipLayer(tf.keras.layers.Layer):
    def __init__(self, min_, max_):
        super(ClipLayer, self).__init__()
        self.min = min_
        self.max = max_
    
    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max)

# Pad the inputs time dimension specific for division by 6
class DynamicPaddingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DynamicPaddingLayer, self).__init__()
    
    def call(self, inputs):
        # Calculate the time dimension dynamically
        time_dim = tf.shape(inputs)[1]
        
        # Calculate the remainder of the division by 6
        remainder = tf.math.floormod(time_dim, 6)
        
        # Calculate the needed padding for both sides
        pad_left = tf.cond(
            tf.greater(remainder, 0),
            lambda: (6 - remainder) // 2,
            lambda: tf.constant(0)
        )
        
        pad_right = tf.cond(
            tf.greater(remainder, 0),
            lambda: (6 - remainder) - pad_left,
            lambda: tf.constant(0)
        )
        
        # Define padding dimensions as a constant tensor
        paddings = tf.stack([[0, 0], [pad_left, pad_right], [0, 0]])  # padding for [batch, time, channels]

        # Apply padding
        padded_inputs = tf.pad(inputs, paddings, mode='CONSTANT')
        return padded_inputs

# Stack the state dimension of the previous n time steps, time step n and the next n time steps for each time step i
# Padding is applied as needed
# New state dimension is 2n + 1
class TemporalContextLayer(tf.keras.layers.Layer):
    def __init__(self, n, **kwargs):
        super(TemporalContextLayer, self).__init__(**kwargs)
        self.n = n  # Number of time steps before and after the current time step to include in the context

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, 1)
        
        # Pad the input along the time dimension with n zeros on each side
        padded_inputs = tf.pad(inputs, [[0, 0], [self.n, self.n], [0, 0]], mode='CONSTANT')

        # Use a 1D convolution to create the sliding window of size (2n + 1)
        # This will create overlapping windows of shape (2n + 1) for each time step
        # Use filters=1 to apply the kernel across the single input channel
        window_size = 2 * self.n + 1
        output = tf.image.extract_patches(
            images=tf.expand_dims(padded_inputs, -1),  # Expand dims to make it compatible with extract_patches
            sizes=[1, window_size, 1, 1],               # Set window size for time dimension
            strides=[1, 1, 1, 1],                       # Slide by 1 on time axis
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        # Now, output has shape (batch_size, time_steps, 1, window_size)
        # Squeeze out the unnecessary dimension
        output = tf.squeeze(output, axis=2)
        
        return output  # Final shape: (batch_size, time_steps, 2n + 1)
