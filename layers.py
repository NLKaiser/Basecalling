import tensorflow as tf

class ReverseLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ReverseLayer, self).__init__(**kwargs)
        self.axis = axis  # The axis to reverse along, default is the last axis
    
    def call(self, inputs):
        # Reverse the input tensor along the specified axis
        return tf.reverse(inputs, axis=[self.axis])

class CastToFloat32(tf.keras.layers.Layer):
    def __init__(self):
        super(CastToFloat32, self).__init__()

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

class ClipLayer(tf.keras.layers.Layer):
    def __init__(self, min_, max_):
        super(ClipLayer, self).__init__()
        self.min = min_
        self.max = max_
    
    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max)

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
        # We use filters=1 to apply the kernel across the single input channel
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

class TemporalContextStackingLayer(tf.keras.layers.Layer):
    def __init__(self, n, **kwargs):
        """
        A custom layer that stacks feature states of previous n steps, 
        current step, and next n steps.
        
        Args:
        n (int): Number of steps to include before and after the current step.
        """
        super(TemporalContextStackingLayer, self).__init__(**kwargs)
        self.n = n

    def call(self, inputs):
        """
        Perform the stacking operation with appropriate padding.
        
        Args:
        inputs (tf.Tensor): A tensor of shape (batch_size, sequence_length, num_features).
        
        Returns:
        tf.Tensor: A tensor of shape (batch_size, sequence_length, (2*n + 1) * num_features).
        """
        # Padding on both sides of the sequence along the time dimension
        padding = [[0, 0], [self.n, self.n], [0, 0]]  # No padding for batch or feature dims
        padded_inputs = tf.pad(inputs, paddings=padding, mode='CONSTANT')

        # Collect slices for stacking
        slices = [
            padded_inputs[:, i:(i + tf.shape(inputs)[1]), :]
            for i in range(2 * self.n + 1)
        ]
        
        # Stack slices along the last dimension
        stacked = tf.concat(slices, axis=-1)
        return stacked

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.
        
        Args:
        input_shape (tuple): Shape of the input tensor (batch_size, sequence_length, num_features).
        
        Returns:
        tuple: Shape of the output tensor.
        """
        batch_size, sequence_length, num_features = input_shape
        return batch_size, sequence_length, num_features * (2 * self.n + 1)

class LinearCRFEncoder(tf.keras.layers.Layer):
    def __init__(self, n_base, state_len, bias=True, scale=None, activation=None, blank_score=-2.0, expand_blanks=True, permute=None, **kwargs):
        super(LinearCRFEncoder, self).__init__(**kwargs)
        self.scale = scale
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        self.expand_blanks = expand_blanks
        self.permute = permute

        size = (n_base + 1) * n_base**state_len if blank_score is None else n_base**(state_len + 1)
        self.linear = tf.keras.layers.Dense(size, use_bias=bias)
        self.activation = tf.keras.layers.Activation(activation) if activation else None
        
        self.blank_score_layer = tf.keras.layers.Dense(1, use_bias=bias)
    
    def call(self, inputs):
        # Permute dimensions if needed
        if self.permute is not None:
            inputs = tf.transpose(inputs, perm=self.permute)

        # Linear transformation
        scores = self.linear(inputs)

        # Calculate blank score
        dynamic_blank_scores = tf.squeeze(self.blank_score_layer(inputs), axis=-1)  # Compute dynamic blank scores

        # Apply activation if provided
        if self.activation is not None:
            scores = self.activation(scores)

        # Scale the scores if scale is provided
        if self.scale is not None:
            scores = scores * self.scale

        # Handle blank score and expansion
        if self.blank_score is not None and self.expand_blanks:
            T, N, C = tf.shape(scores)[0], tf.shape(scores)[1], tf.shape(scores)[2]
            scores = tf.reshape(scores, [T, N, C // self.n_base, self.n_base])

            # Create a tensor for the blank scores
            blank_score_tensor = tf.fill([T, N, C // self.n_base - 1, 1], self.blank_score)  # [T, N, C // self.n_base - 1, 1]
            dynamic_blank_scores_expanded = tf.expand_dims(dynamic_blank_scores, axis=-1)  # [T, N, 1]
            dynamic_blank_scores_expanded = tf.expand_dims(dynamic_blank_scores_expanded, axis=-1)  # [T, N, 1, 1]

            # Combine dynamic and static blank scores
            blank_score_tensor = tf.concat([dynamic_blank_scores_expanded, blank_score_tensor], axis=2)  # Adjust axis for proper combination

            # Concatenate blank scores to the beginning of the scores along the feature dimension
            scores = tf.concat([blank_score_tensor, scores], axis=-1)
            scores = tf.reshape(scores, [T, N, -1])
        
        return scores
