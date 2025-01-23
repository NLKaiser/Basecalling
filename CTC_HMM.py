import tensorflow as tf

class HMM:
    
    def __init__(self, batch_size, padded_label_length, max_time, blank_index, epsilon=-1e6):
        # The batch size
        self.batch_size = batch_size
        # Non dynamic shape of alphas last dimension
        self.max_length = 2 * padded_label_length + 1
        # Non dynamic shape of alphas time dimension
        self.max_time = max_time
        
        # Large negative value for log space, e.g. -tf.float32.max
        self.epsilon = epsilon
        
        # Used in the calculation of the expanded labels
        self.n = padded_label_length
        self.blank_tensor = tf.fill([self.batch_size, self.n + 1], blank_index) # one more blank_index for the end
        
        # Used in the calculation of the transition matrix
        self.A_zero = tf.fill([self.batch_size, self.max_length - 2], 0.0)
        
        # Used in the calculation of alpha
        self.initial_distribution = self.build_initial_distribution_matrix()
        # F_prev has to be padded to the correct length
        self.subdiagonal_1_padding = tf.fill((self.batch_size, 1), self.epsilon)
        self.subdiagonal_2_padding = tf.fill((self.batch_size, 2), self.epsilon)
        
        # Used in the calculation of the loss
        # Create a batch index tensor: [0, 1, ..., batch_size-1]
        self.batch_indices = tf.range(self.batch_size, dtype=tf.int32)  # Shape: (batch_size,)
    
    def build_initial_distribution_matrix(self):
        """
        The first two entries are 1, the rest is 0.
        
        Returns:
            Tensor ((batch_size, max_length)): The start states, either blank or the first actual label.
        """
        ones = tf.fill([self.batch_size, 2], 0.0)
        zeros = tf.fill([self.batch_size, self.max_length - 2], self.epsilon)
        initial_distribution = tf.concat([ones, zeros], axis=1)
        return initial_distribution
    
    @tf.function
    def expand_labels(self, labels):
        """
        The labels with blanks in between them. A blank character is also at the start and at the end.
        
        Args:
            labels ((batch_size, padded_label_length)): The labels to predict during training.
        
        Returns:
            Tensor ((batch_size, max_length)): Labels with blanks in between, including the front and end.
        """
        # Interleave and reshape
        interleaved = tf.stack([self.blank_tensor[:,:-1], labels], axis=-1) # slice to avoid last blank_index
        reshaped = tf.reshape(interleaved, [self.batch_size, self.n * 2])
        
        # Add the last blank_index at the end
        output = tf.concat([reshaped, self.blank_tensor[:,-1:]], axis=-1)
        return output[:, :self.max_length]
    
    @tf.function
    def build_sparse_transition_matrix(self, expanded_labels):
        """
        Only the positions of the transition matrix that lie in the second row below the diagonal
        are extracted here. These are transitions from one actual label to the next, only if the
        two labels are not the same.
        
        Args:
            expanded_labels ((batch_size, max_length)): Labels with blanks inserted.
        
        Returns:
            Tensor ((batch_size, max_length)): Positions where expanded_labels[j] are not equal to expanded_labels[j-2].
        """
        # Compare expanded_labels[j] and expanded_labels[j-2]
        comparison = tf.not_equal(expanded_labels[:, :-2], expanded_labels[:, 2:])
        # A in log space
        A = tf.where(comparison, self.A_zero, self.epsilon)
        return A
    
    @tf.function
    def build_UB_matrix(self, logits, expanded_labels):
        """
        Gather the indices from softmaxed logits based on expanded_labels.
        
        Args:
            logits ((batch_size, t, num_labels)): Softmaxed logits.
            expanded_labels ((batch_size, max_length)): Labels with blanks inserted.
        
        Returns:
            Tensor ((batch_size, max_time, max_length)): Values for every time step and label.
        """
        expanded_indices = tf.repeat(expanded_labels, self.max_time, axis=0)
        expanded_indices = tf.reshape(expanded_indices, [self.batch_size, self.max_time, self.max_length])
        UB = tf.gather(logits, expanded_indices, axis=2, batch_dims=2)
        # There is no check for 0 values. In the models used the logits are clipped and then softmaxed,
        # we do not expect numerical underflow here.
        # UB in log space
        UB = tf.math.log(UB)
        return UB
    
    @tf.function
    def calculate_alpha(self, A_sparse, UB):
        """
        Forward probabilities. Calculated in log space. Calculated over the full range max_time, max_length for
        every entry in the batch.
        
        Args:
            A_sparse ((batch_size, max_length)): Positions where expanded_labels[j] are not equal to expanded_labels[j-2].
            UB ((batch_size, max_time, max_length)): Values for every time step and label.
        
        Returns:
            Tensor ((batch_size, max_time, max_length)): Forward probabilities in log space.
        """
        # Initialise F[0]
        F_init = self.initial_distribution + UB[:, 0, :]
        
        # Initialise the TensorArray
        alpha_ta = tf.TensorArray(tf.float32, size=self.max_time, clear_after_read=False, element_shape=[self.batch_size, self.max_length])
        alpha_ta = alpha_ta.write(0, F_init)
        
        for t in range(1, self.max_time):
            F_prev = alpha_ta.read(t-1)
            # Contributions from subdiagonal 1. Pad the first entry of F_prev.
            subdiagonal_1 = tf.concat([self.subdiagonal_1_padding, F_prev[:, :-1]], axis=1)
            
            # Contributions from subdiagonal 2
            subdiagonal_2 = tf.concat([self.subdiagonal_2_padding, F_prev[:, :-2] + A_sparse], axis=1)
            
            # Combine contributions in log space
            F_current = tf.reduce_logsumexp(
                [F_prev, subdiagonal_1, subdiagonal_2],
                axis=0  # Combine over the 3 contributions
            ) + UB[:, t, :]  # Add UB for the current time step
            alpha_ta = alpha_ta.write(t, F_current)
        
        # Stack the result
        alpha = alpha_ta.stack()  # Shape: (max_time, batch_size, max_length)
        return alpha
    
    @tf.function
    def loss(self, alpha, label_length, input_length):
        """
        Calculate the loss value from the last possible positions, either
        the last actual label or the last blank.
        
        Args:
            alpha ((batch_size, max_time, max_length)): Forward probabilities in log space.
            label_length ((batch_size)): Length of the original labels.
            input_length ((batch_size)): Time steps considered.
        
        Returns:
            Tensor ((batch_size)): Loss values.
        """
        expanded_length = 2 * label_length + 1
        
        # Gather the values alpha[input_length, label_length]
        gather_indices_main = tf.stack([input_length - 1, self.batch_indices, expanded_length - 1], axis=1)
        alpha_main = tf.gather_nd(alpha, gather_indices_main)  # Shape: (batch_size)
        
        # Gather the values alpha[input_length, label_length - 1]
        gather_indices_minus1 = tf.stack([input_length - 1, self.batch_indices, expanded_length - 2], axis=1)
        alpha_minus1 = tf.gather_nd(alpha, gather_indices_minus1)  # Shape: (batch_size)
        
        values = tf.stack([alpha_minus1, alpha_main], axis=-1)
        
        return -tf.reduce_logsumexp(values, axis=-1)
    
    @tf.function
    def __call__(self, labels, logits, label_length, input_length):
        # Prepare inputs
        logits = tf.nn.softmax(logits)
        
        # Calculate matrices
        expanded_labels = self.expand_labels(labels)
        A_sparse = self.build_sparse_transition_matrix(expanded_labels)
        UB = self.build_UB_matrix(logits, expanded_labels)
        alpha = self.calculate_alpha(A_sparse, UB)
        loss = self.loss(alpha, label_length, input_length)
        return loss
