import tensorflow as tf

class HMM:
    
    """
    Example usage:
    batch_size = 32
    time_ = 834
    num_labels = 5 # blank, A, C, G, T
    min_num_labels = 350
    max_num_labels = 500
    blank_index = 0
    
    def generate_row():
        length = tf.random.uniform(shape=[], minval=min_num_labels, maxval=max_num_labels+1, dtype=tf.int32)  # Random length
        values = tf.random.uniform(shape=[length], minval=1, maxval=num_labels, dtype=tf.int32)  # Random values between 1 and num_labels
        padding = tf.zeros([max_num_labels - length], dtype=tf.int32)  # Padding with zeros
        return tf.concat([values, padding], axis=0)  # Combine values and padding
    # Generate the tensor
    labels = tf.stack([generate_row() for _ in range(batch_size)])
    logits = tf.random.uniform((batch_size, time_, num_labels), minval=-5, maxval=5)
    label_length = tf.reduce_sum(tf.cast(labels != 0, tf.int32), axis=1)
    logit_length = tf.shape(logits)[1] * tf.ones(tf.shape(logits)[0], dtype=tf.int32)
    
    H = HMM(batch_size, max_num_labels, time_, blank_index, -tf.float32.max)
    loss = H(labels, logits, label_length, logit_length)
    """
    
    def __init__(self, batch_size, padded_label_length, max_time, blank_index, epsilon=-1e6):
        # The batch size
        self.batch_size = batch_size
        # Non dynamic shape of the state dimension
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
        
        # Decoding
        self.index_to_char = {1: "A", 2: "C", 3: "G", 4: "T"}
    
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
    def reverse_labels(self, labels, label_length):
        """
        Reverse the labels up to the padding positions.
        
        Args:
            labels ((batch_size, padded_label_length)): The labels to predict during training.
            label_length ((batch_size)): Length of the labels.
        
        Returns:
            Tensor ((batch_size, padded_label_length)): The labels reversed.
        """
        return tf.reverse_sequence(labels, label_length, seq_axis=1, batch_axis=0)
    
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
    def reverse_UB(self, UB, expanded_label_length):
        """
        Reverse UB along the state and time dimension.
        
        Args:
            UB ((batch_size, max_time, max_length)): Values for every time step and label.
            expanded_label_length ((batch_size)): Lengths of the expanded labels.
        
        Returns:
            Tensor ((batch_size, max_time, max_length)): UB reversed along the state and time dimension.
        """
        UB_reverse = tf.reverse_sequence(UB, expanded_label_length, seq_axis=2, batch_axis=0)
        UB_reverse = tf.reverse(UB_reverse, axis=[1])
        return UB_reverse
    
    @tf.function
    def calculate_fwd_bwd(self, A_sparse, A_sparse_beta, UB, UB_beta, expanded_label_length):
        """
        Forward and backward probabilities. Calculated in log space. Calculated over the full range max_time, max_length for
        every entry in the batch.
        
        Args:
            A_sparse ((batch_size, max_length)): Positions where expanded_labels[j] are not equal to expanded_labels[j-2].
            A_sparse_beta ((batch_size, max_length)): Positions where expanded_labels[j] are not equal to expanded_labels[j-2]. To calculate beta.
            UB ((batch_size, max_time, max_length)): Values for every time step and label.
            UB_beta ((batch_size, max_time, max_length)): Values for every time step and label. To calculate beta.
            expanded_label_length ((batch_size)): Lengths of the expanded labels.
        
        Returns:
            alpha ((max_time, batch_size, max_length)): Forward probabilities in log space.
            beta ((max_time, <batch_size, max_length)): Backward probabilities in log space.
        """
        # Initialise alpha
        alpha_ta = tf.TensorArray(tf.float32, size=self.max_time, clear_after_read=False, element_shape=[self.batch_size, self.max_length])
        alpha_ta = alpha_ta.write(0, self.initial_distribution + UB[:, 0, :])
        # Initialise beta
        beta_ta = tf.TensorArray(tf.float32, size=self.max_time, clear_after_read=False, element_shape=[self.batch_size, self.max_length])
        beta_ta = beta_ta.write(0, self.initial_distribution + UB_beta[:, 0, :])
        
        for t in range(1, self.max_time):
            alpha_prev = alpha_ta.read(t-1)
            beta_prev = beta_ta.read(t-1)
            
            # Contributions from subdiagonal 1. Pad the first entry of F_prev.
            subdiagonal_1_alpha = tf.concat([self.subdiagonal_1_padding, alpha_prev[:, :-1]], axis=1)
            subdiagonal_1_beta = tf.concat([self.subdiagonal_1_padding, beta_prev[:, :-1]], axis=1)
            
            # Contributions from subdiagonal 2
            subdiagonal_2_alpha = tf.concat([self.subdiagonal_2_padding, alpha_prev[:, :-2] + A_sparse], axis=1)
            subdiagonal_2_beta = tf.concat([self.subdiagonal_2_padding, beta_prev[:, :-2] + A_sparse_beta], axis=1)
            
            # Combine contributions in log space
            alpha_current = tf.reduce_logsumexp(
                [alpha_prev, subdiagonal_1_alpha, subdiagonal_2_alpha],
                axis=0  # Combine over the 3 contributions
            ) + UB[:, t, :]  # Add UB for the current time step
            alpha_ta = alpha_ta.write(t, alpha_current)
            
            beta_current = tf.reduce_logsumexp(
                [beta_prev, subdiagonal_1_beta, subdiagonal_2_beta],
                axis=0
            ) + UB_beta[:, t, :]
            beta_ta = beta_ta.write(t, beta_current)
        
        # Stack the result
        alpha = alpha_ta.stack()  # Shape: (max_time, batch_size, max_length)
        beta = beta_ta.stack()
        # Reverse beta to get the classic backward matrix
        beta = tf.reverse_sequence(beta, expanded_label_length, seq_axis=2, batch_axis=1)
        beta = tf.reverse(beta, axis=[0])
        return alpha, beta
    
    @tf.function
    def loss_alpha(self, alpha, expanded_label_length, input_length):
        """
        Calculate the loss value from the last possible positions, either
        the last actual label or the last blank.
        
        Args:
            alpha ((max_time, batch_size, max_length)): Forward probabilities in log space.
            expanded_label_length ((batch_size)): Length of the expanded labels.
            input_length ((batch_size)): Time steps considered.
        
        Returns:
            Tensor ((batch_size)): Loss values.
        """
        # Gather the values alpha[input_length, label_length]
        gather_indices_main = tf.stack([input_length - 1, self.batch_indices, expanded_label_length - 1], axis=1)
        alpha_main = tf.gather_nd(alpha, gather_indices_main)  # Shape: (batch_size)
        
        # Gather the values alpha[input_length, label_length - 1]
        gather_indices_minus1 = tf.stack([input_length - 1, self.batch_indices, expanded_label_length - 2], axis=1)
        alpha_minus1 = tf.gather_nd(alpha, gather_indices_minus1)  # Shape: (batch_size)
        
        values = tf.stack([alpha_minus1, alpha_main], axis=-1)
        return -tf.reduce_logsumexp(values, axis=-1)
    
    @tf.function
    def loss_beta(self, beta):
        """
        Calculate the loss value from the first two positions, either
        the first blank or the first actual label.
        
        Args:
            beta ((max_time, batch_size, max_length)): Backward probabilities in log space.
        
        Returns:
            Tensor ((batch_size)): Loss values.
        """
        return -tf.reduce_logsumexp(beta[0, :, :2], axis=-1)
    
    @tf.function
    def get_loss(self, alpha, beta, l):
        """
        Total loss.
        
        Args:
            alpha ((max_time, batch_size, max_length)): Forward probabilities in log space.
            beta ((max_time, batch_size, max_length)): Backward probabilities in log space.
            l ((batch_size)): Loss values for every entry in the batch.
        
        Returns:
            Tensor ((batch_size)): Total loss for every entry in the batch.
        """
        # l ist the ctc loss
        prob = alpha + beta
        prob = tf.reduce_logsumexp(prob, axis=-1) # Reduce over S
        prob = tf.reduce_logsumexp(prob, axis=0) # Reduce over T
        return -(prob - l)
    
    @tf.function
    def get_y(self, labels, logits, label_length, input_length):
        """
        Get the matrix with posterior probabilities.
        
        Args:
            labels ((batch_size, padded_label_length)): The training labels. Padded with zeros.
            logits ((batch_size, max_time, 5)): Values for every time step and class (blank, A, C, G, T).
            label_length ((batch_size)): The actual label lengths without the padding.
            input_length ((batch_size)): Number of time steps to be considered.
        
        Returns:
            y ((max_time, batch_size, max_length)): Posterior probabilities.
        """
        # Prepare inputs
        logits = tf.nn.softmax(logits)
        
        # Calculate matrices
        reversed_labels = self.reverse_labels(labels, label_length)
        expanded_label_length = 2 * label_length + 1
        expanded_labels = self.expand_labels(labels)
        expanded_reversed_labels = self.expand_labels(reversed_labels)
        A_sparse = self.build_sparse_transition_matrix(expanded_labels)
        A_sparse_beta = self.build_sparse_transition_matrix(expanded_reversed_labels)
        UB = self.build_UB_matrix(logits, expanded_labels)
        UB_beta = self.reverse_UB(UB, expanded_label_length)
        alpha, beta = self.calculate_fwd_bwd(A_sparse, A_sparse_beta, UB, UB_beta, expanded_label_length)
        #loss_alpha = self.loss_alpha(alpha, expanded_label_length, input_length)
        loss_beta = self.loss_beta(beta)
        return alpha + beta - tf.expand_dims(tf.expand_dims(loss_beta, axis=0), axis=-1)
    
    def decode(self, logits):
        """
        Decode the model output. Greedy strategy using the class with the maximum value. Repeated characters are collapsed. Blanks are removed.
        
        Args:
            logits ((batch_size, time_steps, 5)): The model output.
        
        Returns:
            List ((batch_size)): The DNA sequences as strings.
        """
        # Greedy decoding
        indices = tf.argmax(logits, axis=-1)
        indices = tf.cast(indices, tf.int32)
        shifted = tf.pad(indices[:, :-1], [[0, 0], [1, 0]], constant_values=-1)
        mask = tf.not_equal(indices, shifted)
        collapsed = tf.where(mask, indices, tf.fill(tf.shape(indices), -1))
        collapsed_np = collapsed.numpy()
        
        decoded = ["".join(self.index_to_char[i] for i in row if i in self.index_to_char)
        for row in collapsed_np]
        return decoded
    
    @tf.function
    def __call__(self, labels, logits, label_length, input_length):
        """
        Calculate the total loss for a batch of inputs.
        
        Args:
            labels ((batch_size, padded_label_length)): The training labels. Padded with zeros.
            logits ((batch_size, max_time, 5)): Values for every time step and class (blank, A, C, G, T).
            label_length ((batch_size)): The actual label lengths without the padding.
            input_length ((batch_size)): Number of time steps to be considered.
        
        Returns:
            loss ((batch_size)): The total loss for every entry in the batch.
        """
        # Prepare inputs
        logits = tf.nn.softmax(logits)
        
        # Calculate matrices
        reversed_labels = self.reverse_labels(labels, label_length)
        expanded_label_length = 2 * label_length + 1
        expanded_labels = self.expand_labels(labels)
        expanded_reversed_labels = self.expand_labels(reversed_labels)
        A_sparse = self.build_sparse_transition_matrix(expanded_labels)
        A_sparse_beta = self.build_sparse_transition_matrix(expanded_reversed_labels)
        UB = self.build_UB_matrix(logits, expanded_labels)
        UB_beta = self.reverse_UB(UB, expanded_label_length)
        alpha, beta = self.calculate_fwd_bwd(A_sparse, A_sparse_beta, UB, UB_beta, expanded_label_length)
        #loss_alpha = self.loss_alpha(alpha, expanded_label_length, input_length)
        loss_beta = self.loss_beta(beta)
        loss = self.get_loss(alpha, beta, loss_beta)
        return loss
