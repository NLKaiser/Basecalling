import numpy as np
import tensorflow as tf

"""
Base case of the CTC_HMM. Used with 5 classes (blank, A, C, G, T).
The alpha_loss and beta_loss are the same as tf.nn.ctc_loss.
"""

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
        # Build a fixed mask for the second dimension:
        mask = tf.constant([False if i % 2 == 0 else True for i in range(self.max_length-2)], dtype=tf.bool)
        # Tile the mask over the batch dimension:
        self.blank_mask = tf.tile(tf.expand_dims(mask, axis=0), [self.batch_size, 1])
        
        # Used in the calculation of alpha
        self.initial_distribution = self.build_initial_distribution_matrix()
        
        # Used in the calculation of the loss
        # Create a batch index tensor: [0, 1, ..., batch_size-1]
        self.batch_indices = tf.range(self.batch_size, dtype=tf.int32)  # Shape: (batch_size,)
        
        # Decoding
        self.index_to_char = {1: "A", 2: "C", 3: "G", 4: "T"}
    
    def build_initial_distribution_matrix(self):
        """
        The first two entries are 0, the rest is self.epsilon.
        
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
        Classic transition matrix read from rows to columns. There are only valid transitions on the diagonal,
        super-diagonal and super-super-diagonal. Only these entries are returned. Transitions are in log space.
        
        Args:
            expanded_labels ((batch_size, max_length)): Labels with blanks inserted.
        
        Returns:
            Tensor ((batch_size, 3, max_length)): The full diagonal contains valid transitions.
                                                  The full super-diagonal contins valid transitions.
                                                  On the super-super-diagonal are positions where expanded_labels[j] are not equal to 
                                                  expanded_labels[j-2].
        """
        diag = tf.zeros([self.batch_size, self.max_length])
        super_diag = tf.concat([tf.fill([self.batch_size, 1], self.epsilon), tf.zeros([self.batch_size, self.max_length-1])], axis=-1)
        # Compare expanded_labels[j] and expanded_labels[j-2]
        comparison = tf.not_equal(expanded_labels[:, :-2], expanded_labels[:, 2:])
        # A in log space
        super2_diag = tf.where(comparison, self.A_zero, self.epsilon)
        # Make blank transitions invalid
        super2_diag = tf.where(self.blank_mask, super2_diag, self.epsilon)
        super2_diag = tf.concat([tf.fill([self.batch_size, 2], self.epsilon), super2_diag], axis=-1)
        A = tf.stack([diag, super_diag, super2_diag], axis=1)
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
        return UB
    
    @tf.function
    def calculate_fwd_bwd(self, UB, A, expanded_labels, expanded_label_length):
        """
        Forward and backward probabilities. Calculated in log space. Calculated over the full range max_time, max_length for
        every entry in the batch.
        
        Args:
            logits ((batch_size, max_time, 5)): Logits in log space.
            A ((batch_size, 3, max_length)): Transition matrix for the diagonals with valid transitions.
            expanded_labels (batch_size, max_length)): Labels with blanks in between, including the front and end.
            expanded_label_length ((batch_size)): Lengths of the expanded labels.
        
        Returns:
            alpha ((max_time, batch_size, max_length)): Forward probabilities in log space.
            beta ((max_time, batch_size, max_length)): Backward probabilities in log space.
        """
        # alpha init
        alpha_ta = tf.TensorArray(tf.float32,
                                  size=self.max_time,
                                  clear_after_read=False,
                                  element_shape=[self.batch_size, self.max_length])
        alpha_ta = alpha_ta.write(0,
                                  self.initial_distribution +
                                  UB[:, 0, :])
        
        # beta init
        beta_ta = tf.TensorArray(tf.float32,
                                 size=self.max_time,
                                 clear_after_read=False,
                                 element_shape=[self.batch_size, self.max_length])
        
        mask_last = tf.one_hot(expanded_label_length - 1,   self.max_length, dtype=tf.float32)
        mask_pen  = tf.one_hot(expanded_label_length - 2, self.max_length, dtype=tf.float32)
        
        # combine into a 0/1 mask, 1 at the two end-states, 0 elsewhere
        mask      = tf.minimum(mask_last + mask_pen, 1.0)  
        
        # where mask==1 zero; where mask==0 self.epsilon
        init_beta = (1.0 - mask) * self.epsilon     # shape (B, L)
        
        # add UB at T-1
        beta_ta = beta_ta.write(
            self.max_time - 1,
            init_beta + UB[:, self.max_time - 1, :]
        )
        
        for t in range(1, self.max_time):
            t_beta = self.max_time - 1 - t
            
            a_prev = alpha_ta.read(t - 1)
            b_prev = beta_ta.read(t_beta+1)
            
            a_stay   = a_prev + A[:, 0, :]
            b_stay = b_prev + A[:,0,:]
            
            a_from1  = tf.concat([tf.fill([self.batch_size, 1], self.epsilon),
                                a_prev[:, :-1]],
                               axis=1) + A[:, 1, :]
            b_to1 = tf.concat([
                b_prev[:,1:] + A[:,1,1:],
                tf.fill([self.batch_size,1], self.epsilon)
            ], axis=1)
                               
            a_from2  = tf.concat([tf.fill([self.batch_size, 2], self.epsilon),
                                a_prev[:, :-2]],
                               axis=1) + A[:, 2, :]
            b_to2 = tf.concat([
                b_prev[:,2:] + A[:,2,2:],
                tf.fill([self.batch_size,2], self.epsilon)
            ], axis=1)
            
            a_t = tf.reduce_logsumexp(
                [a_stay, a_from1, a_from2], axis=0
            ) + UB[:, t, :]
            b_t = tf.reduce_logsumexp([b_stay, b_to1, b_to2], axis=0) + UB[:, t_beta, :]
            
            # Mask invalid states that are beyond the label length.
            valid_length_alpha = tf.minimum(tf.fill([self.batch_size], 2*t+2), expanded_label_length)
            mask_alpha = tf.sequence_mask(valid_length_alpha, maxlen=self.max_length, dtype=tf.bool)
            a_t = tf.where(mask_alpha, a_t, self.epsilon)
            
            alpha_ta = alpha_ta.write(t, a_t)
            beta_ta = beta_ta.write(t_beta, b_t)
        
        alpha = alpha_ta.stack()  # (max_time, batch_size, max_length)
        beta = beta_ta.stack()   # (max_time, batch_size, max_length)
        
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
    def middle_HMM_transition_matrix(self, transitions):
        abc_logits = tf.concat([
            transitions[:2],                          # [a, b]
            tf.zeros((1,), dtype=transitions.dtype)  # [0]
        ], axis=0)                                    # shape (3,)
        abc_logits = tf.nn.softmax(abc_logits)
        a = abc_logits[0]
        b = abc_logits[1]
        c = abc_logits[2]
        
        # build the (d,e) logits as a single Tensor of shape [2]
        de_logits = tf.stack([
            transitions[2],                           # d
            tf.constant(0.0, dtype=transitions.dtype) # e=0
        ], axis=0)                                   # shape (2,)
        de_logits = tf.nn.softmax(de_logits)
        d = de_logits[0]
        e = de_logits[1]
        
        A = tf.stack([[e, d, d, d, d],
             [a, c, b, b, b],
             [a, b, c, b, b],
             [a, b, b, c, b],
             [a, b, b, b, c]])
        
        return tf.math.log(A)
    
    @tf.function
    def logits_posteriors(self, logits, A):
        alpha = tf.TensorArray(tf.float32, size=self.max_time, clear_after_read=False, element_shape=[self.batch_size, 5])
        alpha = alpha.write(0, logits[:, 0, :])
        
        beta = tf.TensorArray(tf.float32, size=self.max_time, clear_after_read=False, element_shape=[self.batch_size, 5])
        beta = beta.write(self.max_time-1, tf.zeros((self.batch_size, 5), dtype=tf.float32))
        
        for t in range(1, self.max_time):
            t_b = self.max_time - t - 1
            prev = alpha.read(t-1)
            scores = tf.expand_dims(prev, 2) + A[None, :, :]
            alpha_t = tf.reduce_logsumexp(scores, axis=1) + logits[:, t, :]
            alpha = alpha.write(t, alpha_t)
            
            next_beta = beta.read(t_b + 1)
            scores = A[None, :, :] + tf.expand_dims(next_beta + logits[:, t_b + 1, :], 1)
            beta_t = tf.reduce_logsumexp(scores, axis=2)
            beta = beta.write(t_b, beta_t)
        
        alpha = alpha.stack() # = 0 #tf.transpose(alpha.stack(), perm=[1, 0, 2])  # (batch, time, 5)
        alpha = tf.transpose(alpha, perm=[1, 0, 2])
        
        beta = beta.stack()
        beta = tf.transpose(beta, perm=[1, 0, 2])
        
        logZ = tf.reduce_logsumexp(alpha[:, -1, :], axis=-1)
        
        return alpha + beta - tf.reshape(logZ, [self.batch_size, 1, 1])
    
    @tf.function
    def get_middle_HMM_logit_posteriors(self, logits, transitions):
        logits = tf.nn.log_softmax(logits, axis=-1)
    
        A = self.middle_HMM_transition_matrix(transitions)
        logits_posteriors = self.logits_posteriors(logits, A)
        
        return logits_posteriors
    
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
        logits = tf.math.log(logits)
        
        # Calculate matrices
        expanded_label_length = 2 * label_length + 1
        expanded_labels = self.expand_labels(labels)
        A_sparse = self.build_sparse_transition_matrix(expanded_labels)
        UB = self.build_UB_matrix(logits, expanded_labels)
        alpha, beta = self.calculate_fwd_bwd(UB, A_sparse, expanded_labels, expanded_label_length)
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
        logits = tf.nn.log_softmax(logits)
        
        # Calculate matrices
        expanded_label_length = 2 * label_length + 1
        expanded_labels = self.expand_labels(labels)
        A_sparse = self.build_sparse_transition_matrix(expanded_labels)
        UB = self.build_UB_matrix(logits, expanded_labels)
        alpha, beta = self.calculate_fwd_bwd(UB, A_sparse, expanded_labels, expanded_label_length)
        #loss_alpha = self.loss_alpha(alpha, expanded_label_length, input_length)
        loss_beta = self.loss_beta(beta)
        loss = self.get_loss(alpha, beta, loss_beta)
        return loss
