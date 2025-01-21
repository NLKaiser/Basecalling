import tensorflow as tf

class HMM:
    
    def __init__(self):
        pass
    
    @tf.function
    def expand_labels(self, labels, label_length, blank_index, batch_size, expanded_length, max_length):
        """
        Insert blank_index before, between and after the labels, pad the rest with 0.
        
        Args:
            labels ((batch_size, n)): Labels for each sequence, padded with 0.
            label_length ((batch_size)): The length of each sequence, equal to the starting point of the padding.
            blank_index ((1)): An integer describing the position in the logits at which the prediction of the blank token is encoded.
            batch_size ((1)): The batch size.
            expanded_length ((batch_size)): The length of the expanded labels after blank_index has been inserted.
                                            Equals 2 * label_length - 1.
            max_length ((1)): An integer that specifies the maximum length of the expanded labels.
                              Equals tf.reduce_max(expanded_length).
        
        Returns:
            Tensor ((batch_size, max_length)): The expanded labels padded with 0.
        """
        # Create a mask for valid labels
        label_mask = tf.sequence_mask(label_length, maxlen=tf.shape(labels)[1])
        valid_labels = tf.where(label_mask, labels, tf.zeros_like(labels))

        # Interleave blank and labels
        blanks = tf.fill(tf.shape(valid_labels), blank_index)
        
        # Create the interleaved sequence (start with blank, then label, then blank...)
        interleaved = tf.reshape(
            tf.concat([blanks[..., None], valid_labels[..., None]], axis=-1),
            [batch_size, -1]
        )
        
        # Trim the interleaved sequence to the maximum length minus 1 (leaving space for the final blank)
        interleaved = interleaved[:, :max_length - 1]
        
        # Add the final blank at the end of the sequence
        end_separator = tf.fill([batch_size, 1], blank_index)
        interleaved_with_end = tf.concat([interleaved, end_separator], axis=1)
        
        # Padding beyond the valid data with 0
        result = tf.where(
            tf.sequence_mask(expanded_length, maxlen=max_length),
            interleaved_with_end,
            tf.fill([batch_size, max_length], 0)
        )
        return result
    
    @tf.function
    def build_sparse_transition_matrix(self, expanded_labels, batch_size, max_length):
        """
        Only the positions of the transition matrix that lie in the second row below the diagonal
        are extracted here. These are transitions from one actual label to the next, only if the
        two labels are not the same.
        
        Args:
            expanded_labels ((batch_size, max_length)): Labels with blanks inserted.
            batch_size ((1)): The batch size.
            max_length ((1)): An integer that specifies the maximum length of the expanded labels.
                              Equals tf.reduce_max(expanded_length).
             
        Returns:
            Tensor ((batch_size, max_length)): Positions where expanded_labels[j] are not equal to expanded_labels[j-2].
        """
        # Compare expanded_labels[j] and expanded_labels[j-2]
        comparison = tf.not_equal(expanded_labels[:, :-2], expanded_labels[:, 2:])
        # A in log space
        A = tf.where(comparison, tf.fill([batch_size, max_length - 2], tf.math.log(1.0)), -1e6)
        # Add two columns of 0 to the front of A, as it is supposed to be the second row below the diagonal.
        padding = tf.fill([batch_size, 2], 0.0)
        A = tf.concat([padding, A], axis=1)
        return A
    
    # NOT USED ANYMORE. Calculate the transition matrix A, no padding beyond expanded_label lengths per batch
    @tf.function
    def build_transition_matrix(self, expanded_labels, label_length, batch_size, max_length):
        A = tf.zeros((batch_size, max_length, max_length), dtype=tf.float32)
        #A += tf.eye(num_rows=max_length, batch_shape=[batch_size])
        #A += tf.linalg.diag(tf.ones([max_length - 1], dtype=tf.float32), k=-1)
        # Create comparison mask for A[j, j-2] == 1 if expanded_labels[j] != expanded_labels[j-2]
        comparison_mask = tf.not_equal(expanded_labels[:, :-2], expanded_labels[:, 2:])
        comparison_mask = tf.cast(comparison_mask, dtype=tf.float32)
        comparison_mask = tf.cast(comparison_mask, dtype=tf.bool)
        comparison_mask = tf.repeat(tf.expand_dims(comparison_mask, axis=1), max_length, axis=1)
        comparison_mask= tf.concat([comparison_mask, tf.zeros((batch_size, max_length, 2), dtype=tf.bool)], axis=-1)
        valid_transit = tf.linalg.diag(tf.ones(max_length - 2), k=-2)
        valid_transit = tf.where(comparison_mask, valid_transit, tf.fill((max_length, max_length), 0.0))
        A += valid_transit
        return A
    
    @tf.function
    def build_UB_matrix(self, logits, expanded_labels, batch_size, max_length, max_time):
        """
        Gather the indices from softmaxed logits based on expanded_labels.
        
        Args:
            logits ((batch_size, t, num_labels)): Softmaxed logits.
            expanded_labels ((batch_size, max_length)): Labels with blanks inserted.
            batch_size ((1)): The batch size.
            max_length ((1)): An integer that specifies the maximum length of the expanded labels.
                              Equals tf.reduce_max(expanded_length).
            max_time ((1)): An integer that specifies the maximum time steps that have to be gathered.
                            Equals tf.reduce_max(input_length).
             
        Returns:
            Tensor ((batch_size, max_time, max_length)): Values for every time step and label.
        """
        expanded_indices = tf.repeat(expanded_labels, max_time, axis=0)
        expanded_indices = tf.reshape(expanded_indices, [batch_size, max_time, max_length])
        UB = tf.gather(logits, expanded_indices, axis=2, batch_dims=2)
        return UB
    
    @tf.function
    def build_initial_distribution_matrix(self, batch_size, max_length):
        """
        The first two entries are 1, the rest is 0.
        
        Args:
            batch_size ((1)): The batch size.
            max_length ((1)): An integer that specifies the maximum length of the expanded labels.
                              Equals tf.reduce_max(expanded_length).
             
        Returns:
            Tensor ((batch_size, max_length)): The start states, either blank or the first actual label.
        """
        ones = tf.ones([batch_size, 2], dtype=tf.float32)
        zeros = tf.zeros([batch_size, max_length - 2], dtype=tf.float32)
        initial_distribution = tf.concat([ones, zeros], axis=1)
        return initial_distribution
    
    @tf.function
    def calculate_alpha(self, A_sparse, UB, initial_distribution, batch_size, max_length, max_time):
        """
        Forward probabilities. Calculated in log space. Calculated over the full range max_time, max_length for
        every entry in the batch.
        
        Args:
            A_sparse ((batch_size, max_length)): Positions where expanded_labels[j] are not equal to expanded_labels[j-2].
            UB ((batch_size, max_time, max_length)): Values for every time step and label.
            initial_distribution ((batch_size, max_length)): The start states, either blank or the first actual label.
            batch_size ((1)): The batch size.
            max_length ((1)): An integer that specifies the maximum length of the expanded labels.
                              Equals tf.reduce_max(expanded_length).
            max_time ((1)): An integer that specifies the maximum time steps that have to be gathered.
                            Equals tf.reduce_max(input_length).
        
        Returns:
            Tensor ((batch_size, max_time, max_length)): Forward probabilities in log space.
        """
        # Convert probabilities to log space with stability
        #A = tf.where(A > 0, tf.math.log(A), -1e6)
        UB = tf.where(UB > 0, tf.math.log(UB), 0)
        initial_distribution = tf.where(initial_distribution > 0, tf.math.log(initial_distribution), -1e6)
        
        # Extract diagonal, subdiagonal 1, and subdiagonal 2 from A
        #A_diag = tf.linalg.diag_part(A)  # Shape: (batch_size, max_length)
        #A_subdiag_1 = tf.pad(tf.linalg.diag_part(A[:, 1:, :-1]), [[0, 0], [1, 0]], constant_values=0)
        #A_subdiag_2 = tf.pad(tf.linalg.diag_part(A[:, 2:, :-2]), [[0, 0], [2, 0]], constant_values=0)
        #A_contributions = tf.stack([A_diag, A_subdiag_1, A_subdiag_2], axis=-1)  # (batch_size, max_length, 3)
        
        # F_prev has to be padded to the correct length
        subdiagonal_1_padding = tf.fill((batch_size, 1), -1e6)
        subdiagonal_2_padding = tf.fill((batch_size, 2), -1e6)
        
        # Define step function for tf.scan
        @tf.function
        def step_fn(F_prev, t):
            # Contributions from diagonal
            #diagonal = F_prev
            
            # Contributions from subdiagonal 1. Pad the first entry of F_prev.
            subdiagonal_1 = tf.concat([subdiagonal_1_padding, F_prev[:, :-1]], axis=1)
            
            # Contributions from subdiagonal 2
            subdiagonal_2 = tf.concat([subdiagonal_2_padding, F_prev[:, :-2] + A_sparse[:, 2:]], axis=1)
            
            # Combine contributions in log space
            F_current = tf.reduce_logsumexp(
                [F_prev, subdiagonal_1, subdiagonal_2],
                axis=0  # Combine over the 3 contributions
            ) + UB[:, t, :]  # Add UB for the current time step
            
            return F_current
        
        # Initialise F[0]
        F_init = initial_distribution + UB[:, 0, :]  # Shape: (batch_size, max_length)
        
        # Use tf.scan to compute alpha over time
        time_steps = tf.range(1, max_time)  # Time steps: [1, 2, ..., max_time - 1]
        alpha = tf.scan(
            fn=step_fn,
            elems=time_steps,  # Iterate over time indices
            initializer=F_init
        )
        
        # Concatenate the initial step and the scanned results
        alpha = tf.concat([tf.expand_dims(F_init, axis=0), alpha], axis=0)  # (max_time, batch_size, max_length)
        
        # Transpose to desired shape (batch_size, max_time, max_length)
        alpha = tf.transpose(alpha, perm=[1, 0, 2])
        return alpha
    
    @tf.function
    def loss(self, alpha, input_length, expanded_length, batch_size):
        """
        Calculate the loss value from the last possible positions, either
        the last actual label or the last blank.
        
        Args:
            alpha ((batch_size, max_time, max_length)): Forward probabilities in log space.
            input_length ((batch_size)): Time steps considered.
            expanded_length ((batch_size)): The length of the expanded labels after blank_index has been inserted.
                                            Equals 2 * label_length - 1.
            batch_size ((1)): The batch size.
        
        Returns:
            Tensor ((batch_size)): Loss values.
        """
        # Create a batch index tensor: [0, 1, ..., batch_size-1]
        batch_indices = tf.range(batch_size, dtype=tf.int32)  # Shape: (batch_size,)
        
        # Gather the values alpha[input_length, label_length]
        gather_indices_main = tf.stack([batch_indices, input_length - 1, expanded_length - 1], axis=1)  # Shape: (batch_size, 3)
        alpha_main = tf.gather_nd(alpha, gather_indices_main)  # Shape: (batch_size,)
        
        # Gather the values alpha[input_length, label_length - 1]
        gather_indices_minus1 = tf.stack([batch_indices, input_length - 1, expanded_length - 2], axis=1)  # Shape: (batch_size, 3)
        alpha_minus1 = tf.gather_nd(alpha, gather_indices_minus1)  # Shape: (batch_size,)
        
        values = tf.stack([alpha_minus1, alpha_main], axis=-1)
        
        return -tf.reduce_logsumexp(values, axis=-1)
    
    @tf.function
    def __call__(self, labels, logits, label_length, input_length, blank_index=0):
        # Prepare inputs
        logits = tf.nn.softmax(logits)
        
        # Prepare reused variables
        batch_size = tf.shape(logits)[0]
        expanded_length = 2 * label_length + 1
        max_length = tf.reduce_max(expanded_length)
        max_time = tf.reduce_max(input_length)
        
        # Calculate matrices
        expanded_labels = self.expand_labels(labels, label_length, blank_index, batch_size, expanded_length, max_length)
        #print(expanded_labels)
        #A = self.build_transition_matrix(expanded_labels, label_length, batch_size, max_length)
        A_sparse = self.build_sparse_transition_matrix(expanded_labels, batch_size, max_length)
        #print(A)
        UB = self.build_UB_matrix(logits, expanded_labels, batch_size, max_length, max_time)
        #print(UB)
        initial_distribution = self.build_initial_distribution_matrix(batch_size, max_length)
        #print(initial_distribution)
        alpha = self.calculate_alpha(A_sparse, UB, initial_distribution, batch_size, max_length, max_time)
        #print(alpha)
        loss = self.loss(alpha, input_length, expanded_length, batch_size)
        #print(loss)
        return loss


batch_size = 32
time_ = 834
num_labels = 5 # blank, A, C, G, T
min_num_labels = 350
max_num_labels = 500
blank_index = 0

logits = tf.random.uniform((batch_size, time_, num_labels), minval=-5, maxval=5)

def generate_row():
    length = tf.random.uniform(shape=[], minval=min_num_labels, maxval=max_num_labels+1, dtype=tf.int32)  # Random length
    values = tf.random.uniform(shape=[length], minval=1, maxval=num_labels, dtype=tf.int32)  # Random values between 1 and num_labels
    padding = tf.zeros([max_num_labels - length], dtype=tf.int32)  # Padding with zeros
    return tf.concat([values, padding], axis=0)  # Combine values and padding
# Generate the tensor
labels = tf.stack([generate_row() for _ in range(batch_size)])

logit_length = tf.shape(logits)[1] * tf.ones(tf.shape(logits)[0], dtype=tf.int32)

label_length = tf.reduce_sum(tf.cast(labels != 0, tf.int32), axis=1)

H = HMM()
hmm_loss = H(labels, logits, label_length, logit_length, blank_index=blank_index)

ctc_loss = tf.nn.ctc_loss(
    labels = labels,
    logits = logits,
    label_length = label_length,
    logit_length = logit_length,
    logits_time_major = False,
    blank_index = blank_index)

print("HMM Loss:", hmm_loss)
print("CTC Loss:", ctc_loss)

# Small example
#logits = tf.random.uniform((3, 8, 3), minval=-5, maxval=5)
#labels = tf.constant([[1,2,0,0,0], [1, 1, 2, 0, 0], [2,2,1,2,1]], dtype=tf.int32)
#logit_length = tf.shape(logits)[1] * tf.ones(tf.shape(logits)[0], dtype=tf.int32)
#label_length = tf.constant([2, 3, 5], dtype=tf.int32)

#hmm_loss = H(labels, logits, label_length, logit_length, blank_index=0)

#ctc_loss = tf.nn.ctc_loss(labels = labels, logits = logits, label_length = label_length, logit_length = logit_length, logits_time_major = False, blank_index = 0)

#print("HMM Loss:", hmm_loss)
#print("CTC Loss:", ctc_loss)
