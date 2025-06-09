import tensorflow as tf

"""
TODO: Still under active development.

Calculates the correct values, but does not lead to the correct training results yet.
There might still be issues with calculating the gradients correctly.
"""

class HMM:
    
    """
    Example usage:
    batch_size = 32
    time_ = 834
    num_labels = 5120
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
    
    H = HMM(batch_size, max_num_labels, time_, normalise=True, epsilon=-tf.float32.max)
    loss = H(labels, logits, label_length, logit_length)
    """
    
    def __init__(self, batch_size, padded_label_length, max_time, state_len=5, n_base=4, alphabet=["N", "A", "C", "G", "T"], normalise=True, epsilon=-1e6):
        # The batch size
        self.batch_size = batch_size
        # Non dynamic shape of the state dimension
        self.max_length = padded_label_length - (state_len - 1)
        # Non dynamic shape of alphas time dimension
        self.max_time = max_time + 1
        
        self.normalise = normalise
        
        # Large negative value for log space, e.g. -tf.float32.max
        self.epsilon = epsilon
        
        # CRF parameters
        self.state_len = state_len
        self.size_alphabet = tf.size(alphabet)
        self.alphabet = alphabet
        self.n_base = n_base
        self.num_states = 1024
        self.n_crf = padded_label_length - (state_len - 1)
        # Create an exponent tensor [state_len-1, state_len-2, ..., 0]
        exponents = tf.range(self.state_len - 1, -1, -1, dtype=tf.int32)
        # Compute n_base ** exponents
        self.powers = tf.pow(n_base, exponents)
        # Index
        i = tf.range(self.num_states, dtype=tf.int32)  # shape [M]
        first_col = tf.reshape(i, [self.num_states, 1])
        # For move transitions (r=1,...,n_base)
        r = tf.range(1, self.size_alphabet, dtype=tf.int32)  # shape [n_base]
        # Note: self.n_base**(self.state_len-1) is 256 for state_len=5, n_base=4.
        move = (i // self.n_base)[:, None] + (r - 1)[None, :] * (self.n_base ** (self.state_len - 1))
        idx = tf.concat([first_col, move], axis=1)  # shape [M, R]
        self.idx = tf.reshape(idx, [-1])
        # Normalisation
        # FWD
        self.alpha_norm_init = tf.zeros([self.batch_size, self.num_states], dtype=tf.float32)
        # Precompute segment IDs for unsorted_segment operations.
        # For each state i and symbol r, flat_idx gives the destination state.
        # To process the entire batch without mixing entries, add an offset of b*M for each batch element.
        batch_offsets = tf.reshape(tf.range(self.batch_size, dtype=tf.int32) * self.num_states, [self.batch_size, 1])
        # Tile flat_idx for each batch: shape becomes [B, M*R]
        seg_ids = tf.reshape(self.idx, [1, self.num_states * self.size_alphabet]) + batch_offsets  # shape: [B, M*R]
        self.seg_ids = tf.reshape(seg_ids, [-1])  # shape: [B*M*R]
        self.num_segments = self.batch_size * self.num_states  # one segment per destination state per batch element
        self.states_times_size_alphabet = self.num_states * self.size_alphabet
        # BWD
        # Initialize beta at time T: log(1) = 0
        self.beta_norm_init = tf.zeros([self.batch_size, self.num_states], dtype=tf.float32)
        # Recover the index mapping as used in the forward pass.
        # self.idx is stored as a flat vector; reshape it to [num_states, size_alphabet]
        idx_map = tf.reshape(self.idx, [self.num_states, self.size_alphabet])
        # First, tile idx_map over the batch.
        self.idx_rev_tiled = tf.tile(tf.expand_dims(idx_map, axis=0), [self.batch_size, 1, 1])  # shape: [B, M, R]
        
        # Initial distribution
        self.initial = tf.concat([tf.fill([self.batch_size, 1], 0.0), tf.fill([self.batch_size, self.max_length - 1], self.epsilon)], axis=1)
        
        # Used in the calculation of the loss
        # Create a batch index tensor: [0, 1, ..., batch_size-1]
        self.batch_indices = tf.range(self.batch_size, dtype=tf.int32)  # Shape: (batch_size,)
        
        # Decoding
        pattern = ["", "A", "C", "G", "T"]
        self.index_to_char = {i: pattern[i % len(pattern)] for i in range(1024 * len(pattern))}
    
    @tf.function
    def create_crf_indices(self, labels):
        """
        Create the crf indices by encoding k-mers in the labels and their transitions as positions in the CRF.
        
        Args:
            labels ((batch_size, padded_label_length)): The labels to predict during training.
        
        Returns:
            stay_indices ((batch_size, max_length)): Indices for the k-mers.
            move_indices ((batch_size, max_length - 1)): Indices for the transitions.
        """
        labels = tf.clip_by_value(labels - 1 , 0, self.size_alphabet)
        # Use broadcasting to apply powers to the correct slices of targets
        frames = tf.signal.frame(labels, frame_length=self.state_len, frame_step=1, axis=1)
        stay_indices = tf.reduce_sum(frames * self.powers, axis=-1)
        stay_indices = stay_indices * self.size_alphabet
        move_indices = stay_indices[:, 1:] + labels[:, :self.n_crf - 1] + 1
        return stay_indices, move_indices
    
    @tf.function
    def gather_crf_scores(self, logits, stay_indices, move_indices):
        """
        Use the indices to gather the correct values from the logits.
        
        Args:
            logits ((batch_size, max_time - 1, 5120)): Model output.
            stay_indices ((batch_size, max_length)): Indices for the k-mers.
            move_indices ((batch_size, max_length - 1)): Indices for the transitions.
        
        Returns:
            stay_scores ((batch_size, max_time - 1, max_length)): Stay scores for each time step.
            move_scores ((batch_size, max_time - 1, max_length - 1)): Move scores for each time step.
        """
        stay_scores = tf.gather(logits, stay_indices, axis=2, batch_dims=1)
        move_scores = tf.gather(logits, move_indices, axis=2, batch_dims=1)
        return stay_scores, move_scores
    
    @tf.function
    def reverse_scores(self, stay_scores, move_scores, crf_label_length):
        """
        Reverse the scores along the time and state dimension.
        
        Args:
            stay_scores ((batch_size, max_time - 1, max_length)): Stay scores for each time step.
            move_scores ((batch_size, max_time - 1, max_length - 1)): Move scores for each time step.
            crf_label_length ((batch_size)): The length for the CRF labels without padding.
        
        Returns:
            stay_scores_rev ((batch_size, max_time - 1, max_length)): Reversed stay scores for each time step.
            move_scores_rev ((batch_size, max_time - 1, max_length - 1)): Reversed move scores for each time step.
        """
        stay_scores_rev = tf.reverse_sequence(stay_scores, crf_label_length, seq_axis=2, batch_axis=0)
        move_scores_rev = tf.reverse_sequence(move_scores, crf_label_length-1, seq_axis=2, batch_axis=0)
        stay_scores_rev = tf.reverse(stay_scores_rev, axis=[1])
        move_scores_rev = tf.reverse(move_scores_rev, axis=[1])
        return stay_scores_rev, move_scores_rev
    
    @tf.function
    def calculate_normalisation_fwd(self, scores):
        """
        Normalisation forward pass. This is an attempt at reconstruction, the original function used is not accessible to us.
        
        Args:
            scores ((max_time - 1, batch_size, 5120)): Unnormalised scores.
        
        Returns:
            logZ ((batch_size)): LogZ value for each entry in the batch.
            Ms_grad ((max_time - 1, batch_size, 1024, 5)): Forward values.
        """
        # Reshape scores from [T, B, S] to [T, B, M, R]
        scores = tf.reshape(scores, [self.max_time-1, self.batch_size, self.num_states, self.size_alphabet])
        
        # Initialize forward message alpha (in log-space, so zeros mean log(1))
        alpha = self.alpha_norm_init
        
        # Prepare a TensorArray to store per-time-step messages.
        Ms_grad_ta = tf.TensorArray(dtype=tf.float32, size=self.max_time-1, clear_after_read=False,
                                      element_shape=[self.batch_size, self.num_states, self.size_alphabet])
        
        # Forward recursion: for each time step, update alpha.
        for t in range(self.max_time-1):
            # For time step t, compute contributions:
            # contributions[b, i, r] = alpha[b, i] + scores[t, b, i, r]
            contributions = tf.expand_dims(alpha, -1) + scores[t]  # shape: [B, M, R]
            # Store these contributions.
            Ms_grad_ta = Ms_grad_ta.write(t, contributions)
            
            # Flatten contributions to shape [B, M*R] then to [B*M*R].
            flat_contrib = tf.reshape(contributions, [self.batch_size, self.states_times_size_alphabet])
            flat_contrib = tf.reshape(flat_contrib, [-1])
            
            # Compute numerically stable log-sum-exp for each destination segment.
            max_vals = tf.math.unsorted_segment_max(flat_contrib, self.seg_ids, self.num_segments)
            gathered_max = tf.gather(max_vals, self.seg_ids)
            sum_exp = tf.math.unsorted_segment_sum(tf.exp(flat_contrib - gathered_max), self.seg_ids, self.num_segments)
            new_alpha_flat = tf.math.log(sum_exp) + max_vals  # shape: [B*M]
            alpha = tf.reshape(new_alpha_flat, [self.batch_size, self.num_states])
        
        # Stack the TensorArray: Ms_grad of shape [T, B, M, R]
        Ms_grad = Ms_grad_ta.stack()
        
        # Compute the final log partition function for each batch by summing over the last time step’s messages.
        # For each batch b, logZ[b] = reduce_logsumexp over states and symbols.
        logZ = tf.reduce_logsumexp(Ms_grad[-1], axis=[1, 2])
        return logZ, Ms_grad
    
    @tf.function
    def calculate_normalisation_bwd(self, scores):
        """
        Compute the backward (beta) messages and logZ via a backward pass.
        
        Args:
            scores ((max_time - 1, batch_size, 5120)): Unnormalised scores.
        
        Returns:
            logZ_bwd ((batch_size)): LogZ value for each entry in the batch.
            beta_all ((max_time, batch_size, 1024)): Backward values.
        """
        # Reshape scores from [T, B, S] to [T, B, M, R]
        scores = tf.reshape(scores, [self.max_time - 1, self.batch_size, self.num_states, self.size_alphabet])
        
        # Prepare a TensorArray to store beta messages (for t = 0,...,T)
        beta_ta = tf.TensorArray(dtype=tf.float32, size=self.max_time, clear_after_read=False,
                                   element_shape=[self.batch_size, self.num_states])
        beta_ta = beta_ta.write(self.max_time - 1, self.beta_norm_init)
        
        # Backward recursion: for t = T-1 down to 0.
        for t in reversed(range(self.max_time - 1)):
            # Read the backward message at time t+1.
            beta_next = beta_ta.read(t+1)  # shape: [B, M]
            # For each state i and symbol r, we need to add:
            #    score[t, b, i, r] + beta_next[b, idx_map[i, r]]
            # Gather beta_next for each (i, r) pair.
            # Here we use tf.gather with batch_dims=1 (available in TF2).
            beta_next_gathered = tf.gather(beta_next, self.idx_rev_tiled, axis=1, batch_dims=1)  # shape: [B, M, R]
            # Compute the local contributions at time t.
            # contributions[b, i, r] = scores[t, b, i, r] + beta_next_gathered[b, i, r]
            contributions = scores[t] + beta_next_gathered  # shape: [B, M, R]
            # For each state i (and batch b), reduce over r with logsumexp.
            beta_t = tf.reduce_logsumexp(contributions, axis=-1)  # shape: [B, M]
            beta_ta = beta_ta.write(t, beta_t)
        
        # Stack beta messages to get a tensor of shape [T+1, B, M]
        beta_all = beta_ta.stack()
        
        # Compute the backward log partition function:
        # Typically, one computes logZ as the logsumexp over the initial beta message.
        logZ_bwd = tf.reduce_logsumexp(beta_all[0], axis=-1)  # shape: [B]
        
        return logZ_bwd, beta_all
    
    @tf.custom_gradient
    def calculate_normalisation(self, logits_time_major):
        """
        Forward backward pass for the normalisation of the scores. Customised calculation of the gradients, as is assumed for Bonito.
        """
        loss, Ms_grad = self.calculate_normalisation_fwd(logits_time_major)
        
        def grad(dy):
            _, betas = self.calculate_normalisation_bwd(logits_time_major)
            #tf.print("betas:", betas)
            temp = Ms_grad + tf.expand_dims(betas[1:], axis=-1)  # shape: [T, B, M, R]
            # Flatten last two dims and apply softmax.
            temp_flat = tf.reshape(temp, [self.max_time - 1, self.batch_size, self.num_states*self.size_alphabet])
            grad_scores = tf.nn.softmax(temp_flat, axis=-1)
            grad_scores = tf.reshape(grad_scores, [self.max_time - 1, self.batch_size, self.num_states, self.size_alphabet])
            # Multiply by the upstream gradient dy (broadcast over time, states, symbols)
            dy_expanded = tf.reshape(dy, [1, self.batch_size, 1, 1])
            #tf.print("dy_expanded:", dy_expanded)
            final_grad = dy_expanded * grad_scores
            # Reshape final_grad back to the original scores shape: [T, B, M*R]
            final_grad = tf.reshape(final_grad, [self.max_time - 1, self.batch_size, self.num_states*self.size_alphabet])
            #tf.print("final_grad:", final_grad)
            return final_grad
        
        return loss, grad
    
    @tf.function
    def calculate_fwd_bwd(self, stay_scores, move_scores, stay_scores_rev, move_scores_rev, crf_label_length):
        """
        Forward backward pass.
        
        Args:
            stay_scores ((batch_size, max_time - 1, max_length)): Stay scores for each time step.
            move_scores ((batch_size, max_time - 1, max_length - 1)): Move scores for each time step.
            stay_scores_rev ((batch_size, max_time - 1, max_length)): Reversed stay scores for each time step.
            move_scores_rev ((batch_size, max_time - 1, max_length - 1)): Reversed move scores for each time step.
            crf_label_length ((batch_size)): The length for the CRF labels without padding.
        
        Returns:
            alpha ((max_time, batch_size, max_length)): Forward values.
            beta ((max_time, batch_size, max_length)): Backward values.
        """
        # Initialise alpha
        alpha_ta = tf.TensorArray(tf.float32, size=self.max_time, clear_after_read=False, element_shape=[self.batch_size, self.max_length])
        alpha_ta = alpha_ta.write(0, self.initial)
        
        beta_ta = tf.TensorArray(tf.float32, size=self.max_time, clear_after_read=False, element_shape=[self.batch_size, self.max_length])
        beta_ta = beta_ta.write(0, self.initial)
        
        for t in range(1, self.max_time):
            alpha_prev = alpha_ta.read(t-1)
            beta_prev = beta_ta.read(t-1)
            
            # First state
            alpha0 = alpha_prev[:, :1] + stay_scores[:, t-1, :1]
            beta0 = beta_prev[:, :1] + stay_scores_rev[:, t-1, :1]
            
            alpha_stay = alpha_prev[:, 1:] + stay_scores[:, t-1, 1:]
            alpha_move = alpha_prev[:, :-1] + move_scores[:, t-1, :]
            
            beta_stay = beta_prev[:, 1:] + stay_scores_rev[:, t-1, 1:]
            beta_move = beta_prev[:, :-1] + move_scores_rev[:, t-1, :]
            
            alpha_rest = tf.reduce_logsumexp(tf.stack([alpha_stay, alpha_move], axis=0), axis=0)
            beta_rest = tf.reduce_logsumexp(tf.stack([beta_stay, beta_move], axis=0), axis=0)
            
            alpha_t = tf.concat([alpha0, alpha_rest], axis=1)
            beta_t = tf.concat([beta0, beta_rest], axis=1)
            
            alpha_ta = alpha_ta.write(t, alpha_t)
            beta_ta = beta_ta.write(t, beta_t)
        
        alpha = alpha_ta.stack()
        beta = beta_ta.stack()
        beta = tf.reverse_sequence(beta, crf_label_length, seq_axis=2, batch_axis=1)
        beta = tf.reverse(beta, axis=[0])
        return alpha, beta
    
    @tf.function
    def calculate_fwd(self, stay_scores, move_scores):
        """
        Calculation of the foward pass only.
        """
        # Initialise alpha
        alpha_ta = tf.TensorArray(tf.float32, size=self.max_time, clear_after_read=False, element_shape=[self.batch_size, self.max_length])
        alpha_ta = alpha_ta.write(0, self.initial)
        
        for t in range(1, self.max_time):
            alpha_prev = alpha_ta.read(t-1)
            
            # First state
            alpha0 = alpha_prev[:, :1] + stay_scores[:, t-1, :1]
            
            alpha_stay = alpha_prev[:, 1:] + stay_scores[:, t-1, 1:]
            alpha_move = alpha_prev[:, :-1] + move_scores[:, t-1, :]
            
            alpha_rest = tf.reduce_logsumexp(tf.stack([alpha_stay, alpha_move], axis=0), axis=0)
            
            alpha_t = tf.concat([alpha0, alpha_rest], axis=1)
            
            alpha_ta = alpha_ta.write(t, alpha_t)
        
        alpha = alpha_ta.stack()
        return alpha
    
    @tf.function
    def calculate_bwd(self, stay_scores_rev, move_scores_rev, crf_label_length):
        """
        Calculation of the backward pass only
        """
        beta_ta = tf.TensorArray(tf.float32, size=self.max_time, clear_after_read=False, element_shape=[self.batch_size, self.max_length])
        beta_ta = beta_ta.write(0, self.initial)
        
        for t in range(1, self.max_time):
            beta_prev = beta_ta.read(t-1)
            
            # First state
            beta0 = beta_prev[:, :1] + stay_scores_rev[:, t-1, :1]
            
            beta_stay = beta_prev[:, 1:] + stay_scores_rev[:, t-1, 1:]
            beta_move = beta_prev[:, :-1] + move_scores_rev[:, t-1, :]
            
            beta_rest = tf.reduce_logsumexp(tf.stack([beta_stay, beta_move], axis=0), axis=0)
            
            beta_t = tf.concat([beta0, beta_rest], axis=1)
            
            beta_ta = beta_ta.write(t, beta_t)
        
        beta = beta_ta.stack()
        beta = tf.reverse_sequence(beta, crf_label_length, seq_axis=2, batch_axis=1)
        beta = tf.reverse(beta, axis=[0])
        return beta
    
    @tf.function
    def loss_alpha(self, alpha, crf_label_length, input_length):
        """
        Calculate the loss value from the last possible position.
        
        Args:
            alpha ((max_time, batch_size, max_length)): Forward values.
            crf_label_length ((batch_size)): The length for the CRF labels without padding.
            input_length ((batch_size)): Time steps considered.
        
        Returns:
            alpha_values ((batch_size)): Loss values.
        """
        # Gather the values alpha[input_length, label_length]
        gather_indices = tf.stack([input_length - 1, self.batch_indices, crf_label_length - 1], axis=1)
        alpha_values = tf.gather_nd(alpha, gather_indices)  # Shape: (batch_size)
        return alpha_values
    
    @tf.function
    def loss_beta(self, beta):
        """
        Calculate the loss value from the first position.
        
        Args:
            beta ((max_time, batch_size, max_length)): Backward values.
        
        Returns:
            beta_values ((batch_size)): Loss values.
        """
        beta_values = -tf.reduce_logsumexp(beta[0, :, :1], axis=1)
        return beta_values
    
    @tf.function
    def loss_beta2(self, beta):
        """
        Calculate the loss value from the first row.
        
        Args:
            beta ((max_time, batch_size, max_length)): Backward values.
        
        Returns:
            beta_values ((batch_size)): Loss values.
        """
        loss_beta = -tf.reduce_logsumexp(beta[0, :, :], axis=1)
        return loss_beta
    
    @tf.function
    def get_loss(self, alpha, beta, l):
        """
        Total loss.
        
        Args:
            alpha ((max_time, batch_size, max_length)): Forward values.
            beta ((max_time, batch_size, max_length)): Backward values.
            l ((batch_size)): Loss values for every entry in the batch.
        
        Returns:
            Tensor ((batch_size)): Total loss for every entry in the batch.
        """
        prob = alpha + beta
        prob = tf.reduce_logsumexp(prob, axis=-1) # Reduce over S
        prob = tf.reduce_logsumexp(prob, axis=0) # Reduce over T
        return -(prob - l)
    
    @tf.function
    def reconstruct_beta_components(self, beta, stay_scores, move_scores):
        """
        Given the full beta tensor and the original stay and move scores,
        reconstruct beta_stay and beta_move.
        
        Args:
            beta: Tensor of shape [T+1, batch, L] containing the overall backward scores.
            stay_scores: Tensor of shape [T, batch, L] for the stay transitions.
            move_scores: Tensor of shape [T, batch, L] for the move transitions.
        
        Returns:
            beta_stay: Tensor of shape [T, batch, L]
            beta_move: Tensor of shape [T, batch, L]
        """
        stay_scores = tf.transpose(stay_scores, perm=[1, 0, 2])
        move_scores = tf.transpose(move_scores, perm=[1, 0, 2])
        # Compute beta_stay using the recurrence:
        # beta_stay[t, l] = stay_scores[t, l] + beta[t+1, l]
        beta_stay = stay_scores + beta[1:]
        
        # Compute beta_move:
        # For each state l, beta_move[t, l] = move_scores[t, l] + beta[t+1, l+1]
        # We need to pad beta[t+1] on the last dimension since l+1 is undefined for the last label.
        beta_next = beta[1:]  # shape: [T, batch, L]
        # Shift beta_next along the last dimension:
        beta_next_shifted = tf.concat([beta_next[:, :, 1:], tf.fill([self.max_time-1, self.batch_size, 1], self.epsilon)], axis=2)
        beta_move = move_scores + beta_next_shifted[:, :, :-1]
        pad = tf.fill([self.max_time-1, self.batch_size, 1], self.epsilon)
        beta_move = tf.concat([beta_move, pad], axis=2)
        return beta_stay, beta_move
    
    @tf.function
    def mask_tensor(self, first, crf_label_length):
        """
        Mask invalid positions.
        """
        # Create a range tensor for the S dimension
        range_S = tf.range(self.max_length)
        
        # Expand label_length for broadcasting
        label_length_exp = tf.expand_dims(crf_label_length, axis=1)  # Shape (batch_size, 1)
        
        # Create a mask where indices >= label_length
        mask = tf.greater_equal(range_S, label_length_exp)  # Shape (batch_size, S)
        
        # Expand and broadcast the mask to match `first` shape
        mask = tf.expand_dims(mask, axis=0)  # Shape (1, batch_size, S)
        mask = tf.broadcast_to(mask, [self.max_time-1, self.batch_size, self.max_length])  # Shape (T, batch_size, S)
        
        # Apply the mask using tf.where
        return tf.where(mask, self.epsilon, first)
    
    @tf.function
    def mask_tensor2(self, first, crf_label_length):
        """
        Mask invalid positions.
        """
        # Create a range tensor for the S dimension
        range_S = tf.range(self.max_length)
        
        # Expand label_length for broadcasting
        label_length_exp = tf.expand_dims(crf_label_length, axis=1)  # Shape (batch_size, 1)
        
        # Create a mask where indices >= label_length
        mask = tf.greater_equal(range_S, label_length_exp)  # Shape (batch_size, S)
        
        # Expand and broadcast the mask to match `first` shape
        mask = tf.expand_dims(mask, axis=0)  # Shape (1, batch_size, S)
        mask = tf.broadcast_to(mask, [self.max_time, self.batch_size, self.max_length])  # Shape (T, batch_size, S)
        
        # Apply the mask using tf.where
        return tf.where(mask, self.epsilon, first)
    
    @tf.custom_gradient
    def crf_loss(self, stay_scores, move_scores, crf_label_length, input_length):
        """
        Forward backward pass. Customised calculation of the gradients, as is assumed for Bonito.
        """
        stay_scores_rev, move_scores_rev = self.reverse_scores(stay_scores, move_scores, crf_label_length)
        alpha, beta = self.calculate_fwd_bwd(stay_scores, move_scores, stay_scores_rev, move_scores_rev, crf_label_length)
        alpha = self.mask_tensor2(alpha, crf_label_length)
        beta = self.mask_tensor2(beta, crf_label_length)
        loss = self.loss_alpha(alpha, crf_label_length, input_length)
        
        def grad(upstream):
            # upstream has shape [batch] (one scalar per example)
            # Expand to broadcast over time and state dimensions.
            upstream_expanded = tf.reshape(upstream, [self.batch_size, 1, 1])
            # Compute logZ from beta (loss_beta returns a tensor of shape [batch])
            #logZ = -self.loss_beta(beta)  # now logZ is positive and has shape [batch]
            
            # Compute the posterior: p[t, b, l] = exp(alpha[t,b,l] + beta[t,b,l] - logZ[b])
            #posterior = alpha + beta - tf.reshape(logZ, [1, self.batch_size, 1])
            #posterior = tf.nn.softmax(posterior, axis=-1)
            # Now, since our forward pass used self.max_time-1 steps, we slice off the extra time step:
            #posterior = posterior[1:]  # shape becomes [self.max_time-1, batch, L]
            #posterior = tf.transpose(posterior, perm=[1, 0, 2])
            
            beta_stay, beta_move = self.reconstruct_beta_components(beta, stay_scores, move_scores)
            beta_stay = self.mask_tensor(beta_stay, crf_label_length)
            beta_move = self.mask_tensor(beta_move, crf_label_length)
            posterior = tf.nn.softmax(tf.concat([alpha[:-1] + beta_stay, alpha[:-1] + beta_move], axis=2), axis=2)
            posterior = tf.transpose(posterior, perm=[1, 0, 2])
            posterior = tf.reshape(posterior, [self.max_time-1, self.batch_size, 2, self.max_length])
            
            post1 = tf.transpose(posterior[:, :, 0], perm=[1, 0, 2])
            post2 = tf.transpose(posterior[:, :, 1, 1:], perm=[1, 0, 2])
            # For the "stay" scores (which have full label dimension L), the gradient is just:
            grad_stay_time_major = upstream_expanded * post1
            # For the "move" scores (which have one fewer state, L-1), we take a slice:
            grad_move_time_major = upstream_expanded * post2
            
            # If the original inputs are expected in time-major order, we return these.
            # Otherwise, if they are batch-major, you would transpose them.
            # In our example, assume the input to calculate_normalisation had shape
            # [self.max_time-1, batch, num_states*size_alphabet]. Then:
            grad_stay = grad_stay_time_major
            grad_move = grad_move_time_major
            # And we assume the same gradients for the reversed scores.
            #grad_stay_rev = grad_stay_time_major
            #grad_move_rev = grad_move_time_major
            
            # Return gradients for each input (for non-differentiable inputs, return None)
            return grad_stay, grad_move, None, None
        
        return loss, grad
    
    def decode(self, logits):
        """
        Decode the model output. Greedy strategy using the class with the maximum value. Multiples of five are not considered.
        
        Args:
            logits ((batch_size, time_steps, 5120)): The model output.
        
        Returns:
            List ((batch_size)): The DNA sequences as strings.
        """
        # Greedy decoding
        indices = tf.argmax(logits, axis=-1)
        indices = indices.numpy()
        
        decoded = ["".join(self.index_to_char[i] for i in row if i in self.index_to_char)
        for row in indices]
        return decoded
    
    @tf.function
    def __call__(self, labels, logits, label_length, input_length):
        """
        Calculate the CRF loss as it is done with the Bonito basecaller.
        
        Args:
            labels ((batch_size, padded_label_length)): The training labels. Padded with zeros.
            logits ((batch_size, max_time, 5120)): Values for every time step and class.
            label_length ((batch_size)): The actual label lengths without the padding.
            input_length ((batch_size)): Number of time steps to be considered.
        
        Returns:
            loss ((batch_size)): The total loss for every entry in the batch.
        """
        # Prepare inputs
        input_length = input_length + 1
        
        if self.normalise:
            logits_time_major = tf.transpose(logits, perm=[1, 0, 2])
            logZ = self.calculate_normalisation(logits_time_major)
            logits = logits - (tf.expand_dims(tf.expand_dims((logZ / (self.max_time - 1)), axis = 1), axis=2))
        else:
            logits = tf.nn.log_softmax(logits)
        
        stay_indices, move_indices = self.create_crf_indices(labels)
        stay_scores, move_scores = self.gather_crf_scores(logits, stay_indices, move_indices)
        #stay_scores_rev, move_scores_rev = self.reverse_scores(stay_scores, move_scores, crf_label_length)
        crf_label_length = label_length - (self.state_len - 1)
        loss = self.crf_loss(stay_scores, move_scores, crf_label_length, input_length)
        if self.normalise:
            loss = - (loss / tf.cast(label_length, dtype=tf.float32))
        return loss

