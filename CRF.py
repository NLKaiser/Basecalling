import tensorflow as tf

"""
Recreation of the Bonito CRF Loss in TensorFlow.
"""

class CRF:
    
    """
    Example usage:
    batch_size = 32
    time_ = 834
    num_labels = 5 # blank, A, C, G, T
    min_num_labels = 350
    max_num_labels = 500
    out_dim = 5120
    
    def generate_row():
        length = tf.random.uniform(shape=[], minval=min_num_labels, maxval=max_num_labels+1, dtype=tf.int32)  # Random length
        values = tf.random.uniform(shape=[length], minval=1, maxval=num_labels, dtype=tf.int32)  # Random values between 1 and num_labels
        padding = tf.zeros([max_num_labels - length], dtype=tf.int32)  # Padding with zeros
        return tf.concat([values, padding], axis=0)  # Combine values and padding
    # Generate the tensor
    labels = tf.stack([generate_row() for _ in range(batch_size)])
    logits = tf.random.uniform((batch_size, time_, out_dim), minval=-5, maxval=5)
    label_length = tf.reduce_sum(tf.cast(labels != 0, tf.int32), axis=1)
    
    H = CRF(32, 500, 834)
    loss = H(logits, labels, label_length)
    """
    
    def __init__(self, batch_size, padded_label_length, max_time, dtype=tf.float32):
        self.batch_size = batch_size
        self.max_length = padded_label_length - (5 - 1) # hard coded state_len
        self.max_time = max_time
        
        self.dtype = dtype
        
        self.epsilon = self.dtype.min
        
        # Index construction
        self.alphabet = ["N", "A", "C", "G", "T"]
        self.n_base = 4
        self.state_len = 5
        self.num_states = self.n_base ** self.state_len
        self.nz = 5
        self.out_size = self.num_states * self.nz
        
        first_col = tf.range(self.n_base ** self.state_len, dtype=tf.int32)[:, tf.newaxis]
        repeated = tf.repeat(tf.range(self.n_base ** self.state_len, dtype=tf.int32), repeats=self.n_base)
        next_cols = tf.reshape(repeated, (self.n_base, -1))
        next_cols = tf.transpose(next_cols)
        self.idx = tf.concat([first_col, next_cols], axis=1)
        
        # Index flat
        idx_flat   = tf.reshape(self.idx, [-1])                     # [C*NZ]
        idx_flat = tf.argsort(idx_flat, axis=0, stable=True) # [C*NZ]
        idx_flat_rs = tf.reshape(idx_flat, (self.num_states, self.nz))
        # Pre-broadcast idx_matrix for batching
        # For gathering previous states: need shape [N, C, NZ]
        self.prev_state_bcast = tf.broadcast_to(idx_flat_rs // self.nz, [self.batch_size, self.num_states, self.nz])
        # For gathering transition scores: need shape [N, C, NZ]
        self.trans_idx_bcast = tf.broadcast_to(idx_flat_rs, [self.batch_size, self.num_states, self.nz])
        
        # Label Encoding
        self.size_alphabet = tf.size(self.alphabet)
        self.n_crf = padded_label_length - (self.state_len - 1)
        # Create an exponent tensor [state_len-1, state_len-2, ..., 0]
        exponents = tf.range(self.state_len - 1, -1, -1, dtype=tf.int32)
        # Compute n_base ** exponents
        self.powers = tf.pow(self.n_base, exponents)
        
        # Others
        self.zero_num_states = tf.zeros([self.batch_size, self.num_states], dtype=self.dtype)
        a0 = tf.concat([
            tf.cast(tf.fill([self.batch_size, 1], 0.0), dtype=self.dtype),                      # initial alpha[0, :, 0] = S.one
            tf.fill([self.batch_size, self.n_crf - 1], self.epsilon)      # rest = S.zero
        ], axis=1)
        self.a0 = tf.cast(a0, self.dtype)
        beta_T = tf.fill([self.batch_size, self.n_crf], self.epsilon)
        self.beta_T = tf.cast(beta_T, self.dtype)
        self.batch_range = tf.range(self.batch_size, dtype=tf.int32)
        self.batch_zeros = tf.zeros([self.batch_size], dtype=self.dtype)
        batch_one_epsilon = tf.fill([self.batch_size, 1], self.epsilon)
        self.batch_one_epsilon = tf.cast(batch_one_epsilon, dtype=self.dtype)
        
        # Decoding
        self._chars = tf.constant(["", "A", "C", "G", "T"], dtype=tf.string)
        self.n_pre_context_bases = self.state_len - 1
        self.n_post_context_bases = 1
    
    @tf.function(jit_compile=True)
    def logZ_fwd(self, Ms):
        """
        FWD for normalisation.
        
        Args:
            Ms ((max_time, batch_size, num_states, nz)): Reshaped scores, states with their transitions.
        
        Returns:
            logZ ((batch_size)): Log-partition function for each entry in the batch.
            Ms_grad ((max_time, batch_size, num_states, nz)): Intermediate values for gradient computation.
        """
        
        # Prepare TensorArray to collect s at each timestep
        ta_grad = tf.TensorArray(dtype=self.dtype, size=self.max_time, element_shape=[self.batch_size, self.num_states, self.nz])
        
        # Initial state a: (N, S); All zeros.
        a = self.zero_num_states
        
        # Python for-loop over timesteps
        for t in range(self.max_time):
            # (N, S, NZ)
            m_t = Ms[t]
            # self.idx (S, NZ)
            # Log probs of valid incoming transitions
            # (N, S, NZ)
            prev_vals = tf.gather(a, self.idx, axis=1)
            # (N, S, NZ)
            # Incoming forward values + transition scores
            s = prev_vals + m_t
            # Update forward value
            # (N, S)
            a = tf.reduce_logsumexp(s, axis=2)
            # Write to TensorArray
            ta_grad = ta_grad.write(t, s)
        
        # Stack collected gradients: (T, N, S, NZ)
        Ms_grad = ta_grad.stack()
        
        # Compute logZ from the final state
        # (N, S)
        final = a + self.zero_num_states
        # (N)
        logZ = tf.reduce_logsumexp(final, axis=1)
        
        return logZ, Ms_grad
    
    @tf.function(jit_compile=True)
    def bwd_scores(self, Ms):
        """
        BWD for normalisation.
        
        Args:
            Ms ((max_time, batch_size, num_states, nz)): Reshaped scores, states with their transitions.
        
        Returns:
            betas ((max_time + 1, batch_size, num_states)): Backward scores.
        """
        # Prepare TensorArray for betas
        betas_ta = tf.TensorArray(dtype=self.dtype, size=self.max_time+1, element_shape=[self.batch_size, self.num_states])

        # beta at time T; All zeros.
        betas_ta = betas_ta.write(self.max_time, self.zero_num_states)
        # (N, S)
        a = self.zero_num_states

        # Reverse-time loop
        for t in range(self.max_time-1, -1, -1):
            # Flatten transitions for this t: (N, S*NZ)
            Ms_flat = tf.reshape(Ms[t], (self.batch_size, self.out_size))
            # Gather previous a values: from (N, S) to (N, S, NZ)
            prev_vals = tf.gather(a, self.prev_state_bcast, axis=1, batch_dims=1)
            # Gather transition scores: from (N, S*NZ) to (N, S, NZ)
            trans_vals = tf.gather(Ms_flat, self.trans_idx_bcast, axis=1, batch_dims=1)
            # (N, S, NZ)
            sum_vals = prev_vals + trans_vals
            # (N, S)
            a = tf.reduce_logsumexp(sum_vals, axis=2)
            # Write betas[t]
            betas_ta = betas_ta.write(t, a)

        # Stack to (T+1, N, S)
        betas = betas_ta.stack()
        return betas
    
    @tf.custom_gradient
    def _LogZ(self, Ms):
        """
        FWD BWD for normalisation.
        
        Args:
            Ms ((max_time, batch_size, num_states, nz)): Reshaped scores, states with their transitions.
        
        Returns:
            logZ ((batch_size)): Normalisation logZ.
            Gradients
        """
        # logZ (N), Ms_grad: (T, N, S, NZ)
        logZ, Ms_grad = self.logZ_fwd(Ms)
        def grad_fn(dlogZ):
            # BWD computation
            # betas: (T+1, N, S)
            betas = self.bwd_scores(Ms)
            
            # Multiply Ms_grad by betas[1:,:,None] broadcasted over NZ
            # (T, N, S, NZ)
            grad_part = Ms_grad + betas[1:, :, :, None]
            # (T, N, S, NZ) -> (T, N, S*NZ)
            reshaped = tf.reshape(grad_part, [self.max_time, self.batch_size, self.out_size])
            # Softmax
            summed = tf.nn.softmax(reshaped, axis=2)
            # (T, N, S*NZ) -> (T, N, S, NZ)
            Ms_grad_back = tf.reshape(summed, [self.max_time, self.batch_size, self.num_states, self.nz])
            
            # Final gradient w.r.t. Ms: dlogZ expanded and multiplied
            dlogZ_exp = tf.reshape(dlogZ, [1, self.batch_size, 1, 1])
            grad_Ms = dlogZ_exp * Ms_grad_back
            
            # Return gradients for all inputs; Ms
            return grad_Ms
        
        return logZ, grad_fn
    
    @tf.function(jit_compile=True)
    def normalise(self, scores):
        """
        Normalise the scores.
        
        Args:
            scores ((max_time, batch_size, features)): Output of the LinearCRFEncoder.
        
        Returns:
            scores_normalised ((max_time, batch_size, features)): The normalised scores.
        """
        # (T, N, C) -> (T, N, S, NZ)
        Ms = tf.reshape(scores, [self.max_time, self.batch_size, self.num_states, self.nz])
        logZ = self._LogZ(Ms)[:, None]
        return (scores - logZ / self.max_time)
    
    @tf.function(jit_compile=True)
    def prepare_ctc_scores(self, logits, labels):
        """
        Args:
            logits: ((max_time, batch_size, out_dim)): Normalised scores.
            labels: ((batch_size, padded_label_length)): Training labels.
        
        Returns:
            stay_scores ((max_time, batch_size, L)): From the indices that are multiples of 5; gathered over the time dimension.
            move_scores ((max_time, batch_size, L - 1)): Moves from one state using one of the bases A, C, G, T.
            with L = padded_label_length - (state_len - 1)
        """
        # Labels encoded as 1 to 4 -> 0 to 3
        labels_clipped = tf.clip_by_value(labels - 1, 0, self.size_alphabet)
        # Sliding window (N, L, state_len)
        frames = tf.signal.frame(
            labels_clipped,
            frame_length=self.state_len,
            frame_step=1,
            axis=1
        )
        # k-mer encoding; stay indices are multiples of 5
        stay_idx = tf.reduce_sum(frames * self.powers, axis=-1)
        stay_idx = stay_idx * self.size_alphabet
        # move indices are stay indices plus the value of the next base (A = 1, C = 2, G = 3, T = 4)
        move_idx = stay_idx[:, 1:] + labels_clipped[:, :self.n_crf - 1] + 1  # → (B, Y)
        
        # Give them a leading time dim of 1, then tile to (T, B, L/L-1)
        stay_idx_tb = tf.tile(tf.expand_dims(stay_idx, 0), [self.max_time, 1, 1])
        move_idx_tb = tf.tile(tf.expand_dims(move_idx, 0), [self.max_time, 1, 1])
        
        # Gather values from the scores using the extracted indices: (T, N, L)
        stay_scores = tf.gather(
            logits,
            stay_idx_tb,
            axis=2,
            batch_dims=2
        )
        # (T, N, L - 1)
        move_scores = tf.gather(
            logits,
            move_idx_tb,
            axis=2,
            batch_dims=2
        )
        return stay_scores, move_scores
    
    @tf.function(jit_compile=True)
    def _simple_lattice_fwd_bwd(self, beta_T, stay_scores, move_scores):
        """
        Forward and backward functions.
        
        Args:
            beta_T ((batch_size, L)): Valid end states.
            stay_scores ((max_time, batch_size, max_length)): Stay scores.
            move_socres ((max_time, batch_size, max_length - 1)): Move scores.
        
        Returns:
            alpha_ta ((max_time + 1, batch_size, max_length)): Forward values.
            beta_stay_ta ((max_time, batch_size, max_length)): Contributions from staying.
            beta_move_ta ((max_time, batch_size, max_length)): Contributions from moving.
        """
        # Allocate alpha
        alpha_ta = tf.TensorArray(dtype=self.dtype, size=self.max_time+1, clear_after_read=False, element_shape=[self.batch_size, self.n_crf])
        # a0: The first state is valid (0), all other states are invalid (-inf)
        # initial alpha[0]
        alpha_ta = alpha_ta.write(0, self.a0)
        # beta arrays
        beta_stay_ta = tf.TensorArray(dtype=self.dtype, size=self.max_time, element_shape=[self.batch_size, self.n_crf])
        beta_move_ta = tf.TensorArray(dtype=self.dtype, size=self.max_time, element_shape=[self.batch_size, self.n_crf])
        # Forward
        a_prev = alpha_ta.read(0)
        for t in range(self.max_time):
            # Contribution from staying in the state
            # (N, L)
            stay = a_prev + stay_scores[t]
            # batch_one_epsilon (N, 1): All -inf
            # (N, L)
            shifted = tf.concat([self.batch_one_epsilon, a_prev[:, :-1]], axis=1)
            # Contributions from moving into the state
            # (N, L)
            move = shifted + tf.pad(move_scores[t], [[0, 0], [1, 0]], constant_values=self.epsilon)
            # Combine contributions
            # (N, L)
            a_next = tf.reduce_logsumexp(tf.stack([stay, move], axis=2), axis=2)
            alpha_ta = alpha_ta.write(t + 1, a_next)
            a_prev = a_next
        
        # Backward
        b = beta_T
        for t in range(self.max_time-1, -1, -1):
            # Stay branch
            beta_stay_t = b + stay_scores[t]
            beta_stay_ta = beta_stay_ta.write(t, beta_stay_t)
            
            # Move branch
            # Shift b left: shifted_b[l] = b[l+1]
            shifted_b = tf.concat([b[:,1:], self.batch_one_epsilon], axis=1)
            # Pad move_scores on the right so padded_move[l] = move_scores[t, l]
            padded_move = tf.concat([move_scores[t], self.batch_one_epsilon], axis=1)
            beta_move_t = shifted_b + padded_move
            beta_move_ta = beta_move_ta.write(t, beta_move_t)
            
            # Merge for next b
            b = tf.reduce_logsumexp(
                tf.stack([beta_stay_t, beta_move_t], axis=-1),
                axis=-1
            )
        return alpha_ta.stack(), beta_stay_ta.stack(), beta_move_ta.stack()
    
    @tf.custom_gradient
    def LogZ(self, stay_scores, move_scores, target_lengths):
        """
        Calculate logZ.
        
        Args:
            stay_scores ((max_time, batch_size, max_length)): Stay scores gathered from the normalised scores.
            move_scores ((max_time, batch_size, max_length - 1)): Move scores gathered from the normalised scores.
            target_lengths ((batch_size)): Lengths of the encoded targets.
        
        Returns:
            logZ ((batch_size)): Main logZ.
            Gradients
        """
        # (N, 2)
        idx = tf.stack([self.batch_range, target_lengths-1], axis=1)
        # beta_T in log space. Only the position at the target length is 0, the rest is -inf: (N, L)
        beta_T = tf.tensor_scatter_nd_update(self.beta_T, idx, self.batch_zeros)
        # Run combined fwd bwd
        # (T + 1, N, L); (T, N, L); (T, N, L)
        alpha_stack, beta_stay, beta_move = self._simple_lattice_fwd_bwd(
            beta_T, stay_scores, move_scores
        )
        # Compute g for backward
        # g[t,n,0,l] = alpha[t,n,l] * beta_stay[t,n,l]
        # g[t,n,1,l] = alpha[t,n,l] * beta_move[t,n,l]
        # (T, N, 2, L)
        stacked = tf.concat([
          tf.expand_dims(alpha_stack[:-1] + beta_stay, 2),  # shape [T,N,1,L]
          tf.expand_dims(alpha_stack[:-1] + beta_move, 2)   # shape [T,N,1,L]
        ], axis=2) 
        
        # (T, N, 2, L) -> (T, N, 2*L)
        flat = tf.reshape(stacked, [self.max_time, self.batch_size, 2*self.n_crf])
        # Softmax over the transition dimension
        weights = tf.nn.softmax(flat, axis=2)
        # (T, N, 2*L) -> (T, N, 2, L)
        g = tf.reshape(weights, [self.max_time, self.batch_size, 2, self.n_crf])
        # Compute logZ
        final = alpha_stack[-1] + beta_T
        logZ = tf.reduce_logsumexp(final, axis=1)
        # Define backward
        def grad_fn(dlogZ):
            # (T, N, 2, L)
            g_scaled = g * dlogZ[None, :, None, None]
            grad_stay = g_scaled[:, :, 0, :]
            grad_move = g_scaled[:, :, 1, :-1]
            return grad_stay, grad_move, None
        return logZ, grad_fn
    
    ######## VITERBI SECTION ###########
    
    @tf.function
    def logZ_fwd_cu_max(self, Ms):
        """
        Pure TensorFlow implementation of the forward pass for logZ_fwd.
        
        Args:
            Ms: Tensor of shape [T, N, C, NZ]
        """
        
        # Prepare TensorArray to collect s at each timestep
        ta_grad = tf.TensorArray(dtype=self.dtype, size=self.max_time, element_shape=[self.batch_size, self.num_states, self.nz])
        
        # Initial semiring state a: [N, C]
        a = self.zero_num_states
        
        # Python for-loop over timesteps
        for t in range(self.max_time):
            m_t = Ms[t]                          # [N, C, NZ]
            prev_vals = tf.gather(a, self.idx, axis=1)  # [N, C, NZ]
            s = prev_vals + m_t            # [N, C, NZ]
            # Update state for next iteration
            a = tf.reduce_max(s, axis=2)                  # [N, C]
            # Write to TensorArray
            ta_grad = ta_grad.write(t, s)
        
        # Stack collected gradients: [T, N, C, NZ]
        Ms_grad = ta_grad.stack()
        
        # Compute logZ: multiply final state a with vT and semiring sum over C
        final = a + self.zero_num_states          # [N, C]
        logZ = tf.reduce_max(final, axis=1)   # [N]
        
        return logZ, Ms_grad
    
    @tf.function
    def bwd_scores_cu_max(self, Ms):
        # Prepare TensorArray for betas
        betas_ta = tf.TensorArray(dtype=self.dtype, size=self.max_time+1, element_shape=[self.batch_size, self.num_states])

        # beta at time T
        betas_ta = betas_ta.write(self.max_time, self.zero_num_states)
        a = self.zero_num_states  # [N, C]

        # Reverse-time loop
        for t in range(self.max_time-1, -1, -1):
            # Flatten transitions for this t: [N, C*NZ]
            Ms_flat = tf.reshape(Ms[t], (self.batch_size, self.out_size))

            # Gather previous a values: from [N, C] to [N, C, NZ]
            prev_vals = tf.gather(a, self.prev_state_bcast, axis=1, batch_dims=1)  # [N, C, NZ]

            # Gather transition scores: from [N, C*NZ] to [N, C, NZ]
            trans_vals = tf.gather(Ms_flat, self.trans_idx_bcast, axis=1, batch_dims=1)  # [N, C, NZ]

            # Compute log-sum-exp: prev_vals + trans_vals over NZ
            sum_vals = prev_vals + trans_vals
            a = tf.reduce_max(sum_vals, axis=2)  # [N, C]

            # Write betas[t]
            betas_ta = betas_ta.write(t, a)

        # Stack to [T+1, N, C]
        betas = betas_ta.stack()
        return betas
    
    @tf.function(jit_compile=True)
    def max_grad(self, x, axis=0):
        """
        Return a one-hot mask of the max indices of `x` along `axis`,
        just like PyTorch’s scatter_ + argmax trick.
    
        Args:
          x: Tensor of any shape.
          axis: integer axis to reduce over.
        Returns:
          A tensor of the same shape as `x`, with 1.0 where x is maximal
          along `axis` and 0.0 elsewhere.
        """
        # 1) find index of the max along the axis
        idx = tf.argmax(x, axis=axis, output_type=tf.int32)       # shape = x.shape without `axis`
        # 2) depth = size along that axis
        depth = tf.shape(x)[axis]
        # 3) build a one-hot tensor at those indices, aligned on the same axis
        return tf.one_hot(idx, depth, axis=axis, dtype=x.dtype)
    
    @tf.custom_gradient
    def _LogZ_max(self, Ms):
        # Forward computation via custom CUDA op
        # logZ: [N, C], Ms_grad: [T, N, C, NZ]
        logZ, Ms_grad = self.logZ_fwd_cu_max(Ms)
        def grad_fn(dlogZ):
            # Compute betas for backward via custom CUDA op
            # betas: [T+1, N, C]
            betas = self.bwd_scores_cu_max(Ms)
            
            # Multiply Ms_grad by betas[1:,:,:] broadcasted over NZ
            # S.mul: elementwise semiring multiplication
            grad_part = Ms_grad + betas[1:, :, :, None]

            # Sum over NZ dimension via semiring dsum
            reshaped = tf.reshape(grad_part, [self.max_time, self.batch_size, self.out_size])
            summed = self.max_grad(reshaped, axis=2)
            Ms_grad_back = tf.reshape(summed, [self.max_time, self.batch_size, self.num_states, self.nz])
            
            # Final gradient w.r.t. Ms: dlogZ expanded and multiplied
            dlogZ_exp = tf.reshape(dlogZ, [1, self.batch_size, 1, 1])
            grad_Ms = dlogZ_exp * Ms_grad_back
            
            # Return gradients for all inputs; None for idx, v0, vT, S, K
            return grad_Ms
        
        return logZ, grad_fn
    
    @tf.function(jit_compile=True)
    def logZ_max(self, scores):
        scores = tf.reshape(scores, [self.max_time, self.batch_size, self.num_states, self.nz])
        return self._LogZ_max(scores)
    
    @tf.function(jit_compile=True)
    def logZ(self, scores):
        scores = tf.reshape(scores, [self.max_time, self.batch_size, self.num_states, self.nz])
        return self._LogZ(scores)
    
    @tf.function(jit_compile=True)
    def posteriors(self, scores, type_="log"):
        """
        Compute the CTC‐style posteriors (the expected counts)
        by differentiating the log‐partition function wrt scores.
        Inputs:
          scores: float32 Tensor, shape [T, B, C]  (pre‐normalized logits)
        Returns:
          same‐shape float32 Tensor of posteriors.
        """
        with tf.GradientTape() as tape:
            tape.watch(scores)
            # logZ returns shape [B], one log‐partition per sequence in batch
            if type_ == "log":
                lZ = self.logZ(scores)           # calls your custom‐grad logZ_cu_sparse
            elif type_ == "max":
                lZ = self.logZ_max(scores)
            loss = tf.reduce_sum(lZ)         # scalar
        # ∂ loss / ∂ scores  is exactly the posterior for each arc
        post = tape.gradient(loss, scores)
        return post
    
    @tf.function(jit_compile=True)
    def viterbi(self, scores):
        """
        scores: [T, B, C]  (already pre‐softmax)  
        Returns: [T, B]  integer path labels (0 means “stay”, >0 means base index)
        """
        # 1) Take log, and grab the “MaxZ”‐based posterior (one‐hot on best path)
        log_scores = tf.math.log(scores)
        traceback = self.posteriors(log_scores, "max")  # [T, B, C]
        # 2) For each (t,b) pick the transition‐index with highest “gradient”
        a_traceback = tf.math.argmax(traceback, axis=2, output_type=tf.int32)  # [T, B]
        # 3) Decode “move vs stay” and the actual base index
        #    remainder mod |alphabet| tells you if it’s a “move” (≠0)
        moves = tf.not_equal(tf.math.mod(a_traceback,
                                             tf.constant(len(self.alphabet), tf.int32)),
                                 0)                                                       # [T, B]
        #    integer‐divide by |alphabet| then mod n_base gives the k‐mer “state”,
        #    +1 to shift into [1..n_base] range
        paths = 1 + tf.math.mod(
                tf.math.floordiv(a_traceback,
                tf.constant(len(self.alphabet), tf.int32)),
                tf.constant(self.n_base, tf.int32)
                )                                                        # [T, B]

        # 4) Where there was no move, emit 0; else emit the base index
        return tf.where(moves,
                        paths,
                        tf.zeros_like(paths, dtype=paths.dtype))            # [T, B]
    
    @tf.function
    def path_to_str(self, path):
        """
        path: [L] int32 Tensor of states, where 0 = blank, 1..4 = bases
        returns: scalar tf.string, e.g. "ACGTAG"
        """
        # 1) mask out the zeros (blanks)
        mask = tf.not_equal(path, 0)               # [L] boolean
        filtered = tf.boolean_mask(path, mask)     # [K] where K ≤ L

        # 2) gather the characters
        chars = tf.gather(self._chars, filtered)   # [K] tf.string

        # 3) join them into one string
        return tf.strings.reduce_join(chars, axis=0)  # scalar tf.string
    
    def decode(self, scores):
        scores = tf.transpose(scores, perm=(1, 0, 2))
        scores = self.posteriors(scores, "log") + 1e-8
        tracebacks = self.viterbi(scores)
        tracebacks = tf.transpose(tracebacks, perm=[1, 0])
        decoded = [(self.path_to_str(x)).numpy().decode("utf-8") for x in tracebacks]
        decoded_cut = [seq[self.n_pre_context_bases:len(seq)-self.n_post_context_bases] for seq in decoded]
        return decoded_cut
    
    ######## VITERBI SECTION ###########
    
    @tf.function(jit_compile=True)
    def __call__(self, scores, targets, target_lengths, normalise_scores=True):
        """
        Calculate the total loss for a batch of inputs.
        
        Args:
            scores ((batch_size, max_time, features)): Output of the LinearCRFEncoder.
            targets ((batch_size, padded_label_length)): The training labels. Padded with zeros.
            target_length ((batch_size)): The actual label lengths without the padding; a constant is substracted.
        
        Returns:
            loss ((batch_size)): The total loss for every entry in the batch.
        """
        # (N, T, C) -> (T, N, C)
        scores = tf.transpose(scores, perm=(1, 0, 2))
        # Scores are always normalised
        if normalise_scores:
            scores = self.normalise(scores)
        # Gather the stay and move scores for the training labels
        # (T, N, padded_label_length - (state_len - 1) (== L)), (T, N, L - 1)
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        # LogZ values: (N)
        logz = self.LogZ(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        # Normalise by the label length
        loss = - (logz / tf.cast(target_lengths, self.dtype))
        return loss
