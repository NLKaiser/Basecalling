import tensorflow as tf

# CosineDecayWithRestarts together with a linear warmup phase
class WarmUpCosineDecayWithRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 warmup_initial,
                 warmup_end,
                 warmup_steps,
                 initial_learning_rate,
                 decay_steps,
                 alpha=0.0,
                 t_mul=2.0,
                 m_mul=1.0):
        super(WarmUpCosineDecayWithRestarts, self).__init__()
        
        self.warmup_steps = warmup_steps
        
		# Linear warmup
        self.linear_warmup = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=warmup_initial,
            end_learning_rate=warmup_end,
            decay_steps=warmup_steps,
            power=1.0
        )
        
        # CosineDecayRestarts schedule
        self.cosine_decay_restarts = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha
        )
        
        self.lr = warmup_initial
    
    def __call__(self, step):
        self.lr =  tf.cond(
            step < self.warmup_steps,
            lambda: self.linear_warmup(step),  # Warmup phase
            lambda: self.cosine_decay_restarts(step - self.warmup_steps)  # Cosine decay with restarts
        )
        return self.lr

# Learning rate reduction based on loss plateau with resets
class LRReductionScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 initial_lr,
                 patience,
                 reduce_difference,
                 factor,
                 wait_steps,
                 min_lr,
                 reset=True):
        super(LRReductionScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.patience = patience # Steps used for mean calculation
        self.reduce_difference = reduce_difference  # Threshold for mean difference in loss
        self.factor = factor  # LR reduction factor
        self.wait_steps = wait_steps # Steps to wait after LR reduction
        self.min_lr = min_lr
        self.reset = reset # Reset LR when min_lr is reached
        
        # Initialize variables
        self.lr = tf.Variable(initial_lr, dtype=tf.float32, trainable=False)
        self.losses = tf.Variable(tf.zeros((self.patience,), dtype=tf.float32), trainable=False)  # Initialize with zeros
        self.wait = tf.Variable(0, dtype=tf.int32, trainable=False)  # Patience counter
        self.current_loss_index = tf.Variable(0, dtype=tf.int32, trainable=False)  # To track the index for the next loss

    def __call__(self, step):
        return self.lr

    def update(self, loss):
        # Update the losses array with the new loss at the current index
        update_index = self.current_loss_index % self.patience
        self.losses[update_index].assign(loss)

        # Increment the index for the next loss update
        self.current_loss_index.assign_add(1)

        # Calculate the actual number of losses considered (up to patience)
        num_losses = tf.minimum(self.current_loss_index, self.patience)

        # Calculate the mean of the losses considered
        mean_loss = tf.reduce_mean(self.losses[:num_losses])

        # Calculate the difference between the mean loss and the most recent loss
        loss_difference = mean_loss - loss

        # Conditions to determine if the learning rate should be reduced
        should_reduce_lr = tf.logical_and(
            tf.less_equal(loss_difference, self.reduce_difference),
            tf.greater_equal(self.wait, self.wait_steps)
        )

        # Reduce learning rate or increment wait
        def reduce_lr():
            new_lr = tf.maximum(self.lr * self.factor, self.min_lr)
            self.lr.assign(new_lr)
            self.wait.assign(0)  # Reset the wait counter
            return new_lr

        def increment_wait():
            self.wait.assign_add(1)
            return self.lr

        # Update learning rate based on conditions
        tf.cond(should_reduce_lr, reduce_lr, increment_wait)
        
        # Reset the learning rate and state if necessary
        def reset_lr():
            self.lr.assign(self.initial_lr)
            self.wait.assign(0)
            self.losses.assign(tf.zeros((self.patience,), dtype=tf.float32))  # Reset losses to zeros
            self.current_loss_index.assign(0)  # Reset index to 0
            return self.lr

        # Check if we reached min_lr and if reset is enabled
        should_reset = tf.logical_and(
            tf.equal(self.lr, self.min_lr),
            self.reset
        )

        # Reset the learning rate if the condition is met
        tf.cond(should_reset, reset_lr, lambda: self.lr)

