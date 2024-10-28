import tensorflow as tf

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
        
        self.linear_warmup = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=warmup_initial,
            end_learning_rate=warmup_end,
            decay_steps=warmup_steps,
            power=1.0
        )
        
        # Define the CosineDecayRestarts schedule
        self.cosine_decay_restarts = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha
        )
    
    def __call__(self, step):
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.linear_warmup(step),  # Warm-up phase
            lambda: self.cosine_decay_restarts(step - self.warmup_steps)  # Cosine decay with restarts
        )