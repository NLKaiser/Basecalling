import tensorflow as tf

class CTC:
    def __init__(self, logits_time_major=True, blank_index=0):
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index
        
        # Decoding
        self.index_to_char = {1: "A", 2: "C", 3: "G", 4: "T"}
    
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
        labels = tf.sparse.from_dense(labels)
        
        if self.logits_time_major:
            logits = tf.transpose(logits, perm=[1, 0, 2])
        
        return tf.nn.ctc_loss(labels=labels, logits=logits, label_length=label_length, logit_length=input_length, logits_time_major=self.logits_time_major, blank_index=self.blank_index)
