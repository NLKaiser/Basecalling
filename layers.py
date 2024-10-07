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
