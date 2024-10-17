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

class ClipLayer(tf.keras.layers.Layer):
    def __init__(self, min_, max_):
        super(ClipLayer, self).__init__()
        self.min = min_
        self.max = max_
    
    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max)

class ConcatLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs)

    def call(self, layers):
        return tf.concat(layers, axis=-1)
