"""Serializable TensorFlow gradient-reversal layer used by the GRL model."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="AdversarialCV")
class GradientReversal(tf.keras.layers.Layer):
    """Identity in the forward pass; multiply incoming gradients by ``-scale``."""

    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def call(self, inputs):
        scale = tf.cast(self.scale, inputs.dtype)

        @tf.custom_gradient
        def reverse_gradient(value):
            def gradient(upstream):
                return -scale * upstream

            return tf.identity(value), gradient

        return reverse_gradient(inputs)

    def get_config(self):
        return {**super().get_config(), "scale": self.scale}
