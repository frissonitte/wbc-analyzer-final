import tensorflow as tf

keras = tf.keras
layers = keras.layers


@tf.keras.utils.register_keras_serializable(package="WBC")
class MedSwish(layers.Layer):
    """Swish-like activation with trainable slope and enhancement terms."""

    def __init__(self, alpha=0.1, beta=1.0, trainable_params=True, **kwargs):
        super(MedSwish, self).__init__(**kwargs)
        self.initial_alpha = alpha
        self.initial_beta = beta
        self.trainable_params = trainable_params

    def build(self, input_shape):
        if self.trainable_params:
            self.alpha = self.add_weight(
                name="alpha",
                shape=(1,),
                initializer=tf.keras.initializers.Constant(self.initial_alpha),
                trainable=True,
            )
            self.beta = self.add_weight(
                name="beta",
                shape=(1,),
                initializer=tf.keras.initializers.Constant(self.initial_beta),
                trainable=True,
            )
        else:
            self.alpha = tf.constant(self.initial_alpha, dtype=tf.float32)
            self.beta = tf.constant(self.initial_beta, dtype=tf.float32)
        super(MedSwish, self).build(input_shape)

    def call(self, x):
        x = tf.cast(x, dtype=self.compute_dtype)
        # Base swish branch + bounded enhancement branch.
        swish_part = x * tf.nn.sigmoid(self.beta * x)
        enhancement = 1.0 + self.alpha * tf.nn.tanh(x)
        return swish_part * enhancement

    def get_config(self):
        config = super(MedSwish, self).get_config()
        config.update(
            {
                "alpha": float(self.initial_alpha),
                "beta": float(self.initial_beta),
                "trainable_params": self.trainable_params,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="WBC")
class WBCAttentionBlock(layers.Layer):
    """CBAM-style attention block with channel then spatial refinement."""

    def __init__(self, reduction_ratio=16, **kwargs):
        super(WBCAttentionBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]

        self.channel_dense1 = layers.Dense(
            channels // self.reduction_ratio,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=True,
        )
        self.channel_dense2 = layers.Dense(
            channels, kernel_initializer="he_normal", use_bias=True
        )

        self.spatial_conv = layers.Conv2D(
            1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=True,
        )

        super(WBCAttentionBlock, self).build(input_shape)

    def channel_attention(self, x):
        # Pool across spatial axes to summarize each channel.
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)

        avg_pool = tf.reshape(avg_pool, [-1, avg_pool.shape[-1]])
        max_pool = tf.reshape(max_pool, [-1, max_pool.shape[-1]])

        avg_out = self.channel_dense2(self.channel_dense1(avg_pool))
        max_out = self.channel_dense2(self.channel_dense1(max_pool))

        channel_att = tf.nn.sigmoid(avg_out + max_out)
        channel_att = tf.reshape(channel_att, [-1, 1, 1, channel_att.shape[-1]])

        channel_att = tf.cast(channel_att, x.dtype)
        return x * channel_att

    def spatial_attention(self, x):
        # Pool across channels to highlight informative spatial locations.
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        spatial_att = self.spatial_conv(concat)

        spatial_att = tf.cast(spatial_att, x.dtype)
        return x * spatial_att

    def call(self, x):
        # Sequential channel->spatial attention keeps behavior deterministic.
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super(WBCAttentionBlock, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config
