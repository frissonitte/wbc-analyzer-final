import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable(package="WBC")
class WBCFocalLoss(keras.losses.Loss):
    """
    WBC focal loss.

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(
        self,
        gamma=2.0,
        alpha=0.25,
        class_weights=None,
        label_smoothing=0.1,
        name="wbc_focal_loss",
        **kwargs
    ):
        super(WBCFocalLoss, self).__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

        if class_weights is None:
            self.class_weights = tf.constant(
                [2.0, 1.2, 1.0, 1.3, 1.0], dtype=tf.float32
            )
        else:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        """Compute class-weighted focal loss with optional label smoothing."""
        num_classes = tf.shape(y_true)[-1]
        y_true_smooth = y_true * (
            1.0 - self.label_smoothing
        ) + self.label_smoothing / tf.cast(num_classes, tf.float32)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        cross_entropy = -y_true_smooth * tf.math.log(y_pred)

        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        weights = tf.reduce_sum(y_true * self.class_weights, axis=-1, keepdims=True)

        focal_loss = self.alpha * focal_weight * cross_entropy * weights

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        """Return serializable layer configuration."""
        config = super(WBCFocalLoss, self).get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config
