import tensorflow as tf
from tensorflow.keras.losses import Loss


class LqLoss(Loss):
    """
    Lq loss is the generalized cross entropy loss, approximating cross entropy and mean absolute error. This is the
    multilabel version of the original multi-class version.
    Add papers.

    q is the value to tune noise robustness by getting closer to cross entropy or MAE (0, 1]
    """

    def __init__(self, q=0.5, name="lq_loss"):
        super().__init__(name=name)
        self.q = tf.constant(q, tf.float32)  #here or in call?
        self.log_offset = tf.constant(1e-8, tf.float32)  #here or in call?

    def call(self, y_true, y_pred):
        # clipping to avoid instability issues
        term1 = tf.clip_by_value(
          y_pred + self.log_offset,
          clip_value_min=0.,
          clip_value_max=1.)
        term2 = tf.clip_by_value(
          (1 - y_pred),
          clip_value_min=self.log_offset,
          clip_value_max=1.)

        lq_loss = tf.cast(y_true,tf.float32) * ((1 - tf.pow(term1, self.q)) / self.q) + (1 - tf.cast(y_true,tf.float32)) * (1 - tf.pow(term2, self.q)) / self.q
        return lq_loss





# # Usage without high level API:
# loss_obj = LqLoss(q=0.5)


# def train_step(input, y_true):

#     with tf.GradientTape() as tape:
#         # training=True is only needed if there are layers with different
#         # behavior during training versus inference (e.g. Dropout).
#         y_pred = model(input, training=True)
#         pred_loss = loss_obj(y_true, y_pred)
#         reg_losses = model.losses
#         loss = tf.math.add_n([pred_loss] + reg_losses)

#     # compute gradients
#     gradients = tape.gradient(loss, model.trainable_variables)
#     opt.apply_gradients(zip(gradients, model.trainable_variables))


# # Usage with high level API:
# model.compile(optimizer=keras.optimizers.Adam(), loss=LqLoss(q=0.5))

