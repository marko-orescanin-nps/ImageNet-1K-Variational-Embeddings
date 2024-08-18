import tensorflow as tf


def get_optimizer(hparams):
    """
    Function to get optimizer string and return the corresponding TF optimizer
    :param opt_string: string optimizer
    :param lr: learning rate
    :param name: name for optimizer
    :return: TF optimizer
    """

    if hparams.optimizer == "SGD":
        return tf.keras.optimizers.SGD(
            learning_rate=hparams.base_learning_rate, momentum=0.0, nesterov=False
        )
    elif hparams.optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=hparams.base_learning_rate)
    else:
        raise ValueError("""optimizer not defined in "get_optimizer" function.""")
