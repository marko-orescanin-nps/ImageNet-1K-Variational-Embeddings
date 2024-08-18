import os
import tensorflow as tf
import math


def make_callbacks(hparams):
    checkpoints = []
    if "checkpoint" in hparams.callback_list:
        checkpoints.append(_make_model_checkpoint_cb(hparams))
    if "csv_log" in hparams.callback_list:
        checkpoints.append(_make_csvlog_cb(hparams))
    if "early_stopping" in hparams.callback_list:
        checkpoints.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        )
    return checkpoints


def _make_model_checkpoint_cb(hparams):
    if hparams.regression:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(hparams.model_dir, "checkpoint{epoch:02d}-{val_loss:.2f}.h5"),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            period=1,
        )
    else:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(hparams.model_dir, "checkpoint{epoch:02d}-{val_loss:.2f}.h5"),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            period=1,
        )
    return checkpoint


def _make_csvlog_cb(hparams):
    print("entered_csv_log")
    csv_log = tf.keras.callbacks.CSVLogger(
        os.path.join(hparams.model_dir, "log.csv"), append=True, separator=";"
    )
    return csv_log
