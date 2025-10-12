import datetime
import os

import keras


def early_stopping_callback(patience=5):
    return keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, min_delta=1e-4,
                                         restore_best_weights=True)


def checkpoint_callback(filepath: str, save_best_only=True):
    return keras.callbacks.ModelCheckpoint(filepath, monitor="val_loss", save_best_only=save_best_only)


def lr_scheduler_callback(factor=0.5, patience=3, min_lr=1e-6, verbose=1):
    return keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=factor, patience=patience, verbose=verbose,
                                             min_lr=min_lr)


def tensorboard_callback(log_dir: str):
    # Add timestamp to avoid overwriting old logs
    run_logdir = os.path.join(
        log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return keras.callbacks.TensorBoard(
        log_dir=run_logdir,
        histogram_freq=1,  # log histogram of weights
        write_graph=True,  # log the computation graph
        write_images=True,  # log model weights as images
        update_freq="epoch",  # log metrics per epoch
        profile_batch=0  # disable profiling
    )
