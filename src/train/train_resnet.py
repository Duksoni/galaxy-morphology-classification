import keras
import tensorflow as tf

from src.train.callbacks import early_stopping_callback, checkpoint_callback, tensorboard_callback

tb_callback = tensorboard_callback("logs/resnet50")


def cosine_scheduler(initial_lr, steps):
    return keras.optimizers.schedules.CosineDecay(initial_lr, steps)


def train_resnet50(
        model: keras.models.Sequential,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weights: dict | None,
        n_epochs: int,
        file_path: str
) -> dict:
    frozen_epochs = min(n_epochs, 10)
    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=frozen_epochs,
        class_weight=class_weights,
        callbacks=[early_stopping_callback(), checkpoint_callback(file_path), tb_callback]
    )

    if n_epochs <= frozen_epochs:
        return history_frozen.history

    base_model = model.layers[0]

    for layer in base_model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    steps_per_epoch = len(train_ds)
    total_finetune_epochs = n_epochs - frozen_epochs
    total_steps_finetune = steps_per_epoch * total_finetune_epochs
    scheduler = cosine_scheduler(initial_lr=1e-5, steps=total_steps_finetune)

    # === Stage 1: unfreeze top 50 layers ===
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=scheduler, momentum=0.9),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    stage1_epochs = 20
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=frozen_epochs,  # continue counting
        epochs=frozen_epochs + stage1_epochs,
        class_weight=class_weights,
        callbacks=[early_stopping_callback(), checkpoint_callback(file_path), tb_callback]
    )

    # === Stage 2: unfreeze the rest ===
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=scheduler, momentum=0.9),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=frozen_epochs + stage1_epochs,  # continue counting
        epochs=n_epochs,
        class_weight=class_weights,
        callbacks=[early_stopping_callback(), checkpoint_callback(file_path), tb_callback]
    )

    # Merge histories
    return {
        "accuracy": history_frozen.history["accuracy"]
                    + history_stage1.history["accuracy"]
                    + history_stage2.history["accuracy"],
        "val_accuracy": history_frozen.history["val_accuracy"]
                        + history_stage1.history["val_accuracy"]
                        + history_stage2.history["val_accuracy"],
        "loss": history_frozen.history["loss"]
                + history_stage1.history["loss"]
                + history_stage2.history["loss"],
        "val_loss": history_frozen.history["val_loss"]
                    + history_stage1.history["val_loss"]
                    + history_stage2.history["val_loss"]
    }
