import keras


def build_resnet50(input_shape=(224, 224, 3), num_classes=10, label_smoothing=0.05):
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg"
    )
    base_model.trainable = False

    model = keras.models.Sequential([
        base_model,

        keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-3)
        ),

        keras.layers.BatchNormalization(),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"]
    )
    return model
