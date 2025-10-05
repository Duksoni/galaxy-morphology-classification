import keras


def augment_pipeline(seed: int = 7):
    return keras.Sequential([
        keras.layers.RandomRotation(0.15, seed=seed + 1),
        keras.layers.RandomTranslation(0.08, 0.08, seed=seed + 2),
        keras.layers.RandomFlip("horizontal", seed=seed + 3),
    ])


def stronger_augment_pipeline(seed: int = 7):
    return keras.Sequential([
        keras.layers.RandomRotation(0.3, seed=seed + 1),
        keras.layers.RandomTranslation(0.1, 0.1, seed=seed + 2),
        keras.layers.RandomFlip("horizontal", seed=seed + 3),
    ])
