import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_BREEDS = 120    # Stanford dataset = 120 dog breeds
NUM_AGE_GROUPS = 3  # DogAge dataset = 3 age categories (Young, Adult, Senior)

def multitask_model():
    #placeholder for input
    input = layers.Input(shape = (IMG_HEIGHT, IMG_WIDTH, 3), name = "input_image")

    #data augmentation (because of breed overfitting)
    x = layers.RandomFlip("horizontal")(input)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.15)(x)

    #rescaling input pixel values
    x = layers.Rescaling(1./255)(input)

    #first convolution
    x = layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)
    #normalization
    x = layers.BatchNormalization()(x)
    #extract dominant features
    x = layers.MaxPooling2D((2, 2))(x)

    #second convolution
    x = layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    #third convolution
    x = layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    #fourth convolution
    x = layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    #average each layer value
    features = layers.GlobalAveragePooling2D()(x)

    #---------Multitask Learning---------

    #breed classification branch
    #apply neurons
    breed_features = layers.Dense(224, activation = 'relu')(features)
    #drop neurons (reduce overfit)
    breed_features = layers.Dropout(0.5)(breed_features)
    #apply neurons, softmax assigns probability of breed type
    breed_output = layers.Dense(NUM_BREEDS, activation = 'softmax', name = 'breed_output')(breed_features)

    #age classification branch
    age_features = layers.Dense(224, activation = 'relu')(features)
    age_features = layers.Dropout(0.2)(age_features)
    age_output = layers.Dense(NUM_AGE_GROUPS, activation = 'softmax', name = 'age_output')(age_features)

    model = Model(inputs=input, outputs=[breed_output, age_output])
    return model

model = multitask_model()