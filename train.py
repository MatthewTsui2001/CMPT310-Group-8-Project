import os
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import multitask_model

#python version 3.13.14:
#pip install tensorflow
#pip install pandas
#pip install matplotlib

BATCH_SIZE = 64
IMG_HEIGHT = 128
IMG_WIDTH = 128
EPOCHS = 10
SEED = 10
TRAIN_SIZE = 0.8

DOG_BREED_DIR = "Stanford Dog Breeds Dataset"
NUM_BREEDS = 120
DOG_AGE_DIR = "DogAge Dataset"
NUM_AGE = 3

#prep train/test age dataframes while balancing image representations
#used by load_datasets()
def prep_age_df():
    age_list = ["Young", "Adult", "Senior"]
    age_index_list = {age: index for index, age in enumerate(age_list)} #Young: 0, Adult: 1, Senior: 2
    #print(f"\n[prep_age_df] {age_index_list}")
    age_folders = ["Expert_Train", "PetFinder_All"]
    consolidated_age_data = []

    for age_folder in age_folders:                      #Expert_Train, PetFinder_All
        for age_category in age_list:                   #Young, Adult, Senior
            age_index = age_index_list[age_category]    #0, 1, 2
            for age_img in glob.glob(f"{DOG_AGE_DIR}\\{age_folder}\\{age_category}\\*"):
                #({filepath, breed (unknown), age}, ...)
                consolidated_age_data.append({"filepath": age_img, "breed": -1, "age": age_index})

    age_df = pd.DataFrame(consolidated_age_data)

    #balance image representations (undersampling)
    smallest_class_size = age_df['age'].value_counts().min()
    print(f"\n[prep_age_df] Smallest class size: {smallest_class_size}")
    balanced_age_df = age_df.groupby('age').sample(n = smallest_class_size, random_state = SEED)
    #shuffle data
    balanced_age_df = balanced_age_df.sample(frac = 1, random_state = SEED).reset_index(drop = True)

    #split 80% train, 20% test
    split_index = int(len(balanced_age_df) * TRAIN_SIZE)
    train_age_df = balanced_age_df.iloc[:split_index]
    test_age_df = balanced_age_df.iloc[split_index:]
    print(f"\n[prep_age_df] Train set: {len(train_age_df)}, Test set: {len(test_age_df)}")
    return train_age_df, test_age_df

#load and form multitask data
def load_datasets():
    breed_list = sorted(os.listdir(f"{DOG_BREED_DIR}\\train"))
    breed_index_list = {breed: index for index, breed in enumerate(breed_list)}
    consolidated_train_breed_data = []
    consolidated_val_breed_data = []

    for breed_folder in breed_list:
        breed_index = breed_index_list[breed_folder]
        for breed_img in glob.glob(f"{DOG_BREED_DIR}\\train\\{breed_folder}\\*"):
            #({filepath, breed, age(unknown)}, ...)
            consolidated_train_breed_data.append({"filepath": breed_img, "breed": breed_index, "age": -1})
    
    train_breed_df = pd.DataFrame(consolidated_train_breed_data)

    for breed_folder in breed_list:
        breed_index = breed_index_list[breed_folder]
        for breed_image in glob.glob(f"{DOG_BREED_DIR}\\test\\{breed_folder}\\*"):
            consolidated_val_breed_data.append({"filepath": breed_img, "breed": breed_index, "age": -1})

    val_breed_df = pd.DataFrame(consolidated_val_breed_data)

    print(f"\n[load_datasets] Train set: {len(train_breed_df)}, Test set: {len(val_breed_df)}")

    train_age_df, val_age_df = prep_age_df()

    train_final = pd.concat([train_breed_df, train_age_df], ignore_index = True)
    train_final = train_final.sample(frac = 1, random_state = SEED).reset_index(drop = True)

    val_final = pd.concat([val_breed_df, val_age_df], ignore_index = True)
    val_final = val_final.sample(frac = 1, random_state = SEED).reset_index(drop = True)

    print(f"\n[load_datasets] final number of training entries: {len(train_final)}")
    print(f"\n[load_datasets] final number of testing entries: {len(val_final)}")

    return train_final, val_final

def pre_process_image(filepath, breed, age):
    #read
    image = tf.io.read_file(filepath)
    #split each pixel to rgb values
    image = tf.io.decode_jpeg(image, channels = 3)
    #resize for model
    image = tf.image.resize(image, [128, 128])
    #normalize each pixel value (0.0 to 1.0)
    image = tf.cast(image, tf.float32)

    return image, {"breed_output": breed, "age_output": age}

def tf_batching(df, is_training = True):
    filepaths = df['filepath'].values
    breeds = df['breed'].values
    ages = df['age'].values

    tf_dataset = tf.data.Dataset.from_tensor_slices((filepaths, breeds, ages))
    
    if is_training:
        tf_dataset = tf_dataset.shuffle(buffer_size = len(df), reshuffle_each_iteration = True)

    tf_dataset = tf_dataset.map(pre_process_image, num_parallel_calls = tf.data.AUTOTUNE)
    tf_dataset = tf_dataset.batch(BATCH_SIZE)
    tf_dataset = tf_dataset.prefetch(buffer_size = tf.data.AUTOTUNE)

    return tf_dataset

def make_masked_loss():
    def masked_loss(y_true, y_pred):
        #mask to filter out unknown (-1) values
        mask = tf.not_equal(y_true, -1)
        mask = tf.cast(mask, tf.float32)

        y_true_clean = tf.where(y_true == -1, tf.zeros_like(y_true), y_true)

        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_clean, y_pred)

        return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-1)

    return masked_loss

def breed_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)            #1.0 if valid label, 0.0 if -1
    predictions = tf.cast(tf.argmax(y_pred, axis = -1), tf.float32) 
    correct = tf.cast(tf.equal(tf.cast(y_true, tf.float32), predictions), tf.float32)

    return tf.reduce_sum(correct * mask) / (tf.reduce_sum(mask) + 1e-8)

def age_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)            #1.0 if valid label, 0.0 if -1
    predictions = tf.cast(tf.argmax(y_pred, axis = -1), tf.float32) 
    correct = tf.cast(tf.equal(tf.cast(y_true, tf.float32), predictions), tf.float32)

    return tf.reduce_sum(correct * mask) / (tf.reduce_sum(mask) + 1e-8)


#-------------RUN------------
print("\nLoading Dataframes...")
train_df, val_df = load_datasets()

print("Loading tf batching")
train_ds = tf_batching(train_df, is_training = True)
val_ds = tf_batching(val_df, is_training = False)

print("Initializing Model...")
model = multitask_model()

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = {
        "breed_output": make_masked_loss(),
        "age_output": make_masked_loss()
    },
    metrics = {
        "breed_output": breed_accuracy,
        "age_output": age_accuracy
    }
)

#model.summary()

print("\nTraining Start!")
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = EPOCHS,
    verbose = 2
)

model.save("data_augmentation_multitask_model.keras")
print("Training completed! Successfully saved as 'multitask_model'")

#--------GRAPH-------
def plot_training_results(history):
    """
    Plots the training and validation curves for both Breed and Age branches.
    Generates a 2-row layout: Top row for Losses, Bottom row for Accuracies.
    """
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    plt.figure(figsize=(14, 10))
    
    # ------------------ ROW 1: LOSSES ------------------
    # Subplot 1: Breed Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history.history['breed_output_loss'], label='Train Breed Loss', color='blue')
    plt.plot(epochs_range, history.history['val_breed_output_loss'], label='Val Breed Loss', color='darkorange', linestyle='--')
    plt.title('Breed Classification Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Age Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history.history['age_output_loss'], label='Train Age Loss', color='green')
    plt.plot(epochs_range, history.history['val_age_output_loss'], label='Val Age Loss', color='red', linestyle='--')
    plt.title('Age Classification Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # ------------------ ROW 2: ACCURACIES ------------------
    # Subplot 3: Breed Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, history.history['breed_output_breed_accuracy'], label='Train Breed Acc', color='blue')
    plt.plot(epochs_range, history.history['val_breed_output_breed_accuracy'], label='Val Breed Acc', color='darkorange', linestyle='--')
    plt.title('Breed Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Subplot 4: Age Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, history.history['age_output_age_accuracy'], label='Train Age Acc', color='green')
    plt.plot(epochs_range, history.history['val_age_output_age_accuracy'], label='Val Age Acc', color='red', linestyle='--')
    plt.title('Age Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and display/save the figures
    plt.tight_layout()
    plt.savefig('multitask_training_performance.png')
    plt.show()
    print("📊 Training graphs saved successfully as 'multitask_training_performance.png'")

plot_training_results(history)