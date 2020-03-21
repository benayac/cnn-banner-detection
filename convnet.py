import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

#hyperparameter
EPOCHS = 10
BATCH_SIZE = 10
SPLIT_RATIO = 0.8

#initialize directory variable and data count variable
train_detected_dir = os.path.join('train','detected')
train_not_detected_dir = os.path.join('train', 'not_detected')

val_detected_dir = os.path.join('validation','detected')
val_not_detected_dir = os.path.join('validation', 'not_detected')

detected_count = len(os.listdir(train_detected_dir))
not_detected_count = len(os.listdir(train_not_detected_dir))
try:
    os.makedirs("validation/not_detected/")
    os.makedirs("validation/detected/")
except Exception:
    pass
#normalize data between detected image and not detected
detected_file = os.listdir(train_detected_dir)
random.shuffle(detected_file)

for data in range(detected_count-not_detected_count):
    os.remove(f"{train_detected_dir}/{detected_file[data]}")

detected_count = len(os.listdir(train_detected_dir))

#separate train and val data
detected_file = os.listdir(train_detected_dir)
not_detected_file = os.listdir(train_not_detected_dir)
random.shuffle(detected_file)
random.shuffle(not_detected_file)

train_detected_size = int(len(detected_file) * SPLIT_RATIO)
train_not_detected_size = int(len(not_detected_file) * (1-SPLIT_RATIO))

for data in range(len(detected_file[:train_detected_size])):
    shutil.move(f"{train_detected_dir}/{detected_file[data]}", f"{val_detected_dir}/{detected_file[data]}")

for data in range(len(not_detected_file[:train_not_detected_size])):
    shutil.move(f"{train_not_detected_dir}/{not_detected_file[data]}", f"{val_not_detected_dir}/{not_detected_file[data]}")

#create data generator
train_dir = 'train'
val_dir = 'validation'

train_datagen = ImageDataGenerator(
    rescale=1./255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=30,
    shear_range=.01,
    zoom_range=[0.9, 1.25],
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_datagen = ImageDataGenerator(rescale=1./255.0)

val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

#create the cnn model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//BATCH_SIZE
STEP_SIZE_VALID=val_generator.n//BATCH_SIZE

#start training
history = model.fit_generator(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=val_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=EPOCHS)

model.evaluate_generator(val_generator, steps=STEP_SIZE_VALID)

model.save('banner_detection_model')