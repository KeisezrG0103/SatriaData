import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from matplotlib import pyplot as plt

train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=r"E:\SatdatRapi\V2\Dataset_ - Copy",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    batch_size=32,
    image_size=(28, 28),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
    follow_links=False
)

# validation_ds = tf.keras.utils.image_dataset_from_directory(
#     directory=r".//Dataset_",
#     labels="inferred",
#     label_mode="categorical",
#     # class_names=None,
#     batch_size=32,
#     image_size=(28,28),
#     shuffle=True,
#     seed=42,
#     validation_split=0.2,
#     subset="validation",
#     interpolation="bilinear",
#     follow_links=False
# )

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = train_ds.take(200)

data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(28, 28, 3)),
    keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255)

# Create the model
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

num_classes = 36

#make 17 layers

model = keras.Sequential([
    # data_augmentation,
    keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(28, 28, 3)),
    keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes)
])

adadelta = tf.keras.optimizers.Adadelta(
    learning_rate=2, rho=0.95, epsilon=1e-07, name="Adadelta"
)
adam = tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam"
)

adamax = tf.keras.optimizers.Adamax(
    learning_rate=0.001, beta_1=0.08, beta_2=0.879, epsilon=1e-07, name="Adamax"
)

model.compile(optimizer=adadelta, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

epochs = 200

history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs)
# new_model = tf.keras.models.load_model('model_sleepy.h5')
# new_model.summary()
# #overfitting check
# from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=2)
# history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs, callbacks=[early_stopping])


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

avg_val_acc = np.average(val_acc)
print("Average validation accuracy: ", avg_val_acc)
avg_Accuracy = np.average(acc)
print("Average accuracy: ", avg_Accuracy)
avg_Val_lose = np.average(val_loss)
print("Average validation loss: ", avg_Val_lose)


model.save('model_sleepy.h5')

# model.save('model.sleepy')
