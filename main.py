import tensorflow as tf
import keras
from tensorflow import keras
models = keras.models
layers = keras.layers
import matplotlib.pyplot as plt
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS=50
dataset= tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",

    shuffle=True,

    image_size=(IMAGE_SIZE,IMAGE_SIZE),

    batch_size= BATCH_SIZE,
)
class_names = dataset.class_names
# print(class_names)
# print(len((dataset)))

plt.figure(figsize=(10,10))
# plt.figure(figsize=(10,10))
# for image_batch, label_batch in dataset.take(1):
#     for i in range(12):
#         ax = plt.subplot(3,4,i+1)
#         plt.imshow(image_batch[i].numpy().astype("uint8"))
#         plt.axis("off")
#         plt.title(class_names[label_batch[i]])
#     plt.show()
# train_size=0.8
# print(len(dataset)*train_size)
# train_ds=dataset.take(54)
# print(len(train_ds))         # output---->54
# train_ds = dataset.skip(54)
# print(len(train_ds))     output------>14   (68-54)


# def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
#     assert (train_split + test_split + val_split) == 1
#
#     ds_size = len(ds)
#
#     if shuffle:
#         ds = ds.shuffle(shuffle_size, seed=12)
#
#     train_size = int(train_split * ds_size)
#     val_size = int(val_split * ds_size)
#
#     train_ds = ds.take(train_size)
#     val_ds = ds.skip(train_size).take(val_size)
#     test_ds = ds.skip(train_size).skip(val_size)
#
#     return train_ds, val_ds, test_ds
# train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
#
# -----------------------------------------------------------------------------------

# Creating a Layer for Resizing and Normalization

resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])
# DATA AGUMENTATION
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])
# ---------------- *BULDING THE MODEL *-----------------
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)
print(model.summary())


