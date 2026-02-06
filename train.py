'''
    RUN IT first and the RUN app.py or else
'''

from keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Dropout,Dense,Flatten
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint

print("\nModel is initailizing...\n")

model = Sequential()

model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(128,128, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(96, kernel_size = (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(25,activation='softmax'))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

train_img_data = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

val_img_data = ImageDataGenerator(rescale = 1./255)

train_dataset = train_img_data.flow_from_directory(
    "dataset/train",
    target_size=(128,128),
    batch_size = 32,
    class_mode="categorical"
)
val_dataset = val_img_data.flow_from_directory(
    "dataset/test",
    target_size=(128,128),
    batch_size = 32,
    class_mode="categorical"
)

callback = [
    EarlyStopping(patience=150,restore_best_weights=True),
    ModelCheckpoint(filepath="bestModel.keras",save_best_only=True,save_weights_only=False,verbose=1)
]



print("\n Model was initailzied.\nModel is training...\n")

model.fit(
    train_dataset,
    steps_per_epoch=375,
    epochs=10,
    validation_data=val_dataset,
    validation_steps=125,
    callbacks=callback
)

# # save model
# print("  Model trained.\n")
# model.save("model.keras")
# model.save_weights("model1.h5")
# print("Model is saved on your disk.")
