
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === CONFIG ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = "flowers"

# === DATA GENERATORS ===
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# === BUILD MOBILENETV2 MODEL ===
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === CALLBACKS ===
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
checkpoint = ModelCheckpoint('mobilenetv2_best.h5', monitor='val_loss', save_best_only=True)

# === TRAINING ===
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[early_stop, checkpoint])

# === SAVE CLASS NAMES ===
class_names = list(train_gen.class_indices.keys())
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("✅ Model and class names saved.")
