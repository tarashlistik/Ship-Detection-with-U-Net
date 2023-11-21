from keras import models, layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

# Import custom utilities from the 'utils' module
from utils import train_df, valid_x, valid_y, gen_make, aug_gen_creat, x_t, IoU, dice_score,  dice_coefficient


# Define the input layer for the neural network
input_img = layers.Input(x_t.shape[1:], name='RGB_Input')
pp_layer = input_img

# Pre-processing layers
pp_layer = layers.AvgPool2D((1, 1))(pp_layer)
pp_layer = layers.GaussianNoise(0.1)(pp_layer)
pp_layer = layers.BatchNormalization()(pp_layer)

# Define the convolutional layers and pooling layers for the encoder
# The architecture follows a U-Net style design for image segmentation

# Encoder
c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pp_layer)
c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)

c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
p3 = layers.MaxPooling2D((2, 2))(c3)

c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

# Decoder
u6 = layers.UpSampling2D(size=(2, 2))(c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

u7 = layers.UpSampling2D(size=(2, 2))(c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

u8 = layers.UpSampling2D(size=(2, 2))(c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

u9 = layers.UpSampling2D(size=(2, 2))(c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

# Output layer
d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

# Upsample the output to match the original image size
d = layers.UpSampling2D(size=(1, 1))(d)

# Create the segmentation model
seg_model = models.Model(inputs=[input_img], outputs=[d])


# Define callbacks for training
checkpoint = ModelCheckpoint("checkpoint_model.hdf5", monitor='val_loss',
                             verbose=1, save_best_only=True,
                             mode='min', save_weights_only=True)

reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                           patience=1, verbose=1,
                           mode='min', min_delta=0.0001,
                           cooldown=0, min_lr=1e-8)

early = EarlyStopping(monitor="val_loss", mode="min",
                      verbose=2, patience=20)

callbacks_list = [checkpoint, early, reduce]


# Compile the model with the Adam optimizer and the defined Dice score metric
seg_model.compile(optimizer=Adam(1e-3), loss=IoU, metrics=['binary_accuracy'])

# Create an augmented image generator using custom utility functions
aug_gen = aug_gen_creat(gen_make(train_df))

# Train the model with the defined callbacks and generators
loss_history = seg_model.fit(aug_gen,
                            steps_per_epoch=9,
                            epochs=20,
                            validation_data=(valid_x, valid_y),
                            callbacks=callbacks_list,
                            workers=1)

# Load the best weights and save the model
seg_model.load_weights('checkpoint_model.hdf5')
seg_model.save('seg_model.h5')

# Plot and save the training history (loss)
plt.figure()
plt.plot(loss_history.history['loss'], label='Loss train')
plt.plot(loss_history.history['val_loss'], label='Loss validation')
plt.xlabel('epohs')
plt.ylabel('loss')
plt.legend()
plt.savefig('Loss_history.png')

# Plot and save the training history (binary accuracy)
plt.figure()
plt.plot(loss_history.history['binary_accuracy'], label='accuracy train')
plt.plot(loss_history.history['val_binary_accuracy'], label='accuracy validation')
plt.xlabel('epohs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('Accuracy_history.png')
