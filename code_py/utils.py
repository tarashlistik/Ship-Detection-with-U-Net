import os
import pandas as pd
import numpy as np
import tensorflow as tf
from skimage.io import imread
import keras.backend as K
from sklearn.model_selection import train_test_split
from skimage.morphology import label
from keras.preprocessing.image import ImageDataGenerator

# Directory paths for training and test images
train_image_dir = os.path.join('/Users/admin/Downloads/airbus-ship-detection/train_v2')
test_image_dir = os.path.join('/Users/admin/Downloads/airbus-ship-detection/test_v2')

# List of image names in the training directory
image_name = os.listdir('/Users/admin/Downloads/airbus-ship-detection/train_v2')

# Reading ship segmentations from CSV file
df_ships = pd.read_csv(os.path.join('/Users/admin/Downloads/airbus-ship-detection/train_ship_segmentations_v2.csv'))

# Selecting masks for images in the training set
masks = df_ships[df_ships['ImageId'].isin(image_name)]

# Function to calculate the Dice score
def dice_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2.0 * intersection) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    return dice

# Function to calculate the IoU
def IoU(y_true, y_pred, eps=1e-6):
    is_zero = K.equal(y_true, 0)
    y_true = K.switch(is_zero, 1 - y_true, y_true)
    y_pred = K.switch(is_zero, 1 - y_pred, y_pred)

    # Ensure both tensors have the same data type
    y_true = K.cast(y_true, 'float32')

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return -K.mean((intersection + eps) / (union + eps), axis=0)

# Function to calculate the dice_coefficient
def dice_coefficient(y_true, y_pred):
    intersection = sum(y_true.flatten() * y_pred.flatten())
    union = sum(y_true.flatten()) + sum(y_pred.flatten())

    dice = (2.0 * intersection) / union if union > 0 else 1.0

    return dice


# Function to decode run-length encoded mask
def decode_rle(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# Function to convert a list of masks to a binary image
def mask_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= decode_rle(mask)
    return all_masks

# Adding a 'ships' column to the masks DataFrame
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

# Aggregating ship information for each image
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
                                                               os.stat(os.path.join(train_image_dir,
                                                                                    c_img_id)).st_size/1024)

# Filtering images with a file size greater than 50 KB
unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50]

# Removing the 'ships' column from the masks DataFrame
masks.drop(['ships'], axis=1, inplace=True)

# Balancing the training set by sampling from each group
SAMPLES_PER_GROUP = 2000
balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

# Splitting the data into training and validation sets
train_ids, valid_ids = train_test_split(balanced_train_df,
                test_size = 0.2,
                stratify = balanced_train_df['ships'])

# Merging ship masks with training and validation sets
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)

# Function to generate batches of augmented training images and masks
def gen_make(in_df, batch_size = 48):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(mask_image(c_masks['EncodedPixels'].values), -1)
            c_img = c_img[::3, ::3]
            c_mask = c_mask[::3, ::3]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []

# Creating a generator for augmented training data
train_gen = gen_make(train_df)
train_x, train_y = next(train_gen)

# Generating batches of augmented validation data
valid_x, valid_y = next(gen_make(valid_df, 900))

# Data augmentation parameters
args = dict(featurewise_center = False,
                  samplewise_center = False,
                  rotation_range = 45,
                  width_shift_range = 0.1,
                  height_shift_range = 0.1,
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],
                  horizontal_flip = True,
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')

# Creating ImageDataGenerator instances for data augmentation
image_gen = ImageDataGenerator(**args)
label_gen = ImageDataGenerator(**args)

# Creating an augmented generator for training data
def aug_gen_creat(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        g_x = image_gen.flow(255*in_x,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)

# Creating an augmented generator for training data
cur_gen = aug_gen_creat(train_gen)
x_t, y_t = next(cur_gen)
x_t = x_t[:9]
y_t = y_t[:9]



