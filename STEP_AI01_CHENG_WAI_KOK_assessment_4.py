#%%
#1. Import packages
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.callbacks import EarlyStopping
import cv2, datetime, os
import matplotlib.pyplot as plt
import numpy as np
import modules
import tensorflow as tf

#%%
#2. Set parameters
BATCH_SIZE = 16
BUFFER_SIZE = 1000
SEED = 42
EPOCHS = 10
VAL_SUBSPLITS = 5

# %%
#3. Data preparation
#3.1 Prepare the path
train_path = r"C:\Users\Guest1\Desktop\DeepLearning\STEP_AI01_CHENG_WAI_KOK_assessment_4\dataset\train"
val_path = r"C:\Users\Guest1\Desktop\DeepLearning\STEP_AI01_CHENG_WAI_KOK_assessment_4\dataset\test"

#3.2 Prepare empty list to hold the data
images = []
masks = []
val_images = []
val_masks = []

#3.3 Load the train images using opencv
image_dir = os.path.join(train_path,'inputs')
for image_file in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir,image_file))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    images.append(img)

#3.4 Load the train masks
masks_dir = os.path.join(train_path,'masks')
for mask_file in os.listdir(masks_dir):
    mask = cv2.imread(os.path.join(masks_dir,mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks.append(mask)

#3.5 Load the test images
val_image_dir = os.path.join(val_path,'inputs')
for val_image_file in os.listdir(val_image_dir):
    val_image = cv2.imread(os.path.join(val_image_dir,val_image_file))
    val_image = cv2.cvtColor(val_image,cv2.COLOR_BGR2RGB)
    val_image = cv2.resize(val_image,(128,128))
    val_images.append(val_image)

#3.6 Load the test masks
val_mask_dir = os.path.join(val_path,'masks')
for val_mask_file in os.listdir(val_mask_dir):
    val_mask = cv2.imread(os.path.join(val_mask_dir,val_mask_file), cv2.IMREAD_GRAYSCALE)
    val_mask = cv2.resize(val_mask,(128,128))
    val_masks.append(val_mask)

#%%
#3.7 Convert the list of np array into a np array
images_np = np.array(images)
masks_np = np.array(masks)
val_images_np = np.array(val_images)
val_masks_np = np.array(val_masks)

#%%
#4. Data preprocessing
#4.1 Expand the mask dimension
masks_np_exp = np.expand_dims(masks_np,axis=-1)
val_masks_np_exp = np.expand_dims(val_masks_np,axis=-1)
#Check the mask output
print(np.unique(masks_np_exp[0]))
print(np.unique(val_masks_np_exp[0]))

#%%
#4.2 Convert the mask values from [0,255] into [0,1]
converted_masks = np.round(masks_np_exp / 255.0).astype(np.int64)
converted_val_masks = np.round(val_masks_np_exp / 255.0).astype(np.int64)
#Check the mask output
print(np.unique(converted_masks[0]))
print(np.unique(converted_val_masks[0]))

#%%
#4.3 Normalize the images
converted_images = images_np / 255.0
converted_val_images = val_images_np / 255.0

#%%
#5. Perform train-test split
X_train,X_test,y_train,y_test = train_test_split(converted_images, converted_masks,test_size=0.2,shuffle=True,random_state=SEED)

#%%
#6. Convert the numpy arrays into tensor slices
X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

#%%
#7. Combine the images and masks using the zip method
train_dataset = tf.data.Dataset.zip((X_train_tensor,y_train_tensor))
test_dataset = tf.data.Dataset.zip((X_test_tensor,y_test_tensor))

#%%
#8. Build the dataset
AUTOTUNE = tf.data.AUTOTUNE
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_dataset.batch(BATCH_SIZE)

# %%
#9. Visualize some pictures as example
for images, masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])

# %%
#10. Model development
#10.1 Use a pretained model as the feature extractor
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

#10.2 Use these activation layers as the outputs from the feature extractor (some of these outputs will be used to perform concatenation at the upsampling path)
layer_names = [
    'block_1_expand_relu',      #64x64
    'block_3_expand_relu',      #32x32
    'block_6_expand_relu',      #8x8
    'block_13_expand_relu',     #4x4
    'block_16_project'
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#10.3 Instantiate the feature extrator
down_stack = keras.Model(inputs=base_model.input,outputs=base_model_outputs)
down_stack.trainable = False

#10.4 Define the upsampling path
up_stack = [
    pix2pix.upsample(512,3),    #4x4 --> 8x8
    pix2pix.upsample(256,3),    #8x8 --> 16x16
    pix2pix.upsample(128,3),    #16x16 --> 32x32
    pix2pix.upsample(64,3),     #32x32 --> 64x64
]

#%%
#10.5 Use Unet to create the model
OUTPUT_CHANNELS = 3
model = unet(OUTPUT_CHANNELS)
model.summary()
keras.utils.plot_model(model)

#%%
#11. Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer,loss=loss,metrics=['acc'])

# %%
#12. Display Prediction 
show_predictions()

#%%
# 13. Create the tensorboard callbacks
#es = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)
log_path = os.path.join('log_dir',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = keras.callbacks.TensorBoard(log_dir=log_path)

# %%
#14. Model training
VALIDATION_STEPS = len(X_test) // BATCH_SIZE // VAL_SUBSPLITS
#history = model.fit(train_batches,val_train_batches, validation_data=(test_batches, val_test_batches), validation_steps=VALIDATION_STEPS,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,callbacks=[DisplayCallback(),es,tb])
history = model.fit(train_batches,validation_data=test_batches, validation_steps=VALIDATION_STEPS,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,callbacks=[DisplayCallback(),tb])

# %%
#15. Model deployment
#To view the prediction
show_predictions(test_batches,3)

# %%
#16. Evaluate the model using test data
#history.history.keys()
test_loss,test_acc = model.evaluate(converted_val_images,converted_val_masks,verbose=1)

print("Loss = ",test_loss)
print("Accuracy = ",test_acc)

#%%
#17. Plot the chart
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation','loss','validation_loss'])
plt.show()

#%%
#18. Model Saving
# To create folder if not exists
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

saved_path = os.path.join('saved_models','model.h5')
model.save(saved_path)
# %%
