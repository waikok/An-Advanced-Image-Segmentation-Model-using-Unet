#%%
#1. Import packages
from tensorflow import keras
import matplotlib.pyplot as plt


# %%
#2. Define classes
#2.1 Define data augmentation pipeline as a single layer through subclassing
class Augment(keras.layers.Layer):
    def __init__(self,seed=123):
        super().__init__()
        self.augment_inputs = keras.layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = keras.layers.RandomFlip(mode='horizontal',seed=seed)

    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels

#2.2. Create a callback funstion to make use of the show_predictions function
class DisplayCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample prediction after epoch{}\n'.format(epoch+1))

# %%
#3. Define functions
#3.1 Create function to display images
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image', 'True Mask','Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
    plt.show()

#3.2 Use functional API to construct the entire U-net
def unet(output_channels:int):
    inputs = keras.layers.Input(shape=[128,128,3])
    #Downsample through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    #Build the upsampling path and establish the concatenation
    for up, skip in zip(up_stack,skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x,skip])

    #Use a transpose convolution layer to perform the last upsampling, this will become the output layer
    last = keras.layers.Conv2DTranspose(filters=output_channels,kernel_size=3,strides=2,padding='same') #64x64 --> 128x128
    outputs = last(x)

    model = keras.Model(inputs=inputs,outputs=outputs)

    return model


#3.3 Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])


# %%
