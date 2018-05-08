#Code for deblurring images with motion blur
from keras.models import Model
from keras.layers import Input,Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Deconv2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

#motion blur image dataset of 256x256 resolution images in numpy format
train_blur = np.load("converteddata/trainBlur.npy")/255
#sharp image dataset of 256x256 resolution images in numpy format
train_sharp = np.load("converteddata/trainsharp.npy")/255

input_1=Input(shape=(256,256,3))
con_layer_1=Conv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(input_1)
con_layer_1 = BatchNormalization()(con_layer_1)
con_layer_2=Conv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(con_layer_1)
con_layer_2 = BatchNormalization()(con_layer_2)
#256x256
con_layer_3=Conv2D(64,kernel_size=(3, 3),strides=2,activation='relu',padding='same')(con_layer_2)
con_layer_3 = BatchNormalization()(con_layer_3)
con_layer_4=Conv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(con_layer_3)
con_layer_4 = BatchNormalization()(con_layer_4)
#128x128
con_layer_5=Conv2D(64,kernel_size=(3, 3),strides=2,activation='relu',padding='same')(con_layer_4)
con_layer_5 = BatchNormalization()(con_layer_5)
con_layer_6=Conv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(con_layer_5)
con_layer_6 = BatchNormalization()(con_layer_6)
#64x64
con_layer_7=Conv2D(64,kernel_size=(3, 3),strides=2,activation='relu',padding='same')(con_layer_6)
con_layer_7 = BatchNormalization()(con_layer_7)
con_layer_8=Conv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(con_layer_7)
con_layer_8 = BatchNormalization()(con_layer_8)
#32x32
con_layer_9=Conv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(con_layer_8)
con_layer_9 = BatchNormalization()(con_layer_9)
con_layer_10=Conv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(con_layer_9)
con_layer_10 = BatchNormalization()(con_layer_10)
#32x32
decon_layer_1=Deconv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(con_layer_10)
decon_layer_1 = BatchNormalization()(decon_layer_1)
decon_layer_2=Deconv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(decon_layer_1)
decon_layer_2 = BatchNormalization()(decon_layer_2)
#32x32
#skip connection b/w convolution layer 8 and deconvolution layer 2
skip1=Add()([con_layer_8, decon_layer_2])
skip1=Activation("relu")(skip1)
decon_layer_3=Deconv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(skip1)
decon_layer_3 = BatchNormalization()(decon_layer_3)
decon_layer_4=Deconv2D(64,kernel_size=(3, 3),strides=2,activation='relu',padding='same')(decon_layer_3)
decon_layer_4 = BatchNormalization()(decon_layer_4)
#64x64
#skip connection b/w convolution layer 6 and deconvolution layer 4
skip2 = Add()([con_layer_6, decon_layer_4])
skip2=Activation("relu")(skip2)
decon_layer_5=Deconv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(skip2)
decon_layer_5 = BatchNormalization()(decon_layer_5)
decon_layer_6=Deconv2D(64,kernel_size=(3, 3),strides=2,activation='relu',padding='same')(decon_layer_5)
decon_layer_6 = BatchNormalization()(decon_layer_6)
#128x128
#skip connection b/w convolution layer 4 and deconvolution layer 6
skip3=Add()([con_layer_4, decon_layer_6])
skip3=Activation("relu")(skip3)
decon_layer_7=Deconv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(skip3)
decon_layer_7 = BatchNormalization()(decon_layer_7)
decon_layer_8=Deconv2D(64,kernel_size=(3, 3),strides=2,activation='relu',padding='same')(decon_layer_7)
decon_layer_8 = BatchNormalization()(decon_layer_8)
#256x256
#skip connection b/w convolution layer 2 and deconvolution layer 8
skip4 = Add()([con_layer_2, decon_layer_8])
skip4=Activation("relu")(skip4)
decon_layer_9=Deconv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(skip4)
decon_layer_9 = BatchNormalization()(decon_layer_9)
decon_layer_10=Deconv2D(64,kernel_size=(3, 3),activation="relu",strides=1,padding='same')(decon_layer_9)
decon_layer_10 = BatchNormalization()(decon_layer_10)
#256x256
#skip connection b/w convolution layer 1 and deconvolution layer 10
Output = Add()([con_layer_1, decon_layer_10])
Output=Activation("relu")(Output)
#sigmoid activation used as we want output between 0 and 1
Output=Conv2D(3,kernel_size=(3, 3),activation="sigmoid",strides=1,padding='same')(Output)
model = Model(input_1,Output)
print(model.summary())
#we use Adam optimizer for training
d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=d_opt, loss='mse')
checkpointer = ModelCheckpoint(filepath='bestmodeldeblur2.h5', verbose=1, save_best_only=True)
model.fit(train_blur, train_sharp,batch_size=8, epochs=50,shuffle=True,validation_split=0.2,callbacks=[checkpointer])
model.save("deblursaved2.h5")
