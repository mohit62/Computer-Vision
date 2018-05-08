#Code for training model without skip connections
from keras.models import Model
from keras.layers import Input,Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Deconv2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

#noisy image dataset of 256x256 resolution images in numpy format
train_blur = np.load("noisydataset/trainblur_mixguassian.npy")/255
#sharp image dataset of 256x256 resolution images in numpy format
train_sharp = np.load("noisydataset/trainsharp_mixguassian.npy")/255

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
decon_layer_3=Deconv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(decon_layer_2)
decon_layer_3 = BatchNormalization()(decon_layer_3)
decon_layer_4=Deconv2D(64,kernel_size=(3, 3),strides=2,activation='relu',padding='same')(decon_layer_3)
decon_layer_4 = BatchNormalization()(decon_layer_4)
#64x64

decon_layer_5=Deconv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(decon_layer_4)
decon_layer_5 = BatchNormalization()(decon_layer_5)
decon_layer_6=Deconv2D(64,kernel_size=(3, 3),strides=2,activation='relu',padding='same')(decon_layer_5)
decon_layer_6 = BatchNormalization()(decon_layer_6)
#128x128

decon_layer_7=Deconv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(decon_layer_6)
decon_layer_7 = BatchNormalization()(decon_layer_7)
decon_layer_8=Deconv2D(64,kernel_size=(3, 3),strides=2,activation='relu',padding='same')(decon_layer_7)
decon_layer_8 = BatchNormalization()(decon_layer_8)
#256x256

decon_layer_9=Deconv2D(64,kernel_size=(3, 3),strides=1,activation='relu',padding='same')(decon_layer_8)
decon_layer_9 = BatchNormalization()(decon_layer_9)
decon_layer_10=Deconv2D(64,kernel_size=(3, 3),activation="relu",strides=1,padding='same')(decon_layer_9)
decon_layer_10 = BatchNormalization()(decon_layer_10)
#256x256

#sigmoid activation used as we want output between 0 and 1
Output=Conv2D(3,kernel_size=(3, 3),activation="sigmoid",strides=1,padding='same')(decon_layer_10)
model = Model(input_1,Output)
print(model.summary())
#we use Adam optimizer for training
d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=d_opt, loss='mse')
checkpointer = ModelCheckpoint(filepath='withoutskipbest.h5', verbose=1, save_best_only=True)
model.fit(train_blur, train_sharp,batch_size=8, epochs=20,shuffle=True,validation_split=0.2,callbacks=[checkpointer])
model.save("withoutskipmodel.h5")
