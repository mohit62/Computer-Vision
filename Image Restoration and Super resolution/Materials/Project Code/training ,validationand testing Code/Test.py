#code for predicting sharp image with blur/noisy image as input
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
#prediction = model.predict(input_1)
model =load_model("finalmodel/bestmodel2.h5")
#blurry image
#train_blur = np.load("converteddata/trainBlur.npy")[5]/255
#noisyimage of 256x256
train_blur = np.load("finaldataset/testblur_mixguassian.npy")[1500]/255
train_blur = np.expand_dims(train_blur, axis = 0)
predict_image=model.predict(train_blur)
train_blur=np.squeeze(train_blur, axis = 0)
train_blur = image.array_to_img(train_blur)
#save input blur/noisy image image
train_blur.save("Noisy.png")
#predict denoised/deblurred imaage
predict_image=np.squeeze(predict_image, axis = 0)
predict_image = image.array_to_img(predict_image)
#save deblurred/denoised prediction
predict_image.save("prediction.png")
#train_sharp = np.load("converteddata/trainSharp.npy")[5]/255#save ground truth sharp image
train_sharp = np.load("finaldataset/testsharp_mixguassian.npy")[1500]/255
train_sharp = np.expand_dims(train_sharp, axis = 0)
train_sharp=np.squeeze(train_sharp, axis = 0)
train_sharp = image.array_to_img(train_sharp)
train_sharp.save("sharp.png")
