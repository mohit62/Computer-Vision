from keras.models import load_model
import numpy as np
import math
#calculating psnr values from mse
test_blur=np.load("finaldataset/trainblur_mixguassian.npy")[2000:2010]/255
test_sharp=np.load("finaldataset/trainsharp_mixguassian.npy")[2000:2010]/255
loaded_model =load_model("finalmodel/bestmodel2.h5")
mse = loaded_model.evaluate(test_blur,test_sharp, verbose=0)
psnr=20 * math.log10(255. / (mse*255*255))
print(psnr)