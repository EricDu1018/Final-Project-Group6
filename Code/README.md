Source of dataset: https://www.kaggle.com/puneet6060/intel-image-classification


Instructions for how to run the code

1. data_prep.py -- This file allow us to load the data.

2. vgg16.py
  vgg19.py
  res.py
  inception.py
  inceptionRes.py
  -- These five files use pre-trained networks without data augmentation.

3.aug_plots.py -- This file plots how selected picture is augmented.

4. vgg16_da.py
  vgg19_da.py
  ResNet152_da.py
  InceptionV3_da.py
  InceptionResV2_da.py
  -- These five files use pre-trained networks with data augmentation.
  
5. Self_trained_cnn.py -- This file performs our self-trained network.

6. vgg_16__Prediction.py
  vgg_19__Prediction.py
  ResNet__Prediction.py
  Self_trained_Prediction.py
  -- These files give us the results of prediction.

