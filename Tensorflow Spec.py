# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 02:41:43 2020

@author: Kachukwu
"""
## FITTING A NN MODEL
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# model.fit(xs, ys, epochs=500)
# print (model.predict([10.0]))

# print (tf.__version__)

## CALLBACK FUNCTION ON EPOCHS
def train_fashion_mnist():
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('loss')<0.2):
                print ("\n Reached 88% accuracy so cancelling training!")
                self.modell.stop_training = True


    ## COMPUTER VISION ON FASHION MNIST (CLOTHING IMAGES)

    callbacks = myCallback()
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_label), (test_images, test_label) = fashion_mnist.load_data()

    import matplotlib.pyplot as plt
    plt.imshow(train_images[89])
    # print (train_label[10])
    # print (train_images[10])

    # Normalizing
    train_images = train_images/255.0
    test_images = test_images/255.0
    
    modell = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

    # Building nn
    modell.compile (optimizer = tf.compat.v1.train.AdamOptimizer(),
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])

    # Fitting model
    history = modell.fit (train_images, train_label, epochs=5, callbacks=[callbacks])
    return history.epoch, history.history['acc'][-1], modell.evaluate(test_images, test_label)

    # Evaluate with test data modell.evaluate()
    

# train_fashion_mnist()



## COMPUTER VISION ON MNIST (HANDWRITING)

# def train_mnist():
#     class myCall(tf.keras.callbacks.Callback):
#         def on_epoch_end(self, epoch, logs={}):
#             if (logs.get('loss')<0.02):
#                 print ("\n Reached 88% accuracy so cancelling training!")
#                 self.model1.stop_training = True

    # callbacks = myCall()
    # # mnist = keras.datasets.mnist
    # # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # x_train = x_train/255.0
    # x_test = x_test/255.0
    
    # modell = keras.Sequential([
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(128, activation=tf.nn.relu),
    #     keras.layers.Dense(10, activation=tf.nn.softmax)
    # ])
        
    # modell.compile (optimizer = tf.compat.v1.train.AdamOptimizer(),
    #                 loss = 'sparse_categorical_crossentropy',
    #                 metrics = ['accuracy'])

    # history = modell.fit (x_train, y_train, epochs=10, callbacks=[callbacks])
    # return history.epoch, history.history['acc'][-1], modell.evaluate(x_test, y_test)



### ADDING CNN TO DNN

## FASHION MNIST



fash_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fash_mnist.load_data()

train_images = train_images.reshape(60000,28,28,1)
train_images = train_images/255.0
test_images = test_images.reshape(10000,28,28,1)
test_images = test_images/255.0

model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
               metrics=['accuracy'])

# model1.summary()
# model1.fit(train_images, train_labels, epochs=5)

# test_loss = model1.evaluate(test_images, test_labels)

# print (test_labels[:100])

''' visualizing the convolutions




'''


## HOW A CONVOLUTION WORKS


# import cv2
# import numpy as np
# from scipy import misc
# i = misc.ascent()

# import matplotlib.pyplot as plt
# plt.grid(False)
# plt.gray()
# plt.axis('off')
# plt.imshow(i)
# plt.show()

# i_transformed = np.copy(i)
# size_x = i_transformed.shape[0]
# size_y = i_transformed.shape[1]















### CONVOLUTIONARY NETS WITH COMPLEX DATA AND PREPROCESSING WITH GENERATORS



# wget --no-check-certificate \
# http://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
#         -o /tmp/horse-or-human.zip
    
import os 
import zipfile

local_zip = 'horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('horse-or-human')
zip_ref.close()


# Directory with our training horse pictures
train_horse_dir = os.path.join('horse-or-human/filtered-horses')

# Directory with our training human pictures
train_human_dir = os.path.join('horse-or-human/filtered-humans')

# train_horse_names = os.listdir(train_horse_dir)
# print (train_horse_names[:10])

# train_human_names = os.listdir(train_human_dir)
# print (train_human_names[:10])

# print ('total training horse images:', len(os.listdir(train_horse_dir)))
# print ('total training human images:', len(os.listdir(train_human_dir)))


# import matplotlib.image as plg
nrows = 4
ncols = 4
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index += 8 
# next_horse_pix = [os.path.join(train_horse_dir,fname)
#                   for fname in train_horse_names[pic_index-8:pic_index]]
# next_human_pix = [os.path.join(train_human_dir,fname)
#                   for fname in train_human_names[pic_index-8:pic_index]]

# for i, img_path in enumerate(next_horse_pix+next_human_pix):
    ## set up subplot; subplot indices start at 1
    # sp = plt.subplot(nrows,ncols, i + 1)
    # sp.axis('Off') #Don't show axis or gridlines
    
    # img = plg.imread(img_path)
    # plt.imshow(img)
    
# plt.show()



from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Instantiating an ImageGen
train_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size = (300,300),
#     batch_size = 128,
#     class_mode='binary')

# test_datagen = ImageDataGenerator(rescale=1./255)

# validation_generator = test_datagen.flow_from_directory(
#     validation_dir,
#     target_size = (300,300),
#     batch_size = 32,
#     class_mode='binary')


modehl = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

modehl.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])



# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=8,
#     epochs=15,
#     validation_data=validation_generator,
#     validation_steps=8,
#     verbose=2)


## PREDICTION ON TRAINED MODEL
import numpy as np
# from google.colab import files  # specific to Colab
from keras.preprocessing import image

# uploaded = files.upload()

# for fn in uploaded.keys():
    
#     #Pedicting Image
#     path = '/content/' + fn
#     img = image.load_img(path, target_size=(300,300))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
    
#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)
#     print (classes[0])
#     if classes[0]>0.5:
#         print (fn + " is a human")
#     else:
#         print (fn + " is a horse")












