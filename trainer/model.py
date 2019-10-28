import numpy as np 
import keras
from keras.layers import Input, Flatten, Conv2D, Activation, Dense, MaxPooling2D
from keras.models import Model
import os
from matplotlib import image, pyplot
import tensorflow as tf
import argparse
from tensorflow.python.lib.io import file_io

def model(input_shape=(32,32,3)):
    X_input = Input(input_shape)
    X = Conv2D(filters=128, kernel_size=5,strides=(2,2), padding='same', name='conv1')(X_input)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)
    X = Conv2D(filters=64, kernel_size=5,strides=(2,2), padding='same', name='conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)
    X = Conv2D(filters=32, kernel_size=5,strides=(2,2), padding='same', name='conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)
    X = Flatten()(X)
    X = Dense(units=1024, activation='tanh', name='dense1')(X)
    X = Dense(units=10, activation='softmax', name='denseoutput')(X)
    model = Model(inputs=X_input, outputs = X, name='cnncifar10')
    return model
#VISUALIZACION


def main(job_dir, **args):
    
    train_data_path = '/home/mrmister/Ivan/Facultad/4to/deeplearning/gcloud/models/torax/trainer/chest_xray/train/'
    
    train_images_negative = []    
    for filename in os.listdir(train_data_path + 'normal/'):
        negative_img_data = image.imread(train_data_path + 'normal/' + filename)
        train_images_negative.append(negative_img_data)
    train_images_positive = []
    for filename in os.listdir(train_data_path + 'pneumonia'):
        positive_img_data = image.imread(train_data_path + 'pneumonia/' + filename)
        train_images_positive.append(positive_img_data)
    
    val_data_path = '/home/mrmister/Ivan/Facultad/4to/deeplearning/gcloud/models/torax/trainer/chest_xray/val/'
    
    val_images_negative = []    
    for filename in os.listdir(val_data_path + 'normal/'):
        negative_img_data = image.imread(val_data_path + 'normal/' + filename)
        val_images_negative.append(negative_img_data)
    val_images_positive = []
    for filename in os.listdir(val_data_path + 'pneumonia'):
        positive_img_data = image.imread(val_data_path + 'pneumonia/' + filename)
        val_images_positive.append(positive_img_data)    
        
    pyplot.imshow(train_images_positive[0])
    """
    with tf.device('/device:GPU:0'):        
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        X_train = np.reshape(X_train, (50000, 32, 32, 3))
        X_test = np.reshape(X_test, (10000, 32, 32, 3))
        Model = model()
        Model.compile(
            optimizer=keras.optimizers.SGD(lr=0.003),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        Model.summary()
        Model.fit(x = X_train, y = Y_train, batch_size=64, epochs=10, validation_data=(X_test, Y_test))
        Model.save('model.h5')
        with file_io.FileIO('model.h5', mode='rb') as input_f:
            with file_io.FileIO(job_dir + '/model.h5', mode='wb+') as output_f:
                output_f.write(input_f.read())
        """
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=False
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
