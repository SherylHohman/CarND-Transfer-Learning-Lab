import pickle
import tensorflow as tf
# BELOW: import Keras layers you need here
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

flags = tf.app.flags
FLAGS =        flags.FLAGS

# command line flags
# flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
# flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")

# 'vgg', 'resnet', or 'inception'
flags.DEFINE_string('network', 'vgg', "Bottleneck features training file (.p)")
# 'cifar10', or 'traffic'
flags.DEFINE_string('dataset', 'cifar10', "Bottleneck features validation file (.p)")

flags.DEFINE_string('batch_size', '128', "batch size")
flags.DEFINE_string('epochs', '50', "EPOCHS")

#def load_bottleneck_data(training_file, validation_file):
def load_bottleneck_data(network, dataset):
    """
    Utility function to load bottleneck features.
    Arguments:
    #     network   - String
    #     dataset   - String
    Used to build the filenames for the training and validation files
    """
    """
     "network"in the above filenames, can be one of
        'vgg',
        'inception', or
        'resnet'.
    "dataset" can be either
        'cifar10' or
        'traffic'.

   Files to try:
    NOTE: I moved all training sets to `training_sets` folder

    {network}_{dataset}_100_bottleneck_features_train.p
    {network}_{dataset}_bottleneck_features_validation.p

    """
    # I moved the training files into a subdirectory for a cleaner root
    training_sets_dir = './training_sets/'

    # build the training/validation file names from supplied flags
    training_file   = training_sets_dir + network + '_' + dataset + '_100_bottleneck_features_train.p'
    validation_file = training_sets_dir + network + '_' + dataset + '_bottleneck_features_validation.p'
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file,   'rb') as f:
        train_data      = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val   = validation_data['features']
    y_val   = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    #X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.network, FLAGS.dataset)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # BELOW: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    EPOCHS =     int(FLAGS.epochs)
    batch_size = int(FLAGS.batch_size)
    print("EPOCHS", EPOCHS, 'batch_size', batch_size)

    num_classes = len(np.unique(y_train))
    print(num_classes, 'num_classes')

    train_shape = X_train.shape
    image_shape = train_shape[1:]

    # BELOW: train your model here
    model = Sequential()
    model.add(Flatten(input_shape=image_shape))
    model.add(Dense(128))                         # 128 is this an appropriate number?
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # preprocess
    #X_normalized = np.array( (X_train/255.0) - 0.5 )

    # one hot encoding
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()

    print(y_train.shape,   'shape y_train')
    y_one_hot = label_binarizer.fit_transform(y_train)
    print(y_one_hot.shape, 'shape y_one_hot')

    # from keras.utils import np_utils
    # y_one_hot = np_utils.to_categorical(y_train, num_classes)
    # print(y_one_hot.shape, 'shape after np_utils\n') #(10000, 1000)

    # train
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_one_hot, shuffle=True, nb_epoch=EPOCHS, batch_size=batch_size, verbose=2)
    # history = model.fit(X_normalized, y_one_hot, shuffle=True, nb_epoch=EPOCHS, batch_size=batch_size, verbose=2)


    print(history)



# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()



    # Training Results:
    # 0s - loss: 0.8378 - acc: 0.6980
    # Epoch 3000/3000
    # 0s - loss: 0.8560 - acc: 0.6830

    # ERROR at end of training:
    # <keras.callbacks.History object at 0x00000000076C4FD0>
    # Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x000000000773D668>>
    # Traceback (most recent call last):
    #   File "C:\Users\i\Anaconda3\envs\carnd-term1\lib\site-packages\tensorflow\python\client\session.py", line 581, in __del__
    # AttributeError: 'NoneType' object has no attribute 'TF_DeleteStatus'

#     Epoch 10000/10000
#     0s - loss: 0.4135 - acc: 0.8640
#other version:
    # Epoch 10000/10000
    # 0s - loss: 0.6913 - acc: 0.7630
    # <keras.callbacks.History object at 0x00000000076C1748>
# <keras.callbacks.History object at 0x00000000076C3780>

# current version achieves accuracy 1.000 at epoch 16
# and by epoch 50: loss 0.0100, acc: 1.000
#Something must be wrong as solution has accuracy ~ 85% at 50 epochs