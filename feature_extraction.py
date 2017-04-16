import pickle
import tensorflow as tf

# BELOW: import Keras layers you need here
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

flags = tf.app.flags
FLAGS =        flags.FLAGS

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
      {network}_{dataset}_100_bottleneck_features_train.p
      {network}_{dataset}_bottleneck_features_validation.p

    NOTE: I moved all training sets to `training_sets` folder
     "network"in the above filenames, can be one of
        'vgg',
        'inception', or
        'resnet'.
    "dataset" can be either
        'cifar10' or
        'traffic'.
    """

    # training files have been moved to a subdirectory for cleaner root
    training_sets_dir = './training_sets/'

    # build the training/validation file names from supplied flags
    training_file   = training_sets_dir + network + '_' + dataset + '_100_bottleneck_features_train.p'
    validation_file = training_sets_dir + network + '_' + dataset + '_bottleneck_features_validation.p'
    print("Training file  ", training_file)
    print("Validation file", validation_file)

    with open(training_file,   'rb') as f:
        train_data      = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_valid = validation_data['features']
    y_valid = validation_data['labels']

    return X_train, y_train, X_valid, y_valid


def main(_):

    X_train, y_train, X_valid, y_valid = load_bottleneck_data(FLAGS.network, FLAGS.dataset)

    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)

    # BELOW: define your model and hyperparams here
    # make sure to adjust the number of classes based on the dataset
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
    model.add(Dense(128))           # 128 is this an appropriate number?
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # preprocess
    # preprocessing gave terrible results: > 10,000 Epochs to train
    #X_normalized = np.array( (X_train/255.0) - 0.5 )

    # one hot encoding
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()

    print(y_train.shape,   'shape y_train')
    y_train_one_hot = label_binarizer.fit_transform(y_train)
    print(y_train_one_hot.shape, 'shape y_train_one_hot\n')
    y_valid_one_hot = label_binarizer.fit_transform(y_valid)
    print(y_valid_one_hot.shape, 'shape y_valid_one_hot\n')

    # from keras.utils import np_utils
    # y_one_hot = np_utils.to_categorical(y_train, num_classes)
    # print(y_one_hot.shape, 'shape after np_utils\n')

    # train
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train_one_hot, validation_data=(X_valid, y_valid_one_hot), shuffle=True, nb_epoch=EPOCHS, batch_size=batch_size, verbose=2)
 

    print(history)



# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

# TRAINING RESULTS:
# _network_ _dataset_                 validation loss   valida accuracy
# _vgg___   _cifar10_
#Epoch 50   loss: 0.0098 acc: 1.0000  val_loss: 0.8692  val_acc: 0.7603
#Epoch 23   loss: 0.0307 acc: 0.9988  val_loss: 0.3904  val_acc: 0.8791
# _vgg___   _traffic_
#Epoch 50   loss: 0.0046 acc: 1.0000  val_loss: 0.3944  val_acc: 0.8865
#Epoch 31   loss: 0.0145 acc: 1.0000  val_loss: 0.3831  val_acc: 0.8828
# _resnet_  _cifar10_
#Epoch 50   loss: 0.0031 acc: 1.0000  val_loss: 0.9856  val_acc: 0.7420
#Epoch 11   loss: 0.0539 acc: 1.0000  val_loss: 0.8184  val_acc: 0.7376
# _resnet_  _traffic_
#Epoch 50   loss: 0.0032 acc: 1.0000  val_loss: 0.6684  val_acc: 0.8104
#Epoch 16   loss: 0.0370 acc: 1.0000  val_loss: 0.6383  val_acc: 0.8023
# _inception__cifar10_
#Epoch 50   loss: 0.0025 acc: 1.0000  val_loss: 1.2242  val_acc: 0.6656
#Epoch  8   loss: 0.0768 acc: 1.0000  val_loss: 1.0226  val_acc: 0.6614
# _inception__traffic_
#Epoch 50   loss: 0.0016 acc: 1.0000  val_loss: 0.9360  val_acc: 0.7436
#Epoch 10   loss: 0.0448 acc: 1.0000  val_loss: 0.8962  val_acc: 0.7260
#(1Dense,E50)loss:0.0134 acc: 1.0000  val_loss: 0.8384  val_acc: 0.7555
