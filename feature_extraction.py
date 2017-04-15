import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.Models import Sequential
from keras.layers.core import Dense, Activation, Flatten

flags = tf.app.flags
FLAGS =        flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_string('batch_size', '', "batch size")
flags.DEFINE_string('EPOCHS', '', "Epochs")

def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file   - String
        validation_file - String
    """
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
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file,
                                                          FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    EPOCHS =     FLAGS.EPOCHS
    batch_size = FLAGS.batch_size
    # sigma =
    # learning_rate =
    num_classes = len(y_train)
    train_shape = X_train.shape()
    image_shape = np.delete(train_shape,[0])

    # TODO: train your model here
    model = Sequential()
    model.add(Flatten(input_shape=image_shape))
    model.add(Dense(128))                           # is this an appropriate number?
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # preprocess
    X_normalized = np.array( (X_train/255.0) - 0.5 )

    # one hot encoding
    from sklearn.preprocessing import LabelBinarizer

    label_binarizer = LabelBinarizer()
    y_one_hot = label_binerizer.fit_transform(y_train)

    # train
    model.train('adam', 'categorical_crossentropy', ['accuracy'])

    history = model.fit(X_normalized, y_one_hot, nb_epoch=EPOCHS, validation_split=0.2)

    print(history)



# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
