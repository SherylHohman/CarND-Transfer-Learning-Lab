
# coding: utf-8

# ## Transfer Learning Lab part 2:  
# # Cifar10 : Run Traffic Sign Classifier on Cifar10 dataset  
# ### Compare performance of my Traffic Sign Model to Transfer Learning Model on the Cifar10 dataset   
# *Transfer Learning Model will be done in a separate notebook, after completing this one  
# 
# Note: this dataset is much more diverse 
# - it is expected to perform poorly on the cifar10 dataset,  
# - as comparted to the traffic signs dataset.
#     
# This Notebook, all cells below this one, is a  
# #### direct copy of the Traffic Sign Recognition Classifier.    
# ### Cell **IN [1]**, contains the Only Change:   
#   - **In [25]**: I load the **Cifar10** dataset 
#   - **In [35]**: I split Cifar10 training data into training and validation sets
#   - **In [1]**:  was the the **traffic sign dataset**  
#   - **In [21]**: commented out traffic sign dataset is now in this cell  
#   - **In [16]**: calculation number of classes no longer relies on reading in a csv file of classnames !! (np.unique - waaay better) :-)
#   
# #### Ignore all references to traffic signs.   
# #### Just re-using the model itself.  

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[17]:

# download Cifar10 dataset
from keras.datasets import cifar10

(X_train_ORIG, y_train_ORIG), (X_test_ORIG, y_test_ORIG) = cifar10.load_data()

# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.

# train will be split into train and valid sets, and named as X_train_ORIG, and X_test_ORIG, etc at that point
y_train_ORIG = y_train_ORIG.reshape(-1)
y_test_ORIG  = y_test_ORIG.reshape(-1)


print("data loaded")
print(X_train_ORIG.shape, y_test_ORIG.shape)


# In[18]:

## I need a validation set, else cannot train my model..
#    np.train_valid_split() (data, labels, valid_size=0.20, random_state=42)
#    could be used to randomly split the dataset
# however, to preserve proportion of examples of each class in each dataset, I'll implement a
#    stratified split:

def stratified_dataset_split(X_all, y_all):
    
    import numpy as np  

    def get_indexes_for_split(y_train, training_proportion=0.85):
 
        y_train=np.array(y_train)
        # read class id,s
        class_ids = np.unique(y_train)
        
        # since class id's are know, create them all, initialize them all to empty sets
        image_indexes = {}
        for class_id in class_ids:
            image_indexes[class_id] = []
            
        # sort images (via index num) into lists keyed by class_id
        for image_index in range(len(y_train)):
            image_class_id = y_train[image_index]
            image_indexes[image_class_id].append(image_index)
            
        # shuffle the indexes of images in each category
        for class_id in class_ids:
            np.random.shuffle(image_indexes[class_id])
        
        # siphon off training_proportion of each list into training set, remaining become validation set
        training_set_indexes, validation_set_indexes = [ [], [] ]
        for class_id in class_ids:
            num_images_in_class = len(image_indexes[class_id])
            num_training = int((training_proportion * num_images_in_class) // 1)
            #num_validation = num_images_in_class - num_training
            
            training_set_indexes   += image_indexes[class_id][:num_training]  # or need to use extend?
            validation_set_indexes += image_indexes[class_id][num_training:]
            
            tr = len(training_set_indexes)
            va = len(validation_set_indexes)
        print(tr/len(y_train), va/len(y_train), tr, va, tr + va, len(y_train)) 
        
        return training_set_indexes, validation_set_indexes
        

    #y = np.array([1,1,2,2,3,3])
    training_indexes, validation_indexes = get_indexes_for_split(y_all, training_proportion=0.85)
    #print(training_indexes)
    #print(validing_indexes)
    
    X_train_new, y_train_new = ([],[])
    for training_index in training_indexes:
            X_train_new.append(X_all[training_index])
            y_train_new.append(y_all[training_index])
          
    X_valid_new, y_valid_new = ([], [])
    for validation_index in validation_indexes :
            X_valid_new.append(X_all[validation_index])
            y_valid_new.append(y_all[validation_index])
       
    assert( len(X_train_new) == len(y_train_new) )
    assert( len(X_valid_new) == len(y_valid_new) )
    print("train: ", len(X_train_new)/len(X_all), "valid", len(X_valid_new)/len(X_all) )
    
    return np.asarray(X_train_new), np.asarray(y_train_new), np.asarray(X_valid_new), np.asarray(y_valid_new)

# split X_train, y_train into training and validation sets
X_train_ORIG, y_train_ORIG, X_valid_ORIG, y_valid_ORIG = stratified_dataset_split(X_train_ORIG, y_train_ORIG)

print(X_train_ORIG.shape, y_train_ORIG.shape, X_valid_ORIG.shape, y_valid_ORIG.shape)


# In[19]:

"""  
### Traffic Sign Data:

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and validation data

training_file  = './traffic-signs-data/train.p'
validation_file= './traffic-signs-data/valid.p'
testing_file   = './traffic-signs-data/valid.p'
 
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train_ORIG, y_train_ORIG = train['features'], train['labels']
X_valid_ORIG, y_valid_ORIG = valid['features'], valid['labels']
X_test_ORIG,  y_test_ORIG  =  test['features'],  test['labels']

assert(len(X_train_ORIG) == len(y_train_ORIG))
assert(len(X_valid_ORIG) == len(y_valid_ORIG))
assert(len(X_test_ORIG)  == len(y_test_ORIG))
"""
print('')  # hide echo of commented out code as string


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[20]:

import numpy as np

# TODO: Number of training examples
n_train = X_train_ORIG.shape[0]
n_valid = X_valid_ORIG.shape[0]

# Number of testing examples.
n_test = X_test_ORIG.shape[0]

# What's the shape of an traffic sign image? 
image_shape = X_train_ORIG.shape[1:]

# How many unique classes/labels there are in the dataset.
num_classes = len(np.unique(y_train_ORIG))

print("Image data shape =", image_shape,"\n")
print("Number of classes =", num_classes)
n_total = n_train + n_valid + n_test
print("Number of training examples   =", "{:5}".format(n_train), ": {:4.1f}% of total".format(100*n_train/n_total))
print("Number of validation examples =", "{:5}".format(n_valid), ": {:4.1f}% of total".format(100*n_valid/n_total))
print("Number of testing examples    =", "{:5}".format(n_test),  ": {:4.1f}% of total".format(100*n_test/n_total))



# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

# In[21]:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt

# Show Visualizations in the notebook.
get_ipython().magic('matplotlib inline')


# SH look at training set labels. count how many images there are of each label
import numpy as np
from collections import Counter, OrderedDict

def plot_data_in_fig(data, title, fig, plot_loc):
    counts = Counter(data)
    counts = OrderedDict(sorted(counts.items()))
    counts = [counts[i] for i in range(len(counts))]

    y_data = counts
    x_data = range(len(y_data))    
    
    plt.subplot(plot_loc, title=title)
    plt.xlabel("traffic sign class number\n")
    plt.ylabel("number of images")
    plt.bar(x_data, y_data)

print("Let's see how traffic sign examples are distributed across \n  the Training, Validation, and Test sets:\n")
print("The distribution appears similar across the three sets, \n  though not uniform across the classes.")

# set figure width, height in inches   
fig1 = plt.figure(1, figsize=(6, 12))

plot_data_in_fig(y_train_ORIG, "Training Data \ndistribution of images per class",  fig1, 311)
plot_data_in_fig(y_valid_ORIG, "Validation Data \ndistribution of images per class",fig1, 312)
plot_data_in_fig(y_test_ORIG,  "Test Data \ndistribution of images per class",      fig1, 313)

# prevent overlapping of labels with subplots
plt.tight_layout()
plt.show()

# save figure to file
#fig1.savefig("data_plotted_image_distribution_amongst_classes.png")  # results in 160x120 px image
print("figure saved as: 'data_plotted_image_distribution_amongst_classes.png'")


# In[22]:

# diplay list of sample images from training data set
def display_images(images):
    num_channels = images[0].shape[-1]
    #print("shape 1st image", images[0].shape)
    num_images = len(images)
    #print(num_images, "num_images, num_channels:", num_channels)
    fig = plt.figure(1)
    rows = 1
    cols = int(num_images/rows)
    for i in range(num_images):
        plt.subplot(rows, cols, i+1)
        if num_channels == 3:
            # color image
            plt.imshow(images[i])
        else:
            # assume grayscale:
            # grayscale is 3D, not 4D: it does NOT have a "num_channels==1" dimension
            # it has simple shape of (num_examples, 32, 32) as opposed to (num_examples, 32, 32, num_channels==3)
            plt.imshow(images[i], cmap = plt.get_cmap('gray'))
    plt.show()
    return fig

print("Sample images from training data set")
sample_images = [X_train_ORIG[50], X_train_ORIG[500], X_train_ORIG[1000]]
fig2 = display_images(sample_images)


# save figure to file
#fig2.savefig("sample_traffic_signs_from_training_set.png", dpi=25)  # results in 160x120 px image
print("figure saved as: 'sample_traffic_signs_from_training_set.png'")

# consider showing histogram of individual sample images
# consider showing average histogram of all images


# In[ ]:




# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[23]:

def test_reshaping_3D_into_4D():
    # grayscale images have no depth dimension
    # we need them to have a depth dimension of 1
    gr = np.zeros([3,2,2])
    print(gr)
    gr = gr.reshape(3,2,2,1)
    print(gr)

    gr = np.zeros((3,2,2))
    print(gr)
    gr = np.reshape(gr, (3,2,2,1))
    print(gr)

    gr = np.zeros([3,2,2])
    gr = [[["x11","x12"],["y11","y12"]],[["x21","x22"],["y21","y22"]],[["x31","x32"],["y31","y32"]]]
    print(gr)
    gr = np.reshape(gr,(3,2,2,1))
    print(gr)
    
    print("old shape", X_train_gray.shape)
    #X_train_gray.reshape(len(X_train_gray), 32, 32, 1)
    np.reshape(X_train_gray, (len(X_train_gray), 32, 32, 1) )
    print("new shape", X_train_gray.shape)
    print("WHY DOES THIS VERSION _NOT_ WORK ?, WHEN THE OTHERS DO ?")
    assert(False == True)

#test_reshaping_3D_into_4D()


# In[24]:

from skimage import color

def get_grayscale_datasets_1channel(input_datasets):
    # eg input_datasets may be: [X_train, X_valid, X_test]
    dataset_labels  = ["training", "validation", "testing"]
    output_datasets = [[], [], []]
    
    print("\nconverting datasets to 1D grayscale")

    num_datasets = len(input_datasets)
    for s in range(num_datasets):

        #print("\nconverting " +  dataset_labels[s]  + " data to grayscale")
        # set output color depth to 1
        num_images, x_pixels, y_pixels, color_depth = input_datasets[s].shape
        color_depth = 1
        output_shape = (num_images, x_pixels, y_pixels, color_depth)
        
        output_datasets[s] = color.rgb2gray(input_datasets[s])
        
        # returned grayscale image is of shape==(num_images, x-pixels, y_pixels)
        # we need to reshape to add color_depth = 1
        #print(output_datasets[s].shape, "before reshape")
        np.reshape(output_datasets[s], [num_images, 32, 32, 1])
        #print(output_datasets[s].shape, "after_reshape")
        
        ## DRAT!! in earlier tests on dummy arrays, I was able to convert [a,b,c] into [a,b,c,1]
        ## DUNNO Why it's NOT working here!! Or why it DID work earlier.
        ## TODO: FIX, or cannot run lenet on grayscale-1D. Actually, probably cannot anyway! cuz of layer reshapes.

    #print("\ndone converting to grayscale\n")
    print(output_datasets[0].shape, output_datasets[1].shape, output_datasets[2].shape)
    
    return output_datasets

# test the above proceedure
gray_train, gray_valid, gray_test = get_grayscale_datasets_1channel([X_train_ORIG, X_valid_ORIG, X_test_ORIG])

fig = display_images([gray_train[1000]])

#fig.savefig("sample_grayscale-1channel_conversion.png")
print("figure saved as 'sample_grayscale-1channel_conversion.png'")


# In[25]:

# turn color data into grayscale image data
from skimage import color

def get_grayscale_datasets(input_datasets):
    X_train, X_valid, X_test = input_datasets
    
    print("converting to grayscale..")
    X_train_gray = color.rgb2gray(X_train)
    X_valid_gray = color.rgb2gray(X_valid)
    X_test_gray  = color.rgb2gray(X_test)
    
    gray_image_shape = X_train_gray.shape
    print(X_train_gray.shape,X_valid_gray.shape, X_test_gray.shape, "gray single channel conversion")
    assert (gray_image_shape[1:] == (32, 32)) #32px x 32px, 1 color channel: grayscale
    
    return [X_train_gray, X_valid_gray, X_test_gray]

# test get_grayscale_datasets
X_train_gray, X_valid_gray, X_test_gray = get_grayscale_datasets([X_train_ORIG, X_valid_ORIG, X_test_ORIG])

# display sample grayscale images from dataset
fig = display_images([X_train_gray[50], X_train_gray[500], X_train_gray[1000]])

#fig.savefig("sample_grayscale_conversions_single_channel.png")
print("figure saved as 'sample_grayscale_conversions_single_channel.png'")


# In[26]:

# turn grayscale into 3channel rgb grayscale
# (not ideal paramater-wise = duplicated data, but for shipping through my LeNet, it should remove shaping problems)

## Running rgb-Grayscales through leNet gave TERRIBLE Results !

def transform_grayscale_into_3D_grayscale(input_dataset, ratioR=1, ratioG=1, ratioB=1):
    X_train_gray, X_valid_gray, X_test_gray = input_dataset
    
    print("converted to 3 channel grayscale")
    # some ratios I've tried: 1/np.sqrt(3), 2*np.sqrt(3), 2/6 3/6 1/6
    #  sqrt(0.299 * R^2 + 0.587 * G^2 + 0.114 * B^2)

    # 3-channel gray looks different than 1-channel gray. Try to retain "look" of 1-channel gray. Find a multiplier ?
    R = X_train_gray * ratioR
    G = X_train_gray * ratioG
    B = X_train_gray * ratioB

    X_train_gray3D_2 = np.stack( (R, G, B), axis=-1)
    X_valid_gray3D_2 = np.stack( (R, G, B), axis=-1)
    X_test_gray3D_2  = np.stack( (R, G, B), axis=-1)
    print(X_train_gray3D_2.shape, "gray 3D shape")

    return [X_train_gray3D_2, X_valid_gray3D_2, X_test_gray3D_2]
    
## test above method

#input_dataset = get_grayscale_datasets([X_train, X_valid, X_test])
input_dataset = get_grayscale_datasets_1channel([X_train_ORIG, X_valid_ORIG, X_test_ORIG])

X_train_gray3D_2, X_valid_gray3D_2, X_test_gray3D_2 = transform_grayscale_into_3D_grayscale(input_dataset)

fig = display_images([X_train_gray3D_2[50], X_train_gray3D_2[500], X_train_gray3D_2[1000]])

#fig.savefig("sample_grayscale_1D_to_3D_conversion.png")
print("saved figure as 'sample_grayscale_1D_to_3D_conversion.png'")

print("\nresulting images are darker and lighter than the single channel grayscale \n some ratios I've tried: 1/np.sqrt(3), 2*np.sqrt(3), (R:2/6 G:3/6 B:1/6) \n How do I create a 3-channel grayscale that looks visually identical to the 1-channel grayscale?")


# In[27]:

# Try per channel zero centering. Find mean for each channel, where the mean for that channel is across all training images
# !! TERRIBLE RESULTS. tried a few learning_rates. NIX This technique !
from sklearn.preprocessing import normalize

def get_per_channel_mean_zero_centered_datasets(input_datasets):
    
    def separate_channels(images):
        # returns 2-D array: (num_examples, total_num_pixels)
        R = images[:,:,:,0].reshape(len(images),-1)
        G = images[:,:,:,1].reshape(len(images),-1)
        B = images[:,:,:,2].reshape(len(images),-1)
        return [R,G,B]

    def combine_channels(channels, output_shape):
        # used to re-combine separated RGB channels into 4-D array, where channels are the 4thD
        return np.stack(channels, axis=-1).reshape(output_shape)

    X_train, X_valid, X_test = input_datasets[0]
    
    initial_shape = X_train.shape
    R, G, B = separate_channels(X_train)

    r_pixels_mean = np.mean(R)
    g_pixels_mean = np.mean(G)
    b_pixels_mean = np.mean(B)

    # Must Save These Values, and apply to valid and test sets
    TRAINING_PIXELS_MEAN = [r_pixels_mean, g_pixels_mean, b_pixels_mean]
    print("TRAINING_PIXELS_MEAN\n", TRAINING_PIXELS_MEAN)


    #apply the same TRAINING_PIXELS_MEAN to: train, validation and test sets
    sets = [X_train, X_valid, X_test]
    zero_centered = [np.zeros(X_test.shape), np.zeros(X_valid.shape), np.zeros(X_test.shape)]

    for set in range(len(sets)):
        initial_shape = sets[set].shape
        R, G, B = separate_channels(sets[set])
        channels = [R, G, B]
        for c in range(len(channels)):
            channel = channels[c]
            channel = channel.astype(np.float64, copy=False)
            channel -= TRAINING_PIXELS_MEAN[c]
            channel = channel/TRAINING_PIXELS_MEAN[c]
        zero_centered[set] = combine_channels(channels, initial_shape)
    # end for loop

    print(zero_centered[0].shape, zero_centered[1].shape, zero_centered[2].shape)
    return zero_centered



# In[28]:

# Try per image zero centering. Find mean for each image, apply that mean to each channel in said image

def get_per_image_mean_centered_datasets(X_input_datasets):
    # ie datasets = [X_train, X_valid, X_test]
    
    X_output_datasets=[]

    for s in range(len(X_input_datasets)):
        setX = X_input_datasets[s]
        initial_shape = setX.shape
        num_images = initial_shape[0]

        # returns new copy with shape [num_images, num_pixels]
        X_output_dataset = setX.reshape(num_images,-1)
        num_pixels = X_output_dataset.shape[-1]
        
        # for accurate calculation of mean
        X_output_dataset.astype(np.float64, copy=False)

        # axis=1 averages all pixels in a single image; dtype=np.float64 for accuracy
        image_mean = np.mean(X_output_dataset, axis=1, dtype=np.float64)

        # copy/create matrix such that each image has num_pixels all set equal to the image's mean
        image_mean_xl = np.empty([num_images, num_pixels], dtype=np.float64)
 
        for i in range(num_images):
            imean = image_mean[i].astype(np.float64)
           
            # pure black image should not be in the dataset, but JustInCase.. divide by zero prevention
            if (imean == 0):
                # smallest non-zero positive number    #np.nextafter(0, 1)   #1e-20
                imean = np.nextafter(np.image_mean.dtype.type(0), np.image_mean.dtype.type(1))
            
            # set all pixel dimensions to the image's mean 
            image_mean_xl[i].fill(imean)

        
        # center the data (-1, 1) by subtracting and dividing the image by its mean
        X_output_dataset = (X_output_dataset - image_mean_xl) / image_mean_xl
        
        # restore to orig shape
        X_output_dataset = X_output_dataset.reshape(initial_shape)
        
        # add to list of returned datasets
        X_output_datasets.append(X_output_dataset)

    return X_output_datasets


# In[29]:

from sklearn.preprocessing import normalize
from skimage import color

# training on images processed via this method did not work

def get_normalized_images(image_sets):
    X_train = image_sets[0]
    X_valid = image_sets[1]
    X_test  = image_sets[2]
    
    # normalization
    print("normalizing")
    #for i in range(len(X_train))
    #X_train=[(i, tf.image.per_image_standardization(X_train[i])) for i in range(len(X_train))] 
    #list(map(lambda x: x**2, range(10)))
    #tf.image.per_image_standardization(X_valid)
    #tf.image.per_image_standardization(X_test)
    ##X_valid=[(i, tf.image.per_image_standardization(X_valid[i])) for i in range(len(X_valid))] 
    #X_test=[(i, tf.image.per_image_standardization(X_test[i]))  for i in range(len(X_test))]
    #X_valid = N_valid

    X_train_preprocessed = normalized(X_train)
    X_valid_preprocessed = normalized(X_valid)
    X_test_preprocessed  = normalized(X_test)

    print("normalized image:")
    x_1000_normalized = normalize(X_train_preprocessed[1000])
    imgplot = plt.imshow(x_1000_normalized)

    # to grayscale
    #Gray Scale could use cvtcolor from OpenCV:
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("to grayscale")
    X_train_preprocessed = color.rgb2gray(X_train_preprocessed)
    X_valid_preprocessed = color.rgb2gray(X_valid_preprocessed)
    X_test_preprocessed  = color.rgb2gray(X_test_preprocessed)
    print("grayscale done")
    #imgplot = plt.imshow(X_train[1000], cmap="gray")
    image_shape = X_train_preprocessed.get_shape().as_list()[1:]
    print(image_shape)
    assert (image_shape == [32, 32, 1])  #32px x 32px, 1 color channel: grayscale
    
    return [X_train_preprocessed, X_valid_preprocessed, X_test_preprocessed]


# In[30]:

# define training variables, constants

EPOCHS = 100
BATCH_SIZE = 128

def filter_size(in_size, out_size, stride):
    assert(padding == "VALID")
    return (in_size+1) - (out_size*stride)

def output_size(in_size, filter_size, stride):
    #(W−F+2P)/S+1
    assert(padding == "VALID")
    pad = 0
    return 1 + (in_size-filter_size+2*pad)/stride

padding = "VALID"
stride = 1
strides = [1, stride, stride, 1]
pool_stride = 2
pool_strides = [1, pool_stride, pool_stride, 1]
ksize = pool_strides


# ### Model Architecture

# In[31]:

### Define your architecture here.
### Feel free to use as many code cells as needed.


# In[32]:

"""
# This was for an attempt at running LeNet on 1-channel grayscale images. 
#    Given various filter shapes, calculate output shape
#    Unsuccessful.

def get_conv_layer_given_filter_shape(x, filter_shape):
    input_height,  input_width,  input_depth  = x.get_shape().as_list()[1:]
    filter_height, filter_width = filter_shape

    output_height = output_size(input_height, filter_height, stride)
    output_width  = output_size(input_width,  filter_width,  stride)
    output_depth  = input_depth
    print("output_size", x , output_height)

    weights_shape  = [filter_height, filter_width, input_depth, output_depth]
    bias_shape     = [output_depth]

    # initialize weights
    filter_weights = tf.Variable(tf.truncated_normal(weights_shape, mean=mu, stddev=sigma))
    filter_bias    = tf.Variable(tf.zeros(bias_shape))

    conv_layer = tf.nn.conv2d(x, filter_weights, strides, padding) + filter_bias
    print("conv output shape:", conv_layer.get_shape().as_list()[1:])

    # Activation
    conv_layer = tf.nn.relu(conv_layer)

    # Pooling (28,28,6?) --> (14,14,?6)
    input_height,  input_width,  input_depth  = conv_layer.get_shape()[1:]

    ksize = [1, 2, 2, 1]
    pool_strides = ksize
    conv_layer = tf.nn.max_pool(conv_layer, ksize, pool_strides, padding)
    print("pool_output_shape: ", conv_layer.get_shape().as_list()[1:] )

    return conv_layer
""" 
("")


# In[33]:

def get_conv_layer(x, conv_output_shape, pool_output_shape):
    input_height,  input_width,  input_depth  = x.get_shape().as_list()[1:]
    output_height, output_width, output_depth = conv_output_shape  #(28, 28, 6)

    filter_height = filter_size(input_height, output_height, stride)
    filter_width  = filter_size(input_width,  output_width,  stride)

    weights_shape  = [filter_height, filter_width, input_depth, output_depth]
    bias_shape     = [output_depth]

    # initialize weights
    filter_weights = tf.Variable(tf.truncated_normal(weights_shape, mean=mu, stddev=sigma))
    filter_bias    = tf.Variable(tf.zeros(bias_shape))

    conv_layer = tf.nn.conv2d(x, filter_weights, strides, padding) + filter_bias

    #print(conv_output_shape, "=?=", conv_layer.get_shape().as_list()[1:])
    assert( conv_output_shape == conv_layer.get_shape().as_list()[1:])

    # Activation
    conv_layer = tf.nn.relu(conv_layer)

    # Pooling (28,28,6?) --> (14,14,?6)
    input_height,  input_width,  input_depth  = conv_layer.get_shape()[1:]
    output_height, output_width, output_depth = pool_output_shape          #(14, 14, input_depth)

    ksize = [1, 2, 2, 1]
    pool_strides = ksize
    conv_layer = tf.nn.max_pool(conv_layer, ksize, pool_strides, padding)
    
    #print( pool_output_shape, "=?=", conv_layer.get_shape().as_list()[1:] )
    assert( pool_output_shape == conv_layer.get_shape().as_list()[1:] )

    return conv_layer


# In[34]:

def get_fcc_layer(prev_layer, output_length):
    input_length  = prev_layer.get_shape().as_list()[1]
    weights_shape = [input_length, output_length]
    bias_shape    = [output_length]
    #print(weights_shape, bias_shape, "get_fcc_layer: input_shape, output_shape")

    fcc_weights = tf.Variable(tf.truncated_normal(weights_shape, mean=mu, stddev=sigma))
    fcc_bias    = tf.Variable(tf.zeros(bias_shape))

    fcc_layer = tf.add(tf.matmul(prev_layer, fcc_weights), fcc_bias)
    assert([output_length] == fcc_layer.get_shape().as_list()[1:])

    return fcc_layer


# In[35]:

from tensorflow.contrib.layers import flatten

def LeNet(x):
    if x.get_shape()[-1] == 3:
        layer1 = get_conv_layer(x, [28,28,6], [14,14,6])
        layer2 = get_conv_layer(layer1, [10,10,16], [5,5,16])
    #elif x.get_shape()[-1] == 1:
    #    layer1 = get_conv_layer_given_filter_shape(     x, [3,3])
    #    layer2 = get_conv_layer_given_filter_shape(layer1, [3,3])
    else: print("error: images should be 1d or 3d")
    
    flattened = tf.contrib.layers.flatten(layer2)
    
    layer3 = get_fcc_layer(flattened, 120)
    layer3 = tf.nn.relu(layer3)
    layer3 = tf.nn.dropout(layer3, keep_probability)
    # maybe try dropout layer

    layer4 = get_fcc_layer(layer3, 84)
    layer4 = tf.nn.relu(layer4)
    # maybe try dropout layer
    layer4 = tf.nn.dropout(layer4, keep_probability)
    
    logits = get_fcc_layer(layer4, num_classes)
    assert( [logits.get_shape().as_list()[1] ] == [num_classes])
        
    return logits


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[36]:

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


# In[37]:

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle
import tensorflow as tf

# images as imported are already sized properly for leNet at (32x32)
assert ((X_train_ORIG.shape[1],X_train_ORIG.shape[2]) == (32, 32))  #32px x 32px, 3 color channels:RGB

# shuffle data
X_train_SHUFFLED, y_train_SHUFFLED = shuffle(X_train_ORIG, y_train_ORIG)
X_valid_SHUFFLED, y_valid_SHUFFLED = shuffle(X_valid_ORIG, y_valid_ORIG)
    # don't need to shuffle test data


# In[38]:

## initialize

input_datasets = [X_train_SHUFFLED, X_valid_SHUFFLED, X_test_ORIG]

# choose which pre-processor to work with:

# X_train, X_valid, X_test = get_grayscale_datasets_1channel(input_dataset)              # must convert to 3D to use LeNet
# X_train, X_valid, X_test = get_grayscale_datasets(input_dataset)                       # must convert to 3D to use LeNet
# X_train, X_valid, X_test = transform_grayscale_into_3D_grayscale(get_grayscale_datasets(input_dataset)) # Got TERRIBLE RESULTS
# X_train, X_valid, X_test = get_per_channel_mean_zero_centered_datasets(input_dataset)  # Got TERRIBLE RESULTS
X_train, X_valid, X_test = get_per_image_mean_centered_datasets(input_datasets)


# labels do not get pre-processed
y_train, y_valid, y_test = [y_train_SHUFFLED, y_valid_SHUFFLED, y_test_ORIG]


# decide on a set training paramaters:
mu = 0
sigma = 0.01  #0.1   #.018  #.018 = 1.0/np.sqrt(pixels_x * pixels_y * color_depth) = 1/sqrt(32*32*3) = 1/55 = .018;  bw: 1/sqrt(32*32*1) = 1/32 = 0.03125
learning_rate = .001 #0.01


# determine placeholder paramater values based on chosen (preprocessor) # (dataset in grayscale, or rgb format)
image_shape = X_train.shape
if len(image_shape) == 3:
    # bw image needs re-shaped to add a color depth of 1
    print("oops, bw datasets need to be reshaped to add a color depth of 1")
assert( len(image_shape) == 4 )
pixels_x, pixels_y, color_depth = image_shape[1:]

# probability of saving the node: Training Set: 0.5 (DROPOUT_ON  = 0.5)
#           NO DROPOUT ON: Valid, Test Sets!! : 1.0 (DROPOUT_OFF = 1.0)
DROPOUT_OFF = 1.0  #(dropout_keep_probability == 1.0 : keep everything)
DROPOUT_ON  = 0.5  #(dropout_keep_probability == 0.5 : randomly set half the nodes weights to zero)


# initialize tf training variables !!
# features, labels
x = tf.placeholder(tf.float32, (None, pixels_x, pixels_y, color_depth))
y = tf.placeholder(tf.int64,   (None))

# using dropout on training set at fcc_3 and fcc_4, not on validation loss calculation, or on test set.
keep_probability = tf.placeholder(tf.float32)


# run leNet
logits = LeNet(x)

# loss
#   tf.nn.sparse_softmax_cross_entropy_with_logits combines:
#   1) softmax with 2) cross_entropy and 3)(sparse version) performs one-hot encoding to the labels, y
cross_entropy  = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)  #labels
loss_operation = tf.reduce_mean(cross_entropy)

# train
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)

# accuracy
model_prediction = tf.argmax(logits, 1)
prediction_is_correct = tf.equal(model_prediction, y)#labels)
accuracy_calculation  = tf.reduce_mean(tf.cast(prediction_is_correct, tf.float32))

# save batch loss and accuracy results to visually plot how the model performed
training_stats = []


# In[39]:

# evaluation routine
def evaluate_data(X_data, y_data):
    print("evaluating..")
    sess = tf.get_default_session()
    total_loss = 0
    total_accuracy = 0
    
    num_samples = len(X_data)
    for batch_start in range(0, num_samples, BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        X_batch = X_data[batch_start:batch_end]
        y_batch = y_data[batch_start:batch_end]
        
        accuracy, loss = sess.run([accuracy_calculation, loss_operation],
                                  feed_dict = {x:X_batch, y:y_batch, keep_probability: DROPOUT_OFF})
        
        this_batch_size = len(X_batch)
        
        total_accuracy += this_batch_size * accuracy
        total_loss     += this_batch_size * loss
        
    total_accuracy = total_accuracy / num_samples
    total_loss = total_loss / num_samples
        
    return total_accuracy, total_loss     


# 

# In[68]:

from IPython import display

tempx=np.asarray([])
tempy=np.asarray([])
for i in range(10):
    tempx = np.append(tempx, i)
    tempy = np.append(tempy, i**2)
    plt.gca().cla() 
    plt.plot(tempx,tempy,label='test')
    plt.legend()
    display.clear_output(wait=True)
    display.display(plt.gcf()) 
    time.sleep(0.5) 
    print(tempx, tempy)


# In[52]:

## SAMPLE DYNAMIC PLOT
# http://stackoverflow.com/a/34486703/5411817

# for dynamic plots in jupyter notebook
get_ipython().magic('matplotlib notebook')

import numpy as np
import matplotlib.pyplot as plt
import time

def pltsin(ax, colors=['b']):
    x = np.linspace(0,1,100)
    if ax.lines:
        for line in ax.lines:
            line.set_xdata(x)
            y = np.random.random(size=(100,1))
            line.set_ydata(y)
    else:
        for color in colors:
            y = np.random.random(size=(100,1))
            ax.plot(x, y, color)
    fig.canvas.draw()

fig,ax = plt.subplots(1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
for f in range(5):
    pltsin(ax, ['b', 'r'])
    time.sleep(1)

# return to state the remaining notebook cells expect
get_ipython().magic('matplotlib inline')


# In[56]:

# TEMP TRUNCATE DATA FOR Alpha TESTING the code

print("hello")
# truncate the training set to be just a bit larger than the BATCH_SIZE (so run at least 2 batches per epoch)
tr = int(BATCH_SIZE * 1.2)
# truncate validation (and training??) set to each be about 15% of the training set size
va = te = int(tr * 0.2)

print(tr, va, te, "total:", (tr+va+te), "percent training: ", tr/(tr+va+te))
X_train = X_train[0:tr]
y_train = y_train[0:tr]
X_valid = X_valid[0:va]
y_valid = y_valid[0:va]
X_test  = X_test[0:te]
y_test  = y_test[0:te]
print('DATA TRUNCATED TO:', len(X_train), "SAMPLES for preliminary testing")
 

EPOCHS = 4
print('EPOCHS TRUNCATED TO:', EPOCHS, "EPOCHS for preliminary testing")
#


# In[60]:

##ATTEMPT TO WRITE ROUTINE FOR DYNAMIC UPDATING THE PLOT

# for displaying a legend
import matplotlib.patches as mpatches

# Initializations for Dynamic Plot figures
vloss, tloss, vaccu, taccu = [[],[],[],[]]
epoch_x_axis = range(1, EPOCHS+1)

# figure size in inches: width, height    
fig = plt.figure(1, figsize=(7, 7))

# to display legend
blue_patch  = mpatches.Patch(color='blue',  label='Validation Set')
red_patch   = mpatches.Patch(color='black', label='Training Set')
black_patch = mpatches.Patch(color='red',   label='Minimum 93.00% Validation Accuracy Required')

plt.subplot(311, title = "Loss vs Epoch")
#plt.plot(epoch_x_axis, vloss, 'b', epoch_x_axis, tloss, 'k')

#     plt.subplot(313, title="% Accuracy vs Epoch")
plt.subplot(312, title="% Accuracy vs Epoch")
req_accuracy = 0.9300
#     plt.plot(epoch_x_axis, vaccu, 'b', epoch_x_axis, taccu, 'k')
#     plt.axhline(req_accuracy, color='r')

# overlay legend on "Accuracy" (the most spacious) subplot
plt.legend(handles=[blue_patch, black_patch, red_patch])

#     # zoomed in accuracy plot, highlighting variance around req_accuracy
#     plt.subplot(312, title="% Accuracy vs Epoch, zoomed in ")
#     plt.plot(epoch_x_axis, vaccu, 'b', epoch_x_axis, taccu, 'k')
#     plt.axhline(req_accuracy, color='r')
#     plt.ylim((.9000, 1.0100))

# prevent overlapping of labels with subplots
plt.tight_layout()
plt.show()
    
def update_plot(plt, epoch_x_axis, vloss, tloss, vaccu, taccu):
    #update plots at each Epoch
    plt.subplot(311, title = "Loss vs Epoch")
    plt.plot(epoch_x_axis, vloss, 'b', epoch_x_axis, tloss, 'k')

    plt.subplot(312, title="% Accuracy vs Epoch")
    plt.plot(epoch_x_axis, vaccu, 'b', epoch_x_axis, taccu, 'k')
    plt.axhline(req_accuracy, color='r')

    
    #plt.show()
   
    
# train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    training_stats = []
    
    print("Training...\n")
    tstart = time.time()
    for i in range(EPOCHS):
        print("EPOCH: ", i+1, "of", EPOCHS, "EPOCHS")
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for batch_start in range(0, num_examples, BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            features = X_train[batch_start:batch_end]
            labels   = y_train[batch_start:batch_end]
            #train
            sess.run(training_operation, feed_dict = {x:features, y:labels, keep_probability: DROPOUT_ON})
#             if batch_start % 100 == 0:
#                 print("        batch ", 1+batch_start//BATCH_SIZE, "of ", 1 + int(num_examples/BATCH_SIZE))#, "batches,  on EPOCH", i+1, "of", EPOCHS, "EPOCHS")
                      
        # evaluate on validation set, and print results of model from this EPOCH
        print(X_valid.shape)
        validation_accuracy, validation_loss = evaluate_data(X_valid, y_valid)
        training_accuracy,   training_loss = evaluate_data(X_train, y_train)
        
#         # TODO: would be awesome to display live charts of these results, rather than this text output 
#         #      (see charts in next cell)
#         print("Time: {:.3f} minutes".format(float( (time.time()-t0) / 60 )))
#         print("Validation Loss = {:.3f}".format(validation_loss))
#         print(" (Training Loss = {:.3f})".format(training_loss))
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#         print(" (Training Accuracy = {:.3f})".format(training_accuracy))
#         print()
        
        # round to nearest even number at 4th decimal place
        #training_stats.append([np.around(validation_loss,4), np.around(training_loss,4), np.around(validation_accuracy,4), np.around(training_accuracy,4)])
        training_stats.append([validation_loss, training_loss, validation_accuracy, training_accuracy])
        np.savetxt('./training_stats/training_stats.tmp.txt', training_stats)
        
        # dynamically plot training_stats
        #update_plot(i, validation_loss, training_loss, validation_accuracy, training_accuracy)
        plt.subplot(311, title = "Loss vs Epoch")
        plt.plot(epoch_x_axis, vloss, 'b', epoch_x_axis, tloss, 'k')

        plt.subplot(312, title="% Accuracy vs Epoch")
        plt.plot(epoch_x_axis, vaccu, 'b', epoch_x_axis, taccu, 'k')
        plt.axhline(req_accuracy, color='r')

    
    
#     tend = time.time()
#     print("\nElapsed Training Time: {:.3f} minutes".format(float( (time.time()-tstart) / 60 )))
    
#     # use current time stamp as id for model and training_stats filenames
#     model_timestamp = time.strftime("%y%m%d_%H%M")
    
#     # save training_stats
#     filename = './training_stats/training_stats_' + model_timestamp + '.txt'
#     np.savetxt(filename, training_stats)
#     print("\ntraining_stats Saved As: ", filename, "\n")    

#     # save trained model
#     print("Saving model..")
#     saver = tf.train.Saver()
#     saver.save(sess, './trained_models/sh_trained_traffic_sign_classifier_' + model_timestamp)
#     print("Model Saved")
#     print()

    
    
#plt.show()
# return to state the remaining notebook cells expect
#% matplotlib inline


# In[43]:

import time

#assert("no, Don't Retrain Model !"  ==  
#        "Re-Running cells for later sections of the notebook. AND Want To Use The already TRAINED NETWORK"
#       )

# train our model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    training_stats = []
    
    print("Training...\n")
    tstart = time.time()
    for i in range(EPOCHS):
        print("EPOCH: ", i+1, "of", EPOCHS, "EPOCHS")
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for batch_start in range(0, num_examples, BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            features = X_train[batch_start:batch_end]
            labels   = y_train[batch_start:batch_end]
            #train
            sess.run(training_operation, feed_dict = {x:features, y:labels, keep_probability: DROPOUT_ON})
            if batch_start % 100 == 0:
                print("        batch ", 1+batch_start//BATCH_SIZE, "of ", 1 + int(num_examples/BATCH_SIZE))#, "batches,  on EPOCH", i+1, "of", EPOCHS, "EPOCHS")
                      
        # evaluate on validation set, and print results of model from this EPOCH
        print(X_valid.shape)
        validation_accuracy, validation_loss = evaluate_data(X_valid, y_valid)
        training_accuracy,   training_loss = evaluate_data(X_train, y_train)
        
        # TODO: would be awesome to display live charts of these results, rather than this text output 
        #      (see charts in next cell)
        print("Time: {:.3f} minutes".format(float( (time.time()-t0) / 60 )))
        print("Validation Loss = {:.3f}".format(validation_loss))
        print(" (Training Loss = {:.3f})".format(training_loss))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print(" (Training Accuracy = {:.3f})".format(training_accuracy))
        print()
        
        # round to nearest even number at 4th decimal place
        #training_stats.append([np.around(validation_loss,4), np.around(training_loss,4), np.around(validation_accuracy,4), np.around(training_accuracy,4)])
        training_stats.append([validation_loss, training_loss, validation_accuracy, training_accuracy])
        np.savetxt('./training_stats/training_stats.tmp.txt', training_stats)
        
    tend = time.time()
    print("\nElapsed Training Time: {:.3f} minutes".format(float( (time.time()-tstart) / 60 )))
    
    model_timestamp = time.strftime("%y%m%d_%H%M")
    filename = './training_stats/training_stats_' + model_timestamp + '.txt'
    np.savetxt(filename, training_stats)
    print("\ntraining_stats Saved As: ", filename, "\n")    

    # save trained model
    print("Saving model..")
    saver = tf.train.Saver()
    saver.save(sess, './trained_models/sh_trained_traffic_sign_classifier_' + model_timestamp)
    print("Model Saved")
    print()

    if validation_accuracy >= 0.93:
        print(" !! Congratulations!! Your model meets the minimum required Validation Accuracy of 0.93")
        print("   You may now run your model on the Test Set :-)")
    else:
        print("KEEP WORKING ON YOUR MODEL to acheive a minimum Validation Accuracy of 0.93")
        print("underfitting if:  low accuracy on training and low accuracy on validation sets.")
        print("overfitting  if: high accuracy on training but low accuracy on validation sets.")

    print()


# In[47]:

with tf.Session() as sess:
 
    # save trained model
    print("Saving model..")
    saver = tf.train.Saver()
    saver.save(sess, './trained_models/sh_trained_traffic_sign_classifier_' + model_timestamp)
    print("Model Saved")
    print()

    if validation_accuracy >= 0.93:
        print(" !! Congratulations!! Your model meets the minimum required Validation Accuracy of 0.93")
        print("   You may now run your model on the Test Set :-)")
    else:
        print("KEEP WORKING ON YOUR MODEL to acheive a minimum Validation Accuracy of 0.93")
        print("underfitting if:  low accuracy on training and low accuracy on validation sets.")
        print("overfitting  if: high accuracy on training but low accuracy on validation sets.")

    print()


# In[51]:

# assert("THEN no need to display empty plot" == 
#        "IF not retraining right now - just resetting the kernal (and running all except training and plotting cells)"
#       )

# for displaying a legend
import matplotlib.patches as mpatches

def read_training_stats_from_file():
    # IF read a model from disk, MUST set model_timestamp associated with the filename !! 
    # else will run into error when saving the figure
    
    assert("did you update" == "model_timestamp??")
    # Update model_timestamp to Read the training_stats from disk
    model_timestamp = '170406_2032'
    
    training_stats_filename = './training_stats/training_stats_' + model_timestamp + '.txt'
    training_stats_read_from_disk = np.loadtxt(training_stats_filename)
    print(' read training_stats from ', training_stats_filename)
    
    return [model_timestamp, training_stats_read_from_disk]

# Uncomment, and update "model_timestamp" Inside the function, to Read the training_stats from disk
#model_timestamp, training_stats = read_training_stats_from_file()


num_epochs = len(training_stats)
vloss, tloss, vaccu, taccu = [[],[],[],[]]
epoch_x_axis = range(1, num_epochs+1)
for i in range(len(training_stats)):
    vloss.append(training_stats[i][0])
    tloss.append(training_stats[i][1])
    vaccu.append(training_stats[i][2])
    taccu.append(training_stats[i][3])

# figure size in inches: width, height    
fig = plt.figure(1, figsize=(7, 7))

# to display legend
blue_patch  = mpatches.Patch(color='blue',  label='Validation Set')
red_patch   = mpatches.Patch(color='black', label='Training Set')
black_patch = mpatches.Patch(color='red',   label='Minimum 93.00% Validation Accuracy Required')

plt.subplot(311, title = "Loss vs Epoch")
plt.plot(epoch_x_axis, vloss, 'b', epoch_x_axis, tloss, 'k')

plt.subplot(313, title="% Accuracy vs Epoch")
req_accuracy = 0.9300
plt.plot(epoch_x_axis, vaccu, 'b', epoch_x_axis, taccu, 'k')
plt.axhline(req_accuracy, color='r')

# overlay legend on "Accuracy" (the most spacious) subplot
plt.legend(handles=[blue_patch, black_patch, red_patch])

# zoomed in accuracy plot, highlighting variance around req_accuracy
plt.subplot(312, title="% Accuracy vs Epoch, zoomed in ")
plt.plot(epoch_x_axis, vaccu, 'b', epoch_x_axis, taccu, 'k')
plt.axhline(req_accuracy, color='r')
plt.ylim((.9000, 1.0100))

# prevent overlapping of labels with subplots
plt.tight_layout()
plt.show()


# model_timestamp for figure should match the timestamp from the model's file, not the current timestamp (see prev cell and top of this cell)
filename = './training_stats/training_stats_' + model_timestamp + '.png'

fig.savefig(filename)  # results in 175x175 px image
print("Figure saved as " + filename + "\n")


# In[ ]:

## Compare Models
# load figure training_stats_plotted-170327_1518.png
# Model Architecture:
#  - Lenet5: no dropout, no augmentantion, 
#  - Pre-proccessing: per image mean centered (-1,1) (not standardized though)
# Comment on figure/training/model:
#  Achieved required accuracy, but..
#  -- the loss chart shows that our model is overfitting to the training data:
#  -- as the training achieves 0% loss, the loss on validation set instead increases
#  Two methods to reduce overfitting are:
#  - add dropout layers: 
#  -- after the first fcc_layer, or after the firtst and second fcc_layers.
#  -- good dropout value is generally .5 (keep_prob = 0.5)
#  - augment the training data:
#  -- on each batch, add a random rotation, zoom, color-cast/brightness, shift image up or down, etc to the entire batch
#  Also, it seems that the learning rate could be decreased partway through training
#  - not too bad on this model (was Very apparent on another model)
#  -- as the accuracy seems to level out, but Oscillate once it's leveled out,
#  -- I wonder if lowering the training rate at that point would be useful.
#  -- maybe something that monitors the  accuracy, and notices once it's rather flat, but oscillates, then it can 
#  -- automatically decrease the learning rate by some set amount. Perhaps divide by 10 ?
#  -- Dunno, But this oscillation appears to me to be a symptom of yaking too large a step. Great at first, but then
#  -- as it hones in to a minimum, decreasing the step size may enable it to land at said assumed minimum. ??
#  -- This could be a good experiment to try.

## Add Dropouts after fcc_3 and fcc_4, with keep_probability = 0.5
# load figure training_stats_plotted-??????????.png
# Model Architecture:
#  - Lenet5: added dropout to previous architecture. No augmentantion, 
#  - Pre-proccessing (same as previous model): per image mean centered (-1,1) (not standardized though)
# Results
#  - Overall Better, I think..
# Training Model did not reach 0.000 loss or 100% accuracy, though it got very close. 
# -- The Previous model definitely saturated out, perfectly (overfitting)
#  - It took longer to train -"bumpy" it oscillates more frequently, however it is also a steadier average;
#  -- ie the oscillations are shallower. In this way, it seems slightly more stable
#  - Total validation loss is lower !
#  - Validation Accuracy is about the same. It might not reach quite as high a max, but again, the oscillations are shallower.
#  - Training Accuracy is slightly lower; still greater than 99% - It didn't reach perfect fit in 100 Epochs. Perhaps eventually it would ?
#  - The difference between Training and Validation loss is less; the graphs are closer.
# I'm not convinced that lowering the learning rate would be as helpful, as it looked in the previous model.
# I probably will not run that experiment on either model.
# Well, if I did, I'd probably do something like, when Validation Accuracy is 93% or 95%, divide learning rate by 10
# That's quick and dirty: using knowledge from this pre3vious graph to inform a future model.
# Ideally, it'd mathematically be programmed in. But that's more work than it's worth for this project.
# The interest here, is to see if it's something that would even make sense to do - does it affect the model or no?

# Would like to see how Augmentation would affect the training graphs. This model performs well enough, however to move on.
#   It's more important to complete this project right now.  Perhaps Augmentation (or even inception) can be done at another time


## Add randomized Augmentation to each batch (generate randomize settings, apply that settting to entire batch of images)
# load figure training_stats_plotted-??????????.png
# Model Architecture:
#  - Lenet5: added augmentation to previous architecture.
#  - Pre-proccessing (same as previous model): per image mean centered (-1,1) (not standardized though)
# 


# In[ ]:

## STOP !! Do NOT Proceed Until Model is FINISHED and has Validation >= 93%

    # underfitting if:  low accuracy on training and validation sets.
    # overfitting  if: high accuracy on training but low accuracy on validation sets.

assert (validation_accuracy >= 0.9300)
#assert ('yes' == 'no')


# In[27]:

# test the trained model
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('./trained_models/.'))

    test_accuracy, test_loss = evaluate_data(X_test, y_test)
    print(test_accuracy)
    print("Test Loss     = {:.3f}".format(test_loss))
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
#assert ('yes' == 'no')


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[28]:

#import numpy as np
#import tensorflow as ts


# In[29]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.

# SH TODO: Load Saved Model
# # save trained model
# print("Saving model..")
# saver = tf.train.Saver()
# saver.save(sess, './sh_trained_traffic_sign_classifier')
# print("Model Saved")
# print()

#saver.restore(sess, './sh_trained_traffic_sign_classifier-170327_1518')

# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.')) 

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./trained_models/.'))



# In[30]:

import numpy as np
import glob
from scipy import misc

paths = ['traffic_signs_from_web/32x32x3/1_straightforward_IN_signnames/*.jpg',
         'traffic_signs_from_web/32x32x3/2_tricky_and_NOT_in_signnames/*.jpg',
         'traffic_signs_from_web/32x32x3/3_difficult_NOT_in_signnames/*.jpg',
        ]
dataset_descriptions = ['Set 1: Straightforward: \n        Expect Good Matches.', 
                        'Set 2: These signs are Not part of the signnames.csv, but look Similar to Signs that are. \n        Curious if it picks signs that I would choose',
                        'set 3: These signs are NOT in signnames, and are NOT similar to any signs that are. \n        Mostly this set will generate rubish. \n        They may or may not provide interesting insights on the nn.\n'
                       ]

num_datasets = len(paths)
web_datasets_ORIG = []
for s in range(num_datasets):
    new_dataset = []
    for image_path in glob.glob(paths[s]):
        new_dataset.append(misc.imread(image_path))    

    new_dataset = np.asarray(new_dataset)
       
    print("\n", dataset_descriptions[s])
    print('      ', new_dataset.shape)
    fig = display_images(new_dataset)
    
    filename = './sample_signs_from_web_Set_' + str(s+1) + '.png'
    #fig.savefig(filename)
    print('   figure saved as', filename, '\n')
    
    web_datasets_ORIG.append(new_dataset)
    
print('\nImporting done...', len(web_datasets_ORIG), "sets of traffic sign images from the web")


# 

# In[31]:

# pre-process images

assert (len(web_datasets_ORIG) == 3)

# Would be better If I did not hard code 3 storage sets.
# (ie X_sets = get...) and change subsequent code to reference:
# X_sets[0], X_sets[1], X_sets[2], ..
# ... it is requiring me to supply exactly 3 sets of data in the above cell :-/

X_set1, X_set2, X_set3 = get_per_image_mean_centered_datasets(web_datasets_ORIG)



# ### Predict the Sign Type for Each Image

# In[33]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

# create np.array of signnames from our csv file. throw "index" away, only want the sign_names
index, sign_names = np.genfromtxt('./signnames.csv', unpack=True, dtype=np.str_, delimiter=',', skip_header=1)
#print(csv)


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./trained_models/.'))
    
    predictions = tf.argmax(logits, 1)
       
    # eval extracts the values from the tensor, and returns them in the form of an np.array, so we can use them
    #   ie cannot use predictions (tensors) directly (except as tensors inside tf)
    signs_set1 = predictions.eval(feed_dict = {x: X_set1, keep_probability: DROPOUT_OFF}) 
    signs_set2 = predictions.eval(feed_dict = {x: X_set2, keep_probability: DROPOUT_OFF}) 

set_num = 0
for set in [signs_set1, signs_set2]:
    display_images(web_datasets_ORIG[set_num])
    for sign in set:        
        print(sign_names[sign])
    set_num += 1
    print('\n')



# ### Analyze Performance
# 

# #### TODO:
# #####  NOTE: I added additional crops of problem images to see how they would compare.
# #####  I also deleted an image that was a useless poor image choice to begin with
# #####  I did NOT Update My Analysis or Accuracy reporting to reflect the changes in this dataset.
# 
# #### Set_1 My classifier performed terrible on the images I expected good results on.  
# Signs 1, 6: To be fair, I could have cropped these better. 
# 
# Sign 6: I expect should get a correct answer with closer cropping.  
#     Then again, I was overly optimistic that it would do well even given the bad crop I handed it.  
#     I can imagine that the crop I gave it lent a partial figure-8 shape to confuse it, though at the wrong scale??  
#     
# Sign 2: Was in black an white, so it is missing color information, which may have otherwise helped nudge it to the correct answer.  
# - Additionally, the sign was skewed at an angle, and placed against another sign,  
# - so it's overall shape could be difficult to discern.  
# - Unfortunately, the misinterpretation is Grave:   
#   - 'No Entry' sign became an "End All Speed and Passing Limits"  
#   - the exact Opposite of the intended !  
#         
# Sign 3:  No Idea what happened there! Perhaps the edges of the triangle resembled a 5, though at the wrong scale??  
#     If so this is similar to what I "imagine" threw it off for image 6. Or I just have an imagination.  
#     
# Sign 4 and 5, Fortunately, It got these Correct !!  
# 
# So this classifier was correct on 2/6 images, or 33% correct.  
# That's Far lower than the test, and validation sets!!  It's downright Terrible.   
# 
# #### Set_2 consisted of street signs that were not part of the street_sign names our classifier was trained on.  It was impossible for my classifier to get any of these correct.
# - Indeed, the correct sign names do not even exist in the csv signnames file,   
# - I was simply curious how it would interpret them: would see the same type similarities **I** see when looking at them?  
# - Would it make the same choices I did (not knowing German signs) ?  
#     
# Sign 2: Surprisingly, it got almost about as close a guess as possible.   Must be beginner's luck.
# 
# Sign 5: Was also surprisingly good guess. 
# 
# Signs 3, 6, 7: These three signs I Expected the classifier to, no-brainer, return specific predictions.  
#     Nope. It passed over images which I think they look Very similar to, 
#     in favor of images that I think look not at all similar to these signs.   What was its "thinking" process ??  
#     
# Signs 1, 4: I expected rubbish responses to these rubbish images. Never-the-less, I see no resemblence to what it _did_ choose.  
#     I guess NaN was not an option?  

# In[34]:

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

## TODO:
#  NOTE: I added additional crops of problem images to see how they would compare.
#  I also deleted an image that was a useless poor image choice to begin with
#  I did NOT Update My Analysis or Accuracy reporting to reflect the changes in this dataset.

print("Classifier was ", 100*2//6, "% accurate on the new images, getting 2 of 6 correct")


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[35]:

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./trained_models/.'))   
    
    probabilities = tf.nn.softmax(logits).eval(feed_dict={x : X_set1, keep_probability : DROPOUT_OFF})
    set_index = 0  # corresponds to the index in list: web_datasets_ORIG, used for displaying the correct images

    num_images = len(probabilities)
    num_guesses = 5
    
    # get top 5 probabilities, and class_id for each image
    top_5_probs, top_5_class_ids = sess.run(tf.nn.top_k(tf.constant(probabilities), k=num_guesses))
    
for i in range(num_images):
    display_images([web_datasets_ORIG[set_index][i]])
    for guess in range(num_guesses):
        percentage = top_5_probs[i][guess] * 10000//1 / 100
        sign_name  = sign_names[top_5_class_ids[i][guess]]
        print('{0:7.2f}%'.format(percentage), sign_name)
    print("\n")


 


# In[36]:

"""
- Interesting how cropping changed the predictions.   
 -- Good Cropping matters. (Giving my trained model, anyhow)
 -- Stop Sign: better crop (tho not great), stop was rated 3rd on the list, but confidence was lower. worse crop, was > 10 x more confident in "stop" as a choice, though it was now 5th. Of course, this is also reflected in that the worse crop lowered it's confidence in anything overall. Better crop gave it a 99% certainty in a wrong answer, vs a 55% top certainty in the close cropped version. 
- I'm surprised that the 100km/h speed limit did not even make the list, even with good cropping.  
- How did the second to last image have 2 predictions at less than 0 % ??  
 --Is this a red-flag that something is wrong (with alogrithm) ??  Or that rounding hit an overflow ??  
"""
("")


# In[37]:

# This is total duplication of above cell. (Not DRY)
# EXCEPT: runing it on _X_set2, set_index==1 instead
#..out of CURIOSITY. 


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./trained_models/.'))   
    
    probabilities = tf.nn.softmax(logits).eval(feed_dict={x : X_set2, keep_probability : DROPOUT_OFF})
    set_index = 1

    num_images = len(probabilities)
    num_guesses = 5
    
    # get top 5 probabilities, and class_id for each image in form of numpy arrays
    top_5_probs, top_5_class_ids = sess.run(tf.nn.top_k(tf.constant(probabilities), k=num_guesses))
    
for i in range(num_images):
    display_images([web_datasets_ORIG[set_index][i]])
    for guess in range(num_guesses):
        percentage = top_5_probs[i][guess] * 10000//1 / 100
        sign_name  = sign_names[top_5_class_ids[i][guess]]
        print('{0:7.2f}%'.format(percentage), sign_name)
    print("\n")


 


# In[38]:

# Second set of Images
"""
Interesting results on this dataset,  
  so I'll leave the cell intact
Remember none of these images were in the training set; the correct answer is not in the list of labels provided.  
So it was impossible for our classifier to get these correct (except, sort of, on the first image..)  
- It's interesting to me that on the 1st image, it did manage to locate the "sub-sign" within it, as 5th prob'  
- The 2nd image also chose what I consider the closest two answers as it's top two  
-- though the difference in confidence is vastly different, and perhaps swapped from what might be expected.  
- The 3rd, 5th,6th images do NOT focus on the Number depicted, which is what *I* do when interpolating their meaning.
--  It surprises me that it does not choose 
- The 4th image, however, does seem to consider the number, and the "not / end-of" in it's top two choices
"""
("")


# ---
# 
# ## Step 4: Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[ ]:

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


# In[ ]:

## images of interest:
# set_1, image2:  _17-no-entry-1-bw.jpg  
#                RE:  the meaning of the Real Image is Directly Opposite 
#                      in meaning of the Predicted Image !! Dire Consequences ensue !!
# set_1, image10  
#                RE:  ?? How did it NOT get this one Correct??  
#                     Were there enough 100km/h in the dataset ??
# set_2: image 2
#                RE:  why 30km/h was Highly Favored over 20km/h
# set_2: image 3, 6, or 7
#                RE:  why it did not focus on the Number, like my eyes do
# set_1; image 3 vs image 4 or 5
#                RE:   crop made a dramatic difference in results


# with tf.Session() as sess:
    
#     image = X_set1[1]   
#     #layer = layer4

#     outputFeatureMap(image, fcc_layer)


# Hmm.. Unfortunately, I don't think I can use the function above to gain insight on training features.
# Not only are my tensorflow training variables are encapsulated inside a LeNet(x) function.
# So the tensor I need to pass into the outputFeatureMap function are not global variables. 
# I have no access or handle to them from here, or anywhere outside that function.
# 
# If it is indeed possible to access the required variable, I would be interested in gaining insight, for about 4 images.
# That is not going to happpen at this time, however.  

# ### Question 9
# 
# Discuss how you used the visual output of your trained network's feature maps to show that it had learned to look for interesting characteristics in traffic sign images
# 

# **Answer:**

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 
