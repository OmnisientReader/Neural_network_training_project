from Network import Network
from Validation import Validation
from Training import Training
from Images_data import ImagesData


# data_path variable is an actual path to your file with images of faces
# your file must be .zip file
# this can be changed to the .tar file, look into ImagesData module to change

data_path = 'YOUR_DATA_PATH.zip'   

# mask_path variable is an actual path to your file with images of non-faces
# your file must be .zip file
# this can be changed to the .tar file, look into ImagesData module to change

mask_path = 'YOUR_MASK_PATH.zip'

# writefiles_path is 1-dimension array of 2 paths: path for weghts and path for biases
# this variable is used to write or read parameters from the file using the
# write_3d_array_to_file an read_3d_array_to_file methods from the Network module
# look into Network for information

writefiles_path = ['weights_file.txt','biases_file.txt'] 

# number_of_photos - number of images for training

number_of_photos = 2000

# network = object of Network class
# input_parameters are: network structure parameters, data_path, mask_path,
# learning rate, minibatch size, number of photos, writefiles_path
# look into Network for more information  

network = Network([1],data_path,mask_path, 0.0000000001, 10, 1, number_of_photos, writefiles_path)

# image = object of ImagesData class
# input_parameters are: data_path, mask_path,
# initial position, minibatch size, parameter to set if first image is face or not(True, False), number_of_photos
# look into ImagesData for more information  

image = ImagesData(data_path, mask_path, 0, 1, 1, number_of_photos)

# model - object of Training class
# naturally, all training happens here
# input_parameters are: network and image(look above)
# look into Training for more information  

model = Training(network, image)

model.start_network_training()

# valid - object of Validation class
# input_parameters are: network and image(look above), number of images to go through validation, parameter between 0.0 and 1.0 
# that is responsible for decision of classifying images

valid = Validation(network, image, 500, 0.5)


"""

NETWORK RESTRAINTS AND TRAINING RESULTS

this project is a training project, that's why it may look very rigid.
well, it really looks like this to me:

--------
architecture is not scalable and there is no protection against the fool 

images should be >=(250*250) pixels for extracting good features 

parameters for features are CONSTANT, that is, to change you have to 
look into ImagesData and find needed method

network doesn't have multilayer structure(hidden layers), nor it's using any of CNN principles

weights and biases should be near zero as it may result in collapsed training(neither weights nor biases will change in some cases)
it is a math problem
--------

on the other hand, network does pretty good work despite its shortcomings
or, better to say, well enough considering its shortcomings:

85% correctly classified images using this data: https://www.kaggle.com/datasets/atulanandjha/lfwpeople
80% correctly classified images using this data: https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1

the first one includes people on some distance
the second one includes faces-only images
"""