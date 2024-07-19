import tarfile
from PIL import Image
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import zipfile

# thiss class is responsible for every image processing operation, like getting image's pixel data, getting features
# and extracting new images from archives
# overall it is used to get new image, process it and conver its parameters into features

class ImagesData:


    def __init__(self, filetype1, filetype2, position, minibatch_size, is_face, number_of_training_photos):

        self.__filetype1 = filetype1
        self.__filetype2 = filetype2
        self.__minibatch_size = minibatch_size
        self.__position1 = position
        self.__position2 = position
        self.__filetype = filetype1
        self.__is_face = is_face
        self.__number_of_training_photos = number_of_training_photos
        


    def set_img_data(self):

        if self.__filetype == self.__filetype1:
            self.__img_data = self.extract_and_process_image_zip(self.__filetype1,self.__position1,self.__minibatch_size)
        else:
            self.__img_data = self.extract_and_process_image_zip(self.__filetype2,self.__position2,self.__minibatch_size)
    
    def get_img_data(self):
        
        return self.__img_data

    def set_filetype(self,filetype):

        if filetype == 'data':
            self.__filetype = self.__filetype1
        elif filetype == 'mask':
            self.__filetype = self.__filetype2


    def get_filetype(self):

        if self.__filetype == self.__filetype1:
            return 'data'
        elif self.__filetype == self.__filetype2:
            return 'mask'


    def get_filetype1(self):
        
        return self.__filetype1
    

    def get_filetype2(self):

        return self.__filetype2

    def get_position(self, _for = ''):
        
        if _for == 'max':
            return max(self.__position1, self.__position2)
        
        if self.__filetype == self.__filetype1:
            return self.__position1
        else:
            return self.__position2

    def set_position(self,position, quantity = 'one'):
        
        if self.__filetype == self.__filetype1:
            self.__position1 = position
        else:
            self.__position2 = position

        if quantity == 'all':
            self.__position1 = position
            self.__position2 = position

    def set_position_increment(self):

        if self.__filetype == self.__filetype1:
            self.__position1 += 1
        else:
            self.__position2 += 1

    def set_features(self, index):

        self.__features = self.haar_features(self.__img_data[index])

    def get_features(self):

        return self.__features

    def set_is_face(self, is_face):

        self.__is_face = is_face

    def get_is_face(self):
        
        return self.__is_face


    def get_minibatch_size(self):

        return self.__minibatch_size


    def get_number_of_training_photos(self):

        return self.__number_of_training_photos


    def get_luminance(self,img_target):
        img_target_grey_scaled = img_target.convert('L')
        width, height = img_target.size
        pixels_luminance = []
        luminance_values = []
       
        for y in range(height):

            for x in range(width):
                pixel_value = img_target_grey_scaled.getpixel((x, y))
                luminance = pixel_value / 255.0
                pixels_luminance.append(luminance)

            luminance_values.append(pixels_luminance)
            pixels_luminance = []

        return luminance_values


    def haar_features(self,image_data):
        print('getting_features...')
        features = []
        widths = [3,5,10,25,50]
        lum_sum =[[0]*251 for i in range(251)]

        for string in range(1,251):
            for pixel in range(1,251):
                lum_sum[string][pixel] = image_data[string-1][pixel-1] + lum_sum[string][pixel-1] + lum_sum[string-1][pixel] - lum_sum[string-1][pixel-1]


        for width in widths:
            for string in range(width+1,250-width):
                for pixel in range(width+1,250-width):

                    gorizontal = (lum_sum[string+width][pixel+width] - lum_sum[string-width][pixel+width] - lum_sum[string+width][pixel-width] + lum_sum[string-width][pixel-width]) - 2*(lum_sum[string+width][pixel] - lum_sum[string+width][pixel-width] - lum_sum[string-width][pixel] + lum_sum[string-width][pixel-width])
                    vertical = (lum_sum[string+width][pixel+width] - lum_sum[string+width][pixel-width] - lum_sum[string-width][pixel+width] + lum_sum[string-width][pixel-width]) - 2*(lum_sum[string][pixel+width] - lum_sum[string][pixel-width] - lum_sum[string-width][pixel+width] + lum_sum[string-width][pixel-width])
                    angle_45 = (lum_sum[string+width][pixel+width] - 2*lum_sum[string][pixel+width] - 2*lum_sum[string+width][pixel] + 2 * lum_sum[string-width][pixel+width] + 2*lum_sum[string+width][pixel-width]) + 4*(lum_sum[string][pixel] - lum_sum[string][pixel - width] - lum_sum[string-width][pixel] + lum_sum[string-width][pixel-width])
                    laplasian = (lum_sum[string+width][pixel+width] - lum_sum[string+width][pixel+int(1/6*width)] - lum_sum[string-width][pixel+width] + lum_sum[string-width][pixel+int(1/6*width)]) - (lum_sum[string+width][pixel+int(1/6*width)] - lum_sum[string+width][pixel-int(1/6*width)] - lum_sum[string-width][pixel+int(1/6*width)] + lum_sum[string-width][pixel-int(1/6*width)]) + (lum_sum[string+width][pixel-int(1/6*width)] - lum_sum[string+width][pixel-width] - lum_sum[string-width][pixel-int(1/6*width)] + lum_sum[string-width][pixel-(width)])
                    if gorizontal >=0:
                        features.append(float(0.5+gorizontal/(((width+1)**2)/2)*0.5))
                    else:
                        features.append(float(0.5-gorizontal/(((width+1)**2)/2)*0.5))
                    if gorizontal >=0:
                        features.append(float(0.5+vertical/(((width+1)**2)/2)*0.5))
                    else:
                        features.append(float(0.5-vertical/(((width+1)**2)/2)*0.5))
                    if gorizontal >=0:
                        features.append(float(0.5+angle_45/(((width+1)**2)/2)*0.5))
                    else:
                        features.append(float(0.5-angle_45/(((width+1)**2)/2)*0.5))
                    if gorizontal >=0:
                        features.append(float(0.5+laplasian/(((width+1)**2)*2/3)*0.5))
                    else:
                        features.append(float(0.5-laplasian/(((width+1)**2)*2/3)*0.5))

        
        print('features ready')
        return features


    def extract_and_process_image_tar(self,tgz_path,current,minibatch_size):
    # Open the .tgz file
        with tarfile.open(tgz_path, "r:gz") as tgz:
            # Initialize a list to hold the pixel data of all images
            images_data = []
            valid = 0
            # Iterate over each member in the .tgz archive
            while valid < int(minibatch_size):

                member = tgz.getmembers()[current]
                # Check if the current member is a file
                if member.isfile() and member.name.endswith('.jpg'):
                    valid += 1
                    current += 1
                    print(member.name)
                    # Extract the image file's content
                    file = tgz.extractfile(member)
                    img_data = file.read()
                    # Open the image using Pillow
                    image = Image.open(BytesIO(img_data))
                    image = image.resize([250,250])
                    # Optionally, you can aplen(current)ply any processing here, e.g., resize, crop, etc.

                    # Convert the image(brightness/luminance) to a numpy array and store the pixel data
                    images_data.append(np.array(self.get_luminance(image)))
                else:
                    current += 1

            return images_data


    def extract_and_process_image_zip(self,zip_path, current, minibatch_size):
        # Open the .zip file
        with zipfile.ZipFile(zip_path, "r") as zip:
        
            images_data = []
            valid = 0

            people_dir = [name for name in zip.namelist() if (name.startswith('images/images/') or self.__filetype == self.__filetype2)]

            for member in people_dir[current:]:

                if valid < int(minibatch_size):
            
                    if member.endswith('.jpg'):
                        valid += 1

                        print(member)
                        # Extract the image file's content
                        file = zip.open(member)
                        img_data = file.read()
                    
                        image = Image.open(BytesIO(img_data))
                        image = image.resize([250, 250])
                    

                        images_data.append(np.array(self.get_luminance(image)))

        return images_data