import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        with open(label_path, 'r') as file:
            self.json_filename_to_classnumber_map = json.load(file)
        # the batch size
        self.batch_size = batch_size
        # the image size
        self.image_size = image_size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        # the current iteration index
        self.iteration_idx = 0
        # the current epoch count
        self.current_epoch_cnt = 0
        self.images_collection = []  # each entry of list is an array 1st element - img file and 2nd element - class name
        self.rng = np.random.default_rng(1234567)  # create a default Generator
        self.parse_raw_images_and_labels()

    def parse_raw_images_and_labels(self):
        # This function creates the images_collection object
        # Each element consists of image file and corresponding label name
        # load all the images here
        for file in os.listdir(self.file_path):
            if file.endswith('.npy'):
                img_file = np.load(os.path.join(self.file_path, file))
                file_name = os.path.splitext(file)[0]
                label_for_file_name = self.json_filename_to_classnumber_map[file_name]
                # add to list image file and its label
                self.images_collection.append([img_file, label_for_file_name])
        
    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        images = []
        labels = []
            
        # if shuffle is true, shuffle the images collection
        if self.shuffle:
            self.rng.shuffle(self.images_collection)
        # get the start and end index for the batch
        start_index = self.iteration_idx * self.batch_size
        end_index = start_index + self.batch_size
        
        # get the images and labels for the batch
        for img_obj in self.images_collection[start_index:end_index]:
            image, label = self.extract_image_and_label_from_obj(img_obj)
            # append the image and label to the batch
            images.append(image)
            labels.append(label)
        # if the batch size is less than the images collection, then append the images and labels from the start
        if len(images) < self.batch_size:
            # reset the iteration index
            self.iteration_idx = 0
            self.current_epoch_cnt += 1
            for img_obj in self.images_collection[0:self.batch_size - len(images)]:
                image, label = self.extract_image_and_label_from_obj(img_obj)
                # append the image and label to the batch
                images.append(image)
                labels.append(label)
                
        # increment the iteration index
        self.iteration_idx += 1
        return np.array(images), np.array(labels)


    def extract_image_and_label_from_obj(self, obj):
        # This function takes one object of the images_collection and returns an image and label
        image = obj[0]
        label = obj[1]
        # augment the image
        image = self.augment(image)
        # resize the image
        image = resize(image, self.image_size)

        return image, label

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if(self.rotation):
            img = self.rotate_img(img)
        if(self.mirroring):
            img = self.mirror_img(img)
        return img
    
    def rotate_img(self, img):
        '''
        Function to rotate the image randomly by 0, 90, 180 or 270 degrees
        '''
        rotation_options = [0, 90, 180, 270]
        for _ in range(self.rng.choice(len(rotation_options))):
            img = np.rot90(img)
        return img
    
    def mirror_img(self, img):
        '''
        Function to mirror the image
        '''
        return np.fliplr(img)

    def current_epoch(self):
        # return the current epoch number
        return self.current_epoch_cnt

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images, labels = self.next()
        n_cols = 3
        n_rows = len(images) // 3 + 1
        fig = plt.figure() 

        for index, label_idx in enumerate(labels):
            fig.add_subplot(n_rows, n_cols, index + 1) 
            plt.title(self.class_name(label_idx))
            plt.imshow(images[index])
        plt.show()

