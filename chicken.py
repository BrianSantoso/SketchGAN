# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data/")

# print('WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW')
# print(mnist.train.next_batch(3)[0].shape) # (3, 784)

from PIL import Image
import numpy as np
import os



# directory = 'images/'
# pic = Image.open(directory + '0a847fc4663a23248bcc50235342d7e6.jpg')
# pix = np.array(pic)

# print(pix)
# print(pix.shape) # (570, 570, 3)

DEFAULT_INPUT_DIRECTORY = 'test/'
SUPPORTED_FILE_TYPES = {'jpg'}

def get_images(directory=DEFAULT_INPUT_DIRECTORY):
	# Returns a list of numpy arrays of each image with a supported file type
	images = []
	for file in os.listdir(directory):

		pic = Image.open(directory + file)

		if get_extension(file) in SUPPORTED_FILE_TYPES:
			pix = np.array(pic)
			images.append(pix)

	return images

def get_extension(file_name):
	# Returns the file extension or type
	index = len(file_name) - file_name.rfind('.') - 1
	extension = file_name[-index:]
	return extension

def grayscale(images):

	grayscale_images = []

	for image in images:
		gray = np.mean(image, -1)
		grayscale_images.append(gray)
		# avg = (image[0] + image[1] + image[2]) / 3

	return grayscale_images

images = get_images('test/')
gray = grayscale(images)
print(gray)
print(gray[0].shape)


# print(len(get_images('images/')))
