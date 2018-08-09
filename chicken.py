# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data/")

# print('WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW')
# print(mnist.train.next_batch(3)[0].shape) # (3, 784)

import matplotlib.pyplot as plt
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

			# if only 1 color channel then triple it
			if len(pix.shape) == 2:
				pix = np.stack([pix, pix, pix], axis=2)

			if len(pix.shape) != 3:
				print('WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW')
			images.append(pix)

	return images

def get_extension(file_name):
	# Returns the file extension or type
	index = len(file_name) - file_name.rfind('.') - 1
	extension = file_name[-index:]
	return extension

def grayscale(images, channels=1):

	grayscale_images = []

	for image in images:

		gray = np.mean(image, -1)
		gray = np.array(gray, dtype=np.uint8)
		grayscale_images.append(np.stack([gray for i in range(channels)], axis=2))
		# avg = (image[0] + image[1] + image[2]) / 3

	return grayscale_images

def display(*images_args):

	for images in images_args:
		for image in images:
			plt.imshow(image)
			plt.show()


# images = get_images('test/')
# gray = grayscale(images, channels=3)
# print(gray)
# print(gray[0].shape)
# print(images[0].dtype)
# display(images)
# display(gray)

images = get_images('images/')
images = images[:10]
gray = grayscale(images, channels=3)
display(images, gray)