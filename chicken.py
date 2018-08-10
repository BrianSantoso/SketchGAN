# print(mnist.train.next_batch(3)[0].shape) # (3, 784)

import matplotlib.pyplot as plt
from PIL import Image
from random import shuffle
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
			if pix.ndim == 2:
				pix = np.stack([pix, pix, pix], axis=2)

			if pix.ndim != 3:
				print('Something wrong with', file)

			images.append(pix)

	return images

def get_extension(file_name):
	# Returns the file extension or type
	index = len(file_name) - file_name.rfind('.') - 1
	extension = file_name[-index:]
	return extension

def grayscale(images, channels=3):

	grayscale_images = []

	for image in images:
		gray = np.mean(image, -1)
		gray = np.array(gray, dtype=np.uint8)
		grayscale_images.append(np.stack([gray for i in range(channels)], axis=2))

	return grayscale_images

def display(*images_args):

	for images in images_args:
		for image in images:
			plt.imshow(image)
			plt.show()

def resize(images, width, height):

	resized_images = []

	for pix in images:
		image = Image.fromarray(pix)
		image_resized = image.resize((width, height), Image.BILINEAR)
		new_pix = np.array(image_resized)
		resized_images.append(new_pix)

	return resized_images

def resize_and_smart_crop_square(images, new_len):

	resized_images = []

	for pix in images:

		image = Image.fromarray(pix)
		image_width, image_height = image.size
		scale_factor = 0

		if image_width < image_height:
			scale_factor = new_len/image_width
		else:
			scale_factor = new_len/image_height

		resize_width = int(image_width * scale_factor)
		resize_height = int(image_height * scale_factor)

		image_resized = image.resize((resize_width, resize_height), Image.BILINEAR)

		left = (resize_width - new_len)/2
		top = (resize_height - new_len)/2
		right = (resize_width + new_len)/2
		bottom = (resize_height + new_len)/2

		image_croppped = image_resized.crop((left, top, right, bottom))

		new_pix = np.array(image_croppped)
		resized_images.append(new_pix)

	return resized_images

def fliplr(images):

	flipped_images = []

	for image in images:
		flipped = np.fliplr(image)
		flipped_images.append(flipped)

	return flipped_images

def flatten(images):

	flattened_images = []

	for image in images:
		flattened = image.flatten()
		flattened_images.append(flattened)

	return flattened_images

def grayscale_to_2d(images):

	data_2d = []

	for image in images:

		image_2d = np.mean(image, axis=2)
		data_2d.append(image_2d)

	return np.asarray(data_2d, dtype=np.uint8)

# images = get_images('test/')
# images = images[:10]
# with_flipped = images + fliplr(images)
# gray = grayscale(with_flipped, channels=3)
# resized_images = resize_and_smart_crop_square(gray, 256)
# display(resized_images)


# data_2d = grayscale_to_2d(resized_images)
# print(data_2d[0])
# print(data_2d[0].shape)
# gim = np.stack([data_2d[0] for i in range(3)], axis=2)
# plt.imshow(gim)
# plt.show()