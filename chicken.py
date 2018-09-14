# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/")
# print(mnist.train.next_batch(3)[0].shape) # (3, 784)
# print(mnist.train.next_batch(3)[0].dtype) # float32
# print(mnist.train.next_batch(3)[0]) # background color is 0.
# print(max(mnist.train.next_batch(3)[0][0])) # max color is 1.0

import matplotlib.pyplot as plt
from PIL import Image
from random import shuffle
from math import sqrt, ceil
import numpy as np
import os



# directory = 'images/'
# pic = Image.open(directory + '0a847fc4663a23248bcc50235342d7e6.jpg')
# pix = np.array(pic)
# print(pix)
# print(pix.shape) # (570, 570, 3)

DEFAULT_INPUT_DIRECTORY = 'test/'
SUPPORTED_FILE_TYPES = {'jpg', 'png'}
UNSUPPORTED_MODES = {'P'}

def get_images(directory=DEFAULT_INPUT_DIRECTORY):
	# Returns a list of numpy arrays of each image with a supported file type
	images = []
	# i=0
	for file in os.listdir(directory):

		pic = Image.open(directory + file)
		
		if pic.mode in UNSUPPORTED_MODES:
			print(file, 'is of unsupported mode:', pic.mode)
			continue

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

def resize_and_smart_crop_square(images, new_size):

	resized_images = []

	for pix in images:

		image = Image.fromarray(pix)
		image_width, image_height = image.size
		scale_factor = 0

		if image_width < image_height:
			scale_factor = new_size/image_width
		else:
			scale_factor = new_size/image_height

		resize_width = int(image_width * scale_factor)
		resize_height = int(image_height * scale_factor)

		image_resized = image.resize((resize_width, resize_height), Image.BILINEAR)

		left = (resize_width - new_size)/2
		top = (resize_height - new_size)/2
		right = (resize_width + new_size)/2
		bottom = (resize_height + new_size)/2

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

def display_many(images, columns=4, rows=5, figure_size=(8, 8), titles=None):

	fig=plt.figure(figsize=figure_size)

	for i in range(1, columns*rows+1):
			a = fig.add_subplot(rows, columns, i)

			if titles is not None:
				a.title.set_text(str(titles[i-1]))

			plt.imshow(images[i-1])

	plt.show()

def display_all(images, figs_per_screen=20, titles=None):

	n = len(images)
	figs_per_screen = min(n, figs_per_screen)

	for i in range(int(n/figs_per_screen)):

		columns, rows = closest_square_factors(figs_per_screen)
		display_many(images[i*figs_per_screen : (i+1)*figs_per_screen], columns=columns, rows=rows, titles=titles)

	# leftovers
	num_leftover = n % figs_per_screen
	if num_leftover != 0:
		columns, rows = closest_square_factors(num_leftover)
		display_many(images[(i+1)*figs_per_screen:], columns=columns, rows=rows, titles=titles)

def data2d_to_grayscale(data2d):

	grayscale_images = []

	for pix in data2d:

		grayscale = np.stack([pix, pix, pix], axis=2)
		grayscale_images.append(grayscale)

	return np.array(grayscale_images)

def squeeze(images):

	squeezed = []

	for pix in images:

		reshaped = np.squeeze(pix, axis=2)
		squeezed.append(reshaped)

	return np.array(squeezed)

# def collage(images, dimensions):

# 	if !dimensions:
# 		dimensions = closest_square_factors(len(images))

# 	width, height = dimensions # (w, h), number of images per dimension
# 	height_image,  width_image, _ = images[0].shape;

# 	for y in height:
# 		for x in width:


class DataSet:

	def __init__(self, data):
		self.data = data
		self.index = 0

	def next_batch(self, batch_size, reuse=True):

		output = []

		for i  in range(batch_size):

			output.append(self.data[self.index])

			self.index += 1
			if self.index >= len(self.data):
				if reuse:
					self.index %= len(self.data)
				else:
					print("Insufficient data left in dataset. Current index:", self.index)
					return

		return np.array(output)
		

def closest_square_factors(integer, larger_first=True):

	factors = []
	root = ceil(sqrt(integer))
	for i in range(root, integer+1):
		if integer % i == 0:
			return (i, int(integer / i)) if  larger_first else (int(integer / i), i)

	return (1, integer)

def save_to_as(images, directory='testoutput/', prefix='img', file_type='jpg'):

	i = 1
	for pix in images:
		image = Image.fromarray(pix)
		image.save(directory + prefix + str(i) + '.' + str.lower(file_type))
		i += 1





# images = get_images('test/')
# images = images + fliplr(images)
# images = grayscale(images)
# images = resize_and_smart_crop_square(images, 256)
# display_all(images)
# data2d = grayscale_to_2d(images)

# print('length', len(data2d))
# dataset = DataSet(data2d)
# display_all(dataset.next_batch(150))

# images = get_images('images/')
# images = images + fliplr(images)
# images = grayscale(images)
# images = resize_and_smart_crop_square(images, 256)
# save_to_as(images, directory='data/', prefix='sketch_', file_type='jpg')
# display_all(images, 100)

# images = get_images('data/')
# data2d = grayscale_to_2d(images)

# print(data2d.shape)
# mean = np.mean(data2d, axis=0)
# mean = np.stack([mean, mean, mean], axis=2)
# mean = np.array(mean, dtype=np.uint8)
# print(mean.shape)
# display([mean])

# images = get_images('data/')
# data2d = grayscale_to_2d(images)
# data2d = np.reshape(data2d, (-1, 256, 256, 1))
# data2d = np.array(data2d, dtype=np.float32)
# data2d = data2d / 255

# im = data2d[0, :, :, 0]
# plt.imshow(1-im, cmap='Greys')
# plt.show()
# print(im)





# images = get_images('images/')
# images = images[:3]
# images = images + fliplr(images)
# images = grayscale(images)
# images = resize_and_smart_crop_square(images, 128)
# display(images)

# images = get_images('images/')
# images = images[:100]
# images = grayscale(images)
# images = resize_and_smart_crop_square(images, 64)
# display_all(images)


# # images 64 preprocessing
# images64 = get_images('images/')
# images64 = images64 + fliplr(images64)
# images64 = grayscale(images64)
# images64 = resize_and_smart_crop_square(images64, 64)
# # display_all(images64)

# data2d_64 = grayscale_to_2d(images64)
# data2d_64 = np.reshape(data2d_64, (-1, 64, 64, 1))
# data2d_64 = np.array(data2d_64, dtype=np.float32)
# data2d_64 = data2d_64 / 255

# print(data2d_64.shape)

# print('Saving...')
# save_to_as(images64, directory='data64/', prefix='sketch64_', file_type='jpg')
# print('Saved.')







def save_as_gif(images, duration=0.1, loops=0, output_directory="testoutput/", filename="my_gif"):


	# sequence = []

	# im = Image.open(....)

	# im is your original image
	# frames = [frame.copy() for frame in ImageSequence.Iterator(im)]

	# frames = images
	# filename = output_directory + filename + '.GIF'
	# # write GIF animation
	# fp = open("filename.gif", "wb")
	# gifmaker.makedelta(fp, frames)
	# fp.close()




	# print(images.shape)
	# frames = [Image.fromarray(pix) for pix in images]
	frames = [Image.fromarray(pix) for pix in images]
	# frames.save(directory + filename+ '.gif', save_all=true, duration=duration, loop=loops)

	frames[0].save(output_directory + filename + '.gif', save_all=True, append_images=frames[1:], duration=duration, loop=loops)

	

