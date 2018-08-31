import tensorflow as tf
import numpy as np
import chicken
import datetime
import matplotlib.pyplot as plt
from random import randint
# from tensorflow.examples.tutorials.mnist import input_data

'''
Helpul Links:
	https://github.com/llSourcell/Generative_Adversarial_networks_LIVE/blob/master/EZGAN.ipynb
	-> (fto ix cell 4 in jupyter notebook) https://github.com/llSourcell/Generative_Adversarial_networks_LIVE/issues/2
	https://github.com/manicman1999/GAN256/blob/master/main.py

	http://cs231n.github.io/neural-networks-2/
	https://en.wikipedia.org/wiki/Truncated_normal_distribution
	https://stats.stackexchange.com/questions/87248/is-binary-logistic-regression-a-special-case-of-multinomial-logistic-regression#comment609940_87270
	https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks

	https://www.tensorflow.org/api_docs/python/tf/reshape

	https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
	https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose

	Checkerboard artifacts -> https://distill.pub/2016/deconv-checkerboard/

'''

class DCGAN:

	def __init__(self):

		# self.mnist = input_data.read_data_sets("MNIST_data/")
		
		self.sketch_dataset = chicken.DataSet(self.load_data())

		
		self.iterations = 499500
		self.load_from_ckpt = 499500

		self.batch_size = 16
		self.z_dimensions = 100
		self.learning_rate = 0.0001

	def load_data(self):

		print('Loading Data...')
		images = chicken.get_images('data64/')
		data2d = chicken.grayscale_to_2d(images)
		data2d = np.reshape(data2d, (-1, 64, 64, 1))
		data2d = np.array(data2d, dtype=np.float32)
		data2d = data2d / 255
		print('shape:',data2d.shape)
		# print('sample image:', data2d[0])
		print('Data Loaded.')
		return data2d

	def discriminator(self, x_image, reuse=False):

		if(reuse):
			tf.get_variable_scope().reuse_variables()


		# Convolutional Block 1
		# 64, 5x5 filters
		d_w1 = tf.get_variable('d_w1', [5, 5, 1, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b1 = tf.get_variable('d_b1', [64], initializer=tf.constant_initializer(0))
		d1 = tf.nn.conv2d(input=x_image, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
		d1 = d1 + d_b1
		d1 = tf.nn.leaky_relu(d1, alpha=0.2)
		d1 = tf.nn.dropout(d1, keep_prob=0.25)
		d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		# Convolutional Block 2
		# 128, 5x5 filters
		d_w2 = tf.get_variable('d_w2', [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b2 = tf.get_variable('d_b2', [128], initializer=tf.constant_initializer(0))
		d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding="SAME")
		d2 = d2 + d_b2
		d2 = tf.nn.leaky_relu(d2, alpha=0.2)
		d2 = tf.nn.dropout(d2, keep_prob=0.25)
		d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

		# Convolutional Block 3
		# 256, 5x5 filters
		d_w3 = tf.get_variable('d_w3', [5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b3 = tf.get_variable('d_b3', [256], initializer=tf.constant_initializer(0))
		d3 = tf.nn.conv2d(input=d2, filter=d_w3, strides=[1, 1, 1, 1], padding="SAME")
		d3 = d3 + d_b3
		d3 = tf.nn.leaky_relu(d3, alpha=0.2)
		d3 = tf.nn.dropout(d3, keep_prob=0.25)
		d3 = tf.nn.avg_pool(d3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

		# Convolutional Block 4
		# 64, 3x3 filters
		d_w4 = tf.get_variable('d_w4', [5, 5, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b4 = tf.get_variable('d_b4', [512], initializer=tf.constant_initializer(0))
		d4 = tf.nn.conv2d(input=d3, filter=d_w4, strides=[1, 1, 1, 1], padding="SAME")
		d4 = d4 + d_b4
		d4 = tf.nn.leaky_relu(d4, alpha=0.2)
		d4 = tf.nn.dropout(d4, keep_prob=0.25)
		d4 = tf.nn.avg_pool(d4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

		# Fully connected layer 1
		d_w5 = tf.get_variable('d_w5', [4 * 4 * 512, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b5 = tf.get_variable('d_b5', [128], initializer=tf.constant_initializer(0))
		d5 = tf.reshape(d4, [-1, 4 * 4 * 512])
		d5 = tf.matmul(d5, d_w5)
		d5 = d5 + d_b5
		d5 = tf.nn.leaky_relu(d5, alpha=0.2)

		# Fully connected layer 2
		d_w6 = tf.get_variable('d_w6', [128, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b6 = tf.get_variable('d_b6', [1], initializer=tf.constant_initializer(0))
		d6 = tf.matmul(d5, d_w6)
		d6 = d6 + d_b6

		# dimensions of output tensor: batchsize x 1
		# (binary classification)
		return d6


	def generator(self, batch_size, z_dim, z_vector=None):
		if z_vector is not None:
			z = z_vector
		else:
			z = tf.truncated_normal([batch_size, z_dim], mean=0, stddev=1, name='z')

		g_w1 = tf.get_variable('g_w1', [z_dim, 4*4*1024], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b1 = tf.get_variable('g_b1', [4*4*1024], initializer=tf.constant_initializer(0))
		g1 = tf.matmul(z, g_w1) + g_b1
		g1 = tf.reshape(g1, [-1, 4, 4, 1024])
		g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
		g1 = tf.nn.relu(g1)

		g_w2 = tf.get_variable('g_w2', [5, 5, 512, 1024], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b2 = tf.get_variable('g_b2', [512], dtype=tf.float32, initializer=tf.constant_initializer(0))
		# print(g1.get_shape())
		# print(g_w2.get_shape())
		g2 = tf.nn.conv2d_transpose(g1, g_w2, output_shape=[batch_size, 8, 8, 512], strides=[1, 2, 2, 1])
		g2 = g2 + g_b2 #tf.nn.bias_add?
		# g2 = tf.reshape(tf.nn.bias_add(g2, g_b2), g2.get_shape())
		g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
		g2 = tf.nn.relu(g2)

		g_w3 = tf.get_variable('g_w3', [5, 5, 256, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b3 = tf.get_variable('g_b3', [256], dtype=tf.float32, initializer=tf.constant_initializer(0))
		g3 = tf.nn.conv2d_transpose(g2, g_w3, output_shape=[batch_size, 16, 16, 256], strides=[1, 2, 2, 1])
		g3 = g3 + g_b3
		# g3 = tf.reshape(tf.nn.bias_add(g3, g_b3), g3.get_shape())
		g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
		g3 = tf.nn.relu(g3)

		g_w4 = tf.get_variable('g_w4', [5, 5, 128, 256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b4 = tf.get_variable('g_b4', [128], dtype=tf.float32, initializer=tf.constant_initializer(0))
		g4 = tf.nn.conv2d_transpose(g3, g_w4, output_shape=[batch_size, 32, 32, 128], strides=[1, 2, 2, 1])
		g4 = g4 + g_b4
		# g4 = tf.reshape(tf.nn.bias_add(g4, g_b4), g4.get_shape())
		g4 = tf.contrib.layers.batch_norm(g4, epsilon=1e-5, scope='bn4')
		g4 = tf.nn.relu(g4)

		g_w5 = tf.get_variable('g_w5', [5, 5, 1, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b5 = tf.get_variable('g_b5', [1], dtype=tf.float32, initializer=tf.constant_initializer(0))
		g5 = tf.nn.conv2d_transpose(g4, g_w5, output_shape=[batch_size, 64, 64, 1], strides=[1, 2, 2, 1])
		g5 = g5 + g_b5
		# g5 = tf.reshape(tf.nn.bias_add(g5, g_b5), g5.get_shape())
		# no batch norm
		# g5 = tf.contrib.layers.batch_norm(g5, epsilon=1e-5, scope='bn5')
		alpha = 0.2
		g5 = tf.sigmoid(alpha * g5)

		return g5


	def run_session(self):

		sess = tf.Session()
		
		x_placeholder = tf.placeholder('float', shape=[None, 64, 64, 1], name='x_placeholder')
		# G(z)
		Gz = self.generator(self.batch_size, self.z_dimensions)
		# D(x)
		Dx = self.discriminator(x_placeholder)

		with tf.variable_scope(tf.get_variable_scope()) as scope:
			pass

		# D(G(z))
		Dg = self.discriminator(Gz, reuse=True)
		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))
		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([self.batch_size, 1], 0.9)))
		d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
		d_loss = d_loss_real + d_loss_fake

		# get lists of variables to optimize for generator and discriminator
		tvars = tf.trainable_variables()
		d_vars = [var for var in tvars if 'd_' in var.name]
		g_vars = [var for var in tvars if 'g_' in var.name]

		with tf.variable_scope(scope):
			d_trainer_fake = tf.train.AdamOptimizer(self.learning_rate).minimize(d_loss_fake, var_list=d_vars)
			d_trainer_real = tf.train.AdamOptimizer(self.learning_rate).minimize(d_loss_real, var_list=d_vars)
			g_trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(g_loss, var_list=g_vars)

		tf.summary.scalar('Generator_loss', g_loss)
		tf.summary.scalar('Discriminator_loss_real', d_loss_real)
		tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

		d_real_count_ph = tf.placeholder(tf.float32)
		d_fake_count_ph = tf.placeholder(tf.float32)
		g_count_ph = tf.placeholder(tf.float32)

		tf.summary.scalar('d_real_count', d_real_count_ph)
		tf.summary.scalar('d_fake_count', d_fake_count_ph)
		tf.summary.scalar('g_count', g_count_ph)

		# Check how discriminator is doing on generated and real images
		d_on_generated = tf.reduce_mean(self.discriminator(self.generator(self.batch_size, self.z_dimensions)))
		d_on_real = tf.reduce_mean(self.discriminator(x_placeholder))

		tf.summary.scalar('d_on_generated_eval', d_on_generated)
		tf.summary.scalar('d_on_real_eval', d_on_real)

		images_for_tensorboard = self.generator(self.batch_size, self.z_dimensions)
		tf.summary.image('Generated_images', images_for_tensorboard, 10)
		merged = tf.summary.merge_all()
		logdir = 'tensorboard/gan/'
		writer = tf.summary.FileWriter(logdir, sess.graph)
		print(logdir)



		self.saver = tf.train.Saver(max_to_keep=3)
		
		sess.run(tf.global_variables_initializer())

		if self.load_from_ckpt:
			self.load(sess, 'models64/pretrained_gan.ckpt-' + str(self.load_from_ckpt))
		
		gLoss = 0
		dLossFake, dLossReal = 1, 1
		d_real_count, d_fake_count, g_count = 0, 0, 0

		for i in range(self.load_from_ckpt, self.iterations):
			real_image_batch = self.sketch_dataset.next_batch(self.batch_size)
			if dLossFake > 0.6:
				# train discriminator on generated images
				_, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss], {x_placeholder: real_image_batch})

				d_fake_count += 1

			if gLoss > 0.5:
				# train the generator
				_, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss], {x_placeholder: real_image_batch})

				g_count += 1

			if dLossReal > 0.45:

				# _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, gLoss],
				# 											 {x_placeholder: real_image_batch})

				_, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
                                                    {x_placeholder: real_image_batch})

				d_real_count += 1

			# if i % 10 == 0:
				# real_image_batch = self.mnist.validation.next_batch(self.batch_size)[0].reshape([self.batch_size, 28, 28, 1])
				# summary = sess.run(merged, {x_placeholder: real_image_batch, d_real_count_ph: d_real_count,
				# 							d_fake_count_ph: d_fake_count, g_count_ph: g_count})

				# writer.add_summary(summary, i)
				# d_real_count, d_fake_count, g_count = 0, 0, 0

			if i % 500 == 0:
				images = sess.run(self.generator(3, self.z_dimensions))
				d_result = sess.run(self.discriminator(x_placeholder), {x_placeholder: images})
				print('TRAINING STEP', i, 'AT', datetime.datetime.now())
				for j in range(1):
					print('Discriminator classification', d_result[j])
					# im = 1-images[j, :, :, 0]
					# plt.imshow(im.reshape([28, 28]), cmap='Greys')
					# plt.ion()
					# plt.imshow(im, cmap='Greys')
					# plt.show()
					# plt.plot(im)
					# plt.draw()
					# plt.show(block=False)

			# if i == 15000 or i == 5000:
			# 	images = sess.run(self.generator(3, self.z_dimensions))
			# 	d_result = sess.run(self.discriminator(x_placeholder), {x_placeholder: images})
			# 	for j in range(3):
			# 		print('Discriminator classificationnnnnnnnnn', d_result[j])
			# 		im = 1-images[j, :, :, 0]
			# 		plt.imshow(im.reshape([64, 64]), cmap='Greys')
			# 		plt.show()


			if i % 500 == 0:
				# save_path = saver.save(sess, 'models/pretrained_gan.ckpt', global_step=i)
				# print('Saved to %s' % save_path)
				if i != self.load_from_ckpt:
					self.save(sess, 'models64/pretrained_gan.ckpt', i)

		

		# real_images = self.mnist.validation.next_batch(10)[0].reshape([10, 28, 28, 1])
		# real_eval = sess.run(self.discriminator(x_placeholder), {x_placeholder: real_images})

		# rib = self.sketch_dataset.next_batch(self.batch_size)
		# _, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss], {x_placeholder: rib})
		# print(dLossReal)
		# print(dLossFake)
		# print(gLoss)


		# test_images = sess.run(self.generator(100, self.z_dimensions))
		# test_eval = sess.run(self.discriminator(x_placeholder), {x_placeholder: test_images})
		
		# # # display images and show discriminator's probabilities
		# # for i in range(20):
		# # 	print(test_eval[i])
		# # 	# print(test_images[i])
		# # 	plt.imshow(1-test_images[i, :, :, 0], cmap='Greys')
		# # 	plt.show()
		
		# test_images = chicken.squeeze(test_images)
		# test_images = chicken.data2d_to_grayscale(test_images)
		# chicken.display_all(test_images, 20)

		# total_parameters = self.get_total_parameters()
		# print("total_parameters: ", total_parameters)


		#234534678678
		#56434523445
		#6972514296
		#8027089512
		#383
		#19040
		#94889
		#89018
		#934732
		# for i in range(10):
		# 	self.latent_space_traversal(sess, segments=19, inclusivity=(True, True))

		self.test_interpolation_sequence_1(sess)


		# z3 = self.noise(self.z_dimensions, seed=2345656)
		# test_image3 = sess.run(self.generator(1, self.z_dimensions, z3))

		# interpolation1 = self.interpolate(test_image1[0], test_image2[0], segments=19, inclusivity=(True, True))
		# interpolation2 = self.interpolate(test_image2[0], test_image3[0], segments=19, inclusivity=(False, True))
		# interpolation = np.concatenate([interpolation1, interpolation2])
		# self.display_all(interpolation, 20)



		# chicken.display_all(np.reshape(test_images, (-1, 256, 256)))
		# chicken.display_all(np.reshape(self.sketch_dataset.next_batch(10), (-1, 256, 256)))

		# asdf = self.sketch_dataset.next_batch(10)
		# for i in range(10):
		# 	print(asdf[i])
		# 	plt.imshow(1-asdf[i, :, :, 0], cmap='Greys')
		# 	plt.show()

		# # Now do the same for real MNIST images
		# for i in range(10):
		#     print(real_eval[i])
		#     plt.imshow(real_images[i, :, :, 0], cmap='Greys')
		#     plt.show()

	def save(self, sess, dir, iteration):
		save_path = self.saver.save(sess, dir, global_step=iteration)
		print('Saved to %s' % save_path)
		return

	def load(self, sess, prefix):
		self.saver.restore(sess, prefix)
		print("Model restored.")
		return

	def get_total_parameters(self):
		total_parameters = 0
		#iterating over all variables
		for variable in tf.trainable_variables():  
			local_parameters = 1
			#getting shape of a variable
			shape = variable.get_shape()
			for i in shape:
				#mutiplying dimension values
				local_parameters *= i.value
			total_parameters+=local_parameters
		return total_parameters

	def interpolate(self, a, b, segments, inclusivity=(False, True)):
		# Linearly interpolate betweem 2 n-dimensional vectors
		includeStart, includeEnd = inclusivity

		increment = (b-a)/segments
		startIndex = 1-int(includeStart)
		endIndex = segments + int(includeEnd)
		vectors = [(a + increment*i) for i in range(startIndex, endIndex)]

		return np.array(vectors)

	def latent_space_traversal(self, sess, seed1=None, seed2=None, segments=20, inclusivity=(True, True), display=False):

		seed1 = seed1 if seed1 is not None else randint(0, 1000000)
		z1 = self.noise(self.z_dimensions, seed=seed1)
		test_image1 = sess.run(self.generator(1, self.z_dimensions, z1))

		seed2 = seed2 if seed2 is not None else randint(0, 1000000)
		z2 = self.noise(self.z_dimensions, seed=seed2)
		test_image2 = sess.run(self.generator(1, self.z_dimensions, z2))

		interpolation = self.interpolate(test_image1[0], test_image2[0], segments=segments, inclusivity=inclusivity)
		
		if display:
			self.display_all(interpolation, 20)
			print(seed1, seed2)
		return interpolation

	def noise(self, z_dim, mean=0, stddev=1,seed=None, amount=1):

		output = tf.truncated_normal([amount, z_dim], mean=0, stddev=1, seed=seed) if seed else tf.truncated_normal([z_dim], mean=0, stddev=1)

		return output

	def display_all(self, images, figs_per_page=20):

		images = chicken.squeeze(images)
		images = chicken.data2d_to_grayscale(images)
		chicken.display_all(images, figs_per_page)
		return

	def save_as_gif(self, images, duration=0.1, loops=0, dither=1, output_directory="testoutput/", filename="my_gif"):

		images = chicken.squeeze(images)
		images = chicken.data2d_to_grayscale(images)
		images = images * 255
		images = np.array(images, dtype=np.uint8)
		chicken.save_as_gif(images=images, duration=duration, loops=loops, output_directory=output_directory, filename=filename)
		return

	def test_interpolation_sequence_1(self, sess):
		a = self.latent_space_traversal(sess, 934732, 234534678678, segments=60, inclusivity=(False, True))
		a0 = self.latent_space_traversal(sess, 234534678678, 234534678678, segments=20, inclusivity=(False, True))

		b = self.latent_space_traversal(sess, 234534678678, 56434523445, segments=60, inclusivity=(False, True))
		b0 = self.latent_space_traversal(sess, 56434523445, 56434523445, segments=20, inclusivity=(False, True))

		c = self.latent_space_traversal(sess, 56434523445, 6972514296, segments=60, inclusivity=(False, True))
		c0 = self.latent_space_traversal(sess, 6972514296, 6972514296, segments=20, inclusivity=(False, True))

		d = self.latent_space_traversal(sess, 6972514296, 8027089512, segments=60, inclusivity=(False, True))
		d0 = self.latent_space_traversal(sess, 8027089512, 8027089512, segments=20, inclusivity=(False, True))

		e = self.latent_space_traversal(sess, 8027089512, 383, segments=60, inclusivity=(False, True))
		e0 = self.latent_space_traversal(sess, 383, 383, segments=20, inclusivity=(False, True))

		f = self.latent_space_traversal(sess, 383, 19040, segments=60, inclusivity=(False, True))
		f0 = self.latent_space_traversal(sess, 19040, 19040, segments=20, inclusivity=(False, True))

		g = self.latent_space_traversal(sess, 19040, 94889, segments=60, inclusivity=(False, True))
		g0 = self.latent_space_traversal(sess, 94889, 94889, segments=20, inclusivity=(False, True))

		h = self.latent_space_traversal(sess, 94889, 89018, segments=60, inclusivity=(False, True))
		h0 = self.latent_space_traversal(sess, 89018, 89018, segments=20, inclusivity=(False, True))

		i = self.latent_space_traversal(sess, 89018, 934732, segments=60, inclusivity=(False, True))
		i0 = self.latent_space_traversal(sess, 934732, 934732, segments=20, inclusivity=(False, True))

		# image_sequence = np.concatenate([a, a0, b, b0, c, c0, d, d0, e, e0, f, f0, g, g0, h, h0, i, i0])
		image_sequence = np.concatenate([a, b, c, d, e, f, g, h, i])

		image_sequence = chicken.squeeze(image_sequence)
		image_sequence = chicken.data2d_to_grayscale(image_sequence)
		image_sequence = image_sequence * 255
		image_sequence = np.array(image_sequence, dtype=np.uint8)

		# chicken.save_to_as(image_sequence, directory='testoutput2/test5/', prefix='imgif', file_type='jpg')
		# self.save_as_gif(image_sequence, duration=1/30, loops=0, output_directory="testoutput/", filename="latent_space_traversal_3")


gan = DCGAN()
gan.run_session()