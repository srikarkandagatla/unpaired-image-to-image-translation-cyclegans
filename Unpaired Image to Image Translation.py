#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from random import random
from numpy import load, zeros, ones, asarray, vstack
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate
from keras.preprocessing.image import img_to_array, load_img
from os import listdir
from matplotlib import pyplot as plt
import numpy as np


# In[ ]:


import sys
sys.path.append('/Path')
from instancenormalization import InstanceNormalization


# In[ ]:


# discriminator model (70x70 patchGAN)
def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	model = Model(in_image, patch_out)

	model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model


# In[ ]:


# generator a resnet block to be used in the generator.
def resnet_block(n_filters, input_layer):
	init = RandomNormal(stddev=0.02)
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Concatenate()([g, input_layer])
	return g


# In[ ]:


# define the  generator model - encoder-decoder type architecture
def define_generator(image_shape, n_resnet=9):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)

	model = Model(in_image, out_image)
	return model


# In[ ]:


def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	g_model_1.trainable = True
	d_model.trainable = False
	g_model_2.trainable = False

	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	output_f = g_model_2(gen1_out)
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)

	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])

	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss=['mse', 'mae', 'mae', 'mae'],
               loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model


# In[ ]:


# load and prepare training images
def load_real_samples(filename):
	data = load(filename)
	X1, X2 = data['arr_0'], data['arr_1']
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]


# In[ ]:


def generate_real_samples(dataset, n_samples, patch_shape):
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y


# In[ ]:


def generate_fake_samples(g_model, dataset, patch_shape):
	X = g_model.predict(dataset)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y


# In[ ]:


def save_models(step, g_model_AtoB, g_model_BtoA):
	filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))


# In[ ]:


def summarize_performance(step, g_model, trainX, name, n_samples=5):
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	pyplot.savefig(filename1)
	pyplot.close()


# In[ ]:


def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			selected.append(image)
		else:
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)


# In[ ]:


# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=1):
	n_epochs, n_batch, = epochs, 1
	n_patch = d_model_A.output_shape[1]
	trainA, trainB = dataset
	poolA, poolB = list(), list()
	bat_per_epo = int(len(trainA) / n_batch)
	n_steps = bat_per_epo * n_epochs

	for i in range(n_steps):
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
		X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)

		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

		print('Iteration>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
		if (i+1) % (bat_per_epo * 1) == 0:
			summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
			summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
		if (i+1) % (bat_per_epo * 5) == 0:
            
			save_models(i, g_model_AtoB, g_model_BtoA)


# # Training the Model

# In[ ]:


# load all images in a directory into memory
def load_images(path, size=(256,256)):
	data_list = list()
	for filename in listdir(path):
		pixels = load_img(path + filename, target_size=size)
		pixels = img_to_array(pixels)
		data_list.append(pixels)
	return asarray(data_list)


# In[ ]:


# dataset path
path = '/path/'


# In[ ]:


# load dataset A - Monet paintings
dataA_all = load_images(path + 'trainA/')
print('Loaded dataA: ', dataA_all.shape)

from sklearn.utils import resample
#To get a subset of all images, for faster training during demonstration
dataA = resample(dataA_all,
                 replace=False,
                 n_samples=500,
                 random_state=42)

# load dataset B - Photos
dataB_all = load_images(path + 'trainB/')
print('Loaded dataB: ', dataB_all.shape)

dataB = resample(dataB_all,
                 replace=False,
                 n_samples=500,
                 random_state=42)


# In[ ]:


# plot source images
n_samples = 3
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + i)
	plt.axis('off')
	plt.imshow(dataA[i].astype('uint8'))
    
# plot target image
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + n_samples + i)
	plt.axis('off')
	plt.imshow(dataB[i].astype('uint8'))
plt.show()


# In[ ]:


# load image data
data = [dataA, dataB]

print('Loaded', data[0].shape, data[1].shape)


# In[ ]:


#Preprocess data to change input range to values between -1 and 1
def preprocess_data(data):
	X1, X2 = data[0], data[1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]


# In[ ]:


dataset = preprocess_data(data)


# In[ ]:


image_shape = dataset[0].shape[1:]
g_model_AtoB = define_generator(image_shape)
g_model_BtoA = define_generator(image_shape)
d_model_A = define_discriminator(image_shape)
d_model_B = define_discriminator(image_shape)
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)


# In[ ]:


from datetime import datetime
start = datetime.now()
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=5)

stop = datetime.now()
#Execution time of the model
execution_time = stop - start
print("Execution time is: ", execution_time)


# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint

# Define a checkpoint callback
checkpoint_path = 'checkpoint.h5'
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                      save_weights_only=True,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)

# During model.compile(), add this callback to the callbacks list
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'val_accuracy'],
              callbacks=[checkpoint_callback])


# In[ ]:


# Use the saved cyclegan models for image translation
from instancenormalization import InstanceNormalization
from keras.models import load_model
from matplotlib import pyplot
from numpy.random import randint

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	return X


# In[ ]:


# plot the image, its translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
	images = vstack((imagesX, imagesY1, imagesY2))
	titles = ['Real', 'Generated', 'Reconstructed']
	images = (images + 1) / 2.0
	for i in range(len(images)):
		pyplot.subplot(1, len(images), 1 + i)
		pyplot.axis('off')
		pyplot.imshow(images[i])

		pyplot.title(titles[i])
	pyplot.show()


# In[ ]:


# load dataset
A_data = resample(dataA_all,
                 replace=False,
                 n_samples=50,
                 random_state=42)

B_data = resample(dataB_all,
                 replace=False,
                 n_samples=50,
                 random_state=42)

A_data = (A_data - 127.5) / 127.5
B_data = (B_data - 127.5) / 127.5


# In[ ]:


# plot A->B->A (Monet to photo to Monet)
A_real = select_sample(A_data, 1)
B_generated  = model_AtoB.predict(A_real)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real, B_generated, A_reconstructed)

# plot B->A->B (Photo to Monet to Photo)
B_real = select_sample(B_data, 1)
A_generated  = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)


# In[ ]:


#Load a single custom image
test_image = load_img('monet2photo/sunset256.jpg')
test_image = img_to_array(test_image)
test_image_input = np.array([test_image])
test_image_input = (test_image_input - 127.5) / 127.5

# plot B->A->B (Photo to Monet to Photo)
monet_generated  = model_BtoA.predict(test_image_input)
photo_reconstructed = model_AtoB.predict(monet_generated)
show_plot(test_image_input, monet_generated, photo_reconstructed)

