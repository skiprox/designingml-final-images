# We're gonna import our tensorflow stuff up here
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model

model = keras.applications.VGG19(weights='imagenet', include_top=True)
feature_extractor = Model(inputs=model.input, outputs=model.get_layer('fc2').output)


# Now we're bringing the images in, randomizing the order (for no real reason)
import glob
import random

image_files = glob.glob('./images/*.jpg', recursive=True)
random.shuffle(image_files)
image_files = image_files[:len(image_files)]


# Creating an array of features
import numpy as np
features = []

for i, image_path in enumerate(image_files):
	if i % 10 == 0:
		print("analyzed " + str(i) + " out of " + str(len(image_files)))
	img = image.load_img(image_path, target_size=model.input_shape[1:3])
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	feat = feature_extractor.predict(x)[0]
	features.append(feat)


# Importing standardscaler and creating a scaled array of features
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaled = ss.fit_transform(features)


# Creating an embedding from the scaled features
import umap.umap_ as umap

embedding = umap.UMAP().fit_transform(scaled)


# Scaling the embedding
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(embedding)
embedding_scaled = scaler.transform(embedding)


# Create an array of arrays of similar images,
# for every image in our image_files array
from sklearn.metrics.pairwise import cosine_similarity
cosine_similar_imgs = cosine_similarity(scaled)

similar_img_arr = []

for i, similar_imgs in enumerate(cosine_similar_imgs):
	if i % 10 == 0:
		print("finding similar images, done " + str(i) + " out of " + str(len(image_files)))
	similar_img_arr.append([(sim, image_files[i][2:]) for j, sim in enumerate(similar_imgs)])

top_similar_imgs = [sorted(k, reverse=True)[1:6] for k in similar_img_arr]


# Creating a JSON file
lookup = []

for i, (img, cluster_pos, closest_imgs) in enumerate(zip(image_files, embedding_scaled, top_similar_imgs)):
	if i % 10 == 0:
		print("creating JSON, done " + str(i) + " out of " + str(len(image_files)))
	lookup.append({
		"filename": img.replace('./', ''),
		"cluster_pos": cluster_pos.tolist(),
		"closest_imgs": closest_imgs
	})

import json

with open('image_umap_position.json', 'w') as outfile:
	json.dump(lookup, outfile)


# Subprocess to resize images
import subprocess

print("Starting subprocess, to resize images")
subprocess.call(['./resize_images.sh'])


# We're done!!
print('done')
