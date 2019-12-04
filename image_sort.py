import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model

model = keras.applications.VGG19(weights='imagenet', include_top=True)
feature_extractor = Model(inputs=model.input, outputs=model.get_layer('fc2').output)


import glob
import random

image_files = glob.glob('./images/*.jpg', recursive=True)
random.shuffle(image_files)
image_files = image_files[:len(image_files)]


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


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaled = ss.fit_transform(features)


import math
from sklearn.cluster import MiniBatchKMeans

sum_squared = []

K = range(1, math.floor(math.sqrt(len(image_files))*6))

for i in K:
    print('Calculating ' + str(i))
    kmeans = MiniBatchKMeans(n_clusters=i)
    kmeans.fit(scaled)
    sum_squared.append(kmeans.inertia_)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(scaled)


from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=4, metric='cosine').fit(scaled)
_, closest = neighbors.kneighbors(kmeans.cluster_centers_)


import umap.umap_ as umap

embedding = umap.UMAP().fit_transform(scaled)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(embedding)
embedding_scaled = scaler.transform(embedding)


lookup = []

for img, cluster_pos in zip(image_files, embedding_scaled):
    lookup.append({
        "filename": img.replace('./', ''),
        "cluster_pos": cluster_pos.tolist()
    })

import json

with open('image_umap_position.json', 'w') as outfile:
    json.dump(lookup, outfile)


import subprocess

subprocess.call(['./resize_images.sh'])

print('done')
