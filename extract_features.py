import cv2
import numpy as np
import scipy
from skimage.io import imread
import pickle
import random
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle

def extract_features(image_path, vector_size=64):
    image = imread(image_path)
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None

    return dsc

def build_file(index_dict, clusters, outpath):
	with open(outpath, 'w') as f:
		for picture in index_dict:
			indices = index_dict[picture]
			words = clusters[indices[0]:indices[1]]
			words_dict = dict()
			for item in words:
				if item in words_dict:
					words_dict[item] += 1
				else:
					words_dict[item] = 1

			picture_line = '{} '.format(len(words_dict))
			for word in words_dict:
				picture_line = '{}{}:{} '.format(picture_line, word, words_dict[word])

			picture_line = picture_line[:-1] + '\n'
			f.write(picture_line)

if __name__ == '__main__':
	index_dict = dict()
	vecs_list = []
	start_index = 0
	i = 0
	SEGMENT_PATH = './vocabulary/'
	LIST_OUTPATH = 'vecs.pkl'
	ORDER_OUTPATH = 'order.pkl'
	'''
	for folder in os.listdir(SEGMENT_PATH):
		if os.path.isdir('{}/{}'.format(SEGMENT_PATH, folder)):
			for file in os.listdir('{}/{}'.format(SEGMENT_PATH, folder)):
				if file[-3:] == 'jpg':
					i += 1
					vecs_list.append(extract_features('{}/{}/{}'.format(SEGMENT_PATH, folder, file)))
					print(i)
			index_dict[folder] = (start_index, i)
			start_index = i
	'''
	order_dict = dict()
	i = 0
	for folder in os.listdir(SEGMENT_PATH):
		if os.path.isdir('{}/{}'.format(SEGMENT_PATH, folder)):
			order_dict[i] = folder
			i += 1
	'''
	with open(LIST_OUTPATH, 'wb') as f:
		pickle.dump(vecs_list, f)	
	'''
	with open(ORDER_OUTPATH, 'wb') as f:
		pickle.dump(order_dict, f)	

	'''
	print('ready')
	km = KMeans(n_clusters = 50)


	clusters = km.fit_predict(vecs_list)
	build_file(index_dict, clusters, 'hlda_data.dat')
	'''


