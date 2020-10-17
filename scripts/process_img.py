import os
import csv
import tensorflow as tf
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt

path_data = './dataset'
path_std_data = './standardized_data'

directories = os.listdir(path_data)
img_height = 256
img_width = 256


# Logs all of the data set into the .csv
with open('data_log.csv','w',newline = '') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',')
	writer = csv.writer(csvfile, delimiter = ',')
	for d in directories:
		if '.' not in d: # Eliminate anything that is a file
			files = os.listdir(path_data + '/' + d)
			for f in files:
				if '.jpg' in f: # Only categorize images
					writer.writerow([d] + [f])

with open('data_log.csv',newline = '') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',')
	for row in reader:
		if not os.path.isdir(os.path.join(path_std_data,row[0])): #If the standardized data directory does not exist
			mode = 0o666
			std_data_dir = os.path.join(path_std_data,row[0])
			os.mkdir(std_data_dir)
			data_dir = pathlib.Path(os.path.join(path_data,row[0])) # Find the corresponding data directory
			files = os.listdir(os.path.join(path_data,row[0]))
			for f in files:
				if '.jpg' in f:
					# Load the image and then standardize it
					try:
						print(os.path.join(data_dir,f))
						img = tf.keras.preprocessing.image.load_img(
							os.path.join(data_dir,f),grayscale = False, color_mode = 'rgb', target_size = (img_height, img_width), interpolation = "nearest")
						array = tf.keras.preprocessing.image.img_to_array(img)
						tf.keras.preprocessing.image.save_img(pathlib.Path(os.path.join(std_data_dir,f)), img, file_format = None, scale = True)
						
					except:
						print("Problem with image with directory name: " +os.path.join(data_dir,f))
