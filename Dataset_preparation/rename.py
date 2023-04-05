import os 
import sys
import shutil

img_folder  = "sorted_2"

out_folder = img_folder+"_"+ "renamed"
if not os.path.exists(out_folder):
	os.makedirs(out_folder)

c= 1100
for file in os.listdir(img_folder):
	source = os.path.join(img_folder, file)
	print(source)
	_, ext = os.path.splitext(file)
	print(ext)
	outfile = str(c)+ ext

	target = os.path.join(out_folder, outfile)

	print(target)
	shutil.copy(source, target)
	c +=1

