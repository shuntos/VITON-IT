import os
import sys
import shutil


out_folder = "/home/santoshadhikari/project/Virtual_Dressing/dataset/images/"
input_folder = "/home/santoshadhikari/project/Virtual_Dressing/dataset/masks/"



for file in os.listdir(input_folder):
	filename,ext = os.path.splitext(file)

	tar_path = out_folder+file

	if not os.path.exists(tar_path):
		print("==", file)