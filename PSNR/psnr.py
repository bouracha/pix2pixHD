import cv2
import sys
import csv
import numpy as np
import os
from image import *

name = sys.argv[0]
epoch_num = sys.argv[1]
data_set = sys.argv[2]
num_images = sys.argv[3]
project = sys.argv[4]

image_list = os.listdir("datasets/"+str(data_set)+"/test_B")
PSNRs = []
for i in image_list:
  print(str(i) + "/" + str(num_images), end="\r")
  img1 = IMAGE(path_to_image="datasets/" + str(data_set) + "/test_B", name_of_image=str(i))
  name = i[:-4]
  img1 = IMAGE(path_to_image="results/" + str(project) + "/test_latest/images",
               name_of_image="{0}_synthesized_image.jpg".format(name))
  if (img1.valid_image == False) or (img2.valid_image == False):
    continue
  assert(img1.shape() == img2.shape())

  PSNRs.append(cv2.PSNR(img1.image, img2.image))

print("Data Set: ", data_set)
print("Epoch: ", epoch_num)
print("PSNR: ", np.mean(PSNRs), "+-", np.std(PSNRs))
fields=[epoch_num, np.mean(PSNRs), np.std(PSNRs)]
with open('PSNR/'+str(data_set)+'.csv', 'a') as f:
  writer = csv.writer(f)
  writer.writerow(fields)


