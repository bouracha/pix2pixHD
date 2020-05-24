import cv2
import sys
import csv
import numpy as np
import os

name = sys.argv[0]
epoch_num = sys.argv[1]
data_set = sys.argv[2]
num_images = sys.argv[3]
project = sys.argv[4]

image_list = os.listdir("datasets/"+str(data_set)+"/test_B")
PSNRs = []
for i in image_list:
  print(str(i) + "/" + str(num_images), end="\r")
  name = i[:-4]
  print("datasets/"+str(data_set)+"/test_B/{0}")
  img1 = cv2.imread("datasets/"+str(data_set)+"/test_B/{0}".format(i))
  img2 = cv2.imread("results/"+str(project)+"/test_latest/images/{0}_synthesized_image.jpg".format(name))
  assert(img1.shape == (1024, 1024, 3))
  assert(img2.shape == (1024, 1024, 3))

  PSNRs.append(cv2.PSNR(img1, img2))

print("Data Set: ", data_set)
print("Epoch: ", epoch_num)
print("PSNR: ", np.mean(PSNRs), "+-", np.std(PSNRs))
fields=[epoch_num, np.mean(PSNRs), np.std(PSNRs)]
with open('PSNR/'+str(data_set)+'.csv', 'a') as f:
  writer = csv.writer(f)
  writer.writerow(fields)


