import cv2
import sys
import csv
import numpy as np

name = sys.argv[0]
epoch_num = sys.argv[1]
data_set = sys.argv[2]
num_images = sys.argv[3]
project = sys.argv[4]

PSNRs = []
for i in range(1, int(num_images)+1):
  print(str(i) + "/" + str(num_images), end="\r")
  img1 = cv2.imread("datasets/"+str(data_set)+"/test_B/{0}.jpg".format(i))
  img2 = cv2.imread("results/"+str(project)+"/test_latest/images/{0}_synthesized_image.jpg".format(i))
  assert(img1.shape == (1024, 1024, 3))
  assert(img2.shape == (1024, 1024, 3))

  PSNRs.append(cv2.PSNR(img1, img2))

with open('PSNR/'+str(data_set)+'.csv', "w") as my_empty_csv:
  pass
print("Data Set: ", data_set)
print("Epoch: ", epoch_num)
print("PSNR: ", np.mean(PSNRs), "+-", np.std(PSNRs))
fields=[epoch_num, np.mean(PSNRs), np.std(PSNRs)]
with open(r'PSNR/'+str(data_set)+'.csv', 'a') as f:
  writer = csv.writer(f)
  writer.writerow(fields)


