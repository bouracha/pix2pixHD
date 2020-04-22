import cv2
import sys
import csv
import numpy as np

name = sys.argv[0]
epoch_num = sys.argv[1]
data_set = sys.argv[2]
num_images = sys.argv[3]

print(name)
print(epoch_num)
print(num_images)

PSNRs = []
for i in range(1, int(num_images)+1):
  print(str(i) + "/" + str(num_images), end="\r")
  #print("training_set/test_A/{0}.jpg".format(i))
  #print("results/vinci/test_latest/images/{0}_synthesized_image.jpg".format(i))
  img1 = cv2.imread(str(data_set)+"/test_B/{0}.jpg".format(i))
  img2 = cv2.imread("results/vinci/test_latest/images/{0}_synthesized_image.jpg".format(i))
  assert(img1.shape == (1024, 1024, 3))
  assert(img2.shape == (1024, 1024, 3))

  PSNRs.append(cv2.PSNR(img1, img2))

print("Epoch: ", epoch_num, np.mean(PSNRs), np.std(PSNRs))
fields=[epoch_num, np.mean(PSNRs), np.std(PSNRs)]
with open(r'PSNR/'+str(data_set)+'.csv', 'a') as f:
  writer = csv.writer(f)
  writer.writerow(fields)


