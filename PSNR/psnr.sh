#!bin/bash

cd ../

for EPOCH_NUM in 130 140 150 160 170
#EPOCH_NUM=170
do
  echo $EPOCH_NUM

  echo "Copying new checkpoints..."
  cp checkpoints/vinci/"$EPOCH_NUM"_net_D.pth checkpoints/vinci/latest_net_D.pth
  cp checkpoints/vinci/"$EPOCH_NUM"_net_G.pth checkpoints/vinci/latest_net_G.pth

  #echo "Deleting old results..."
  #rm -r results/
  #echo "Running inference on train images.."
  #python test.py --name vinci --label_nc 0 --no_instance --loadSize 1024 --how_many 3000 --dataroot ./training_set
  #echo "Calculating PSNR on train"
  #python PSNR/psnr.py "$EPOCH_NUM" "training_set" 2761

  echo "Deleting old results..."
  rm -r results/
  echo "Running inference on test images.."
  python test.py --name vinci --label_nc 0 --no_instance --loadSize 1024 --how_many 3000 --dataroot ./test_set
  echo "Calculating PSNR on test"
  python PSNR/psnr.py "$EPOCH_NUM" "test_set" 2

  echo "Deleting old results..."
  rm -r results/
  echo "Running inference on labelled test images.."
  python test.py --name vinci --label_nc 0 --no_instance --loadSize 1024 --how_many 3000 --dataroot ./test_set_labelled
  echo "Calculating PSNR on test"
  python PSNR/psnr.py "$EPOCH_NUM" "test_set_labelled" 2
done
