#!/bin/bash

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -p|--project)
    PROJECT="$2"
    shift # past argument
    shift # past value
    ;;
    -d|--dataset)
    DATASET="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done


cd ../

for EPOCH_NUM in 10 20 30 40 50 60 70 80 90 100 100 120 130 140 150 160 170 180 190 200
do
  echo "Data Set $DATASET, Epoch Number $EPOCH_NUM"

  echo "Copying new checkpoints..."
  cp checkpoints/"$PROJECT"/"$EPOCH_NUM"_net_D.pth checkpoints/"$PROJECT"/latest_net_D.pth
  cp checkpoints/"$PROJECT"/"$EPOCH_NUM"_net_G.pth checkpoints/"$PROJECT"/latest_net_G.pth

  echo "Deleting old results..."
  rm -r results/
  NUM_IMAGES=$(ls datasets/$DATASET/test_A |grep -v / | wc -l)
  echo "Running inference on $DATASET.. ($NUM_IMAGES images)"
  python3 test.py --name $PROJECT --label_nc 0 --no_instance --loadSize 1024 --how_many $NUM_IMAGES --dataroot ./datasets/$DATASET
  echo "Calculating PSNR on $DATASET, $EPOCH_NUM.."
  python3 PSNR/psnr.py "$EPOCH_NUM" "$DATASET" $NUM_IMAGES $PROJECT

done
