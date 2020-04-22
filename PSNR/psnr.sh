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

for EPOCH_NUM in 130 140 150 160 170
do
  echo "Data Set $DATASET, Epoch Number $EPOCH_NUM"

  echo "Copying new checkpoints..."
  cp checkpoints/"$PROJECT"/"$EPOCH_NUM"_net_D.pth checkpoints/vinci/latest_net_D.pth
  cp checkpoints/"$PROJECT"/"$EPOCH_NUM"_net_G.pth checkpoints/vinci/latest_net_G.pth

  echo "Deleting old results..."
  rm -r results/
  echo "$DATASET/test_A"
  NUM_IMAGES=$(ls $DATASET/test_A |grep -v / | wc -l)
  echo "Running inference on $DATASET.. ($NUM_IMAGES images)"
  python test.py --name $PROJECT --label_nc 0 --no_instance --loadSize 1024 --how_many $NUM_IMAGES --dataroot ./$DATASET
  echo "Calculating PSNR on $DATASET, $EPOCH_NUM.."
  python PSNR/psnr.py "$EPOCH_NUM" "$DATASET" $NUM_IMAGES $PROJECT

done
