# Commands 

python3 data/prepare_train_data.py /data/KITTI-raw-data/raw_data_KITTI/ --dataset-format 'kitti' --dump-root formatted_data/ --width 416 --height 128 --num-threads 4

python3 train.py formatted_data/ -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 3 --log-output


# train.py

## train_set

数据集的处理：
* 每个文件夹中放一个 scene, 每个 scene 中包括一个共同的相机内参和一组图片
* scene 中的图片被组成3个一组的 sequence
* 对每张图片进行以下变换
  - RandomHorizontalFlip: 随机图片翻转, 同时更改内参的 cx
  - RandomScaleCrop: scale 到 1-1.5 , 再 crop 到原始大小
  - ArrayToTensor: 每个 RGB 色值 / 255
  - normalize: mean=0.5, std=0.5

## val_set

* valid_transform
  - ArrayToTensor
  - normalize

## net

disp_net
* models.DispNets
* models.PoseExpNet

## optimizer

[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

## train

w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight

torch.nn.init.xavier_uniform_(tensor, gain=1.0)
   Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010)

## valid

* with_gt
* without_gt

# 训练时间

$ python3 train.py formatted_data/ -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 3 --log-output
=> will save everything to checkpoints/formatted_data,epoch_size3000,m0.2/10-27-13:53
=> fetching scenes in 'formatted_data/'
32218 samples found in 64 train scenes
12010 samples found in 8 valid scenes
=> creating model
=> setting adam solver


100% (200 of 200) |##################################################################################################################################################| Elapsed Time: 18:02:11 Time: 18:02:11

 * Avg Loss : 0.405
100% (3000 of 3000) |#################################################################################################################################################| Elapsed Time: 0:04:10 ETA:  00:00:00
 93% (2796 of 3000) |###############################################################################      | Elapsed Time: 0:03:54 ETA:   0:00:17
 * Avg Validation Total loss : 0.166, Validation Photo loss : 0.147, Validation Exp loss : 0.092
100% (3003 of 3003) |#################################################################################################################################################| Elapsed Time: 0:01:12 ETA:  00:00:00
 * Avg Validation Total loss : 0.167, Validation Photo loss : 0.145, Validation Exp loss : 0.112

