import torch
# filename = '/mnt/data/seg_det/result/pillarseg/pillarseg_radar_4class_celoss_fusion_attn_bs12/ckpt/checkpoint_epoch_97.pth'
filename = '/home/rpf/Pillarseg/OpenPCDet/output/pillarseg/pillarseg_radar_4class_celoss_fusion_attn_nobnrelu_bs16/ckpt/checkpoint_epoch_71.pth'
checkpoint = torch.load(filename)
model_state_disk = checkpoint['model_state']
print('debug')