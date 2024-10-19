import torch.nn.functional as F
from tools.visual_utils.visual_bev import vis_pred_label_bev
from .detector3d_template import Detector3DTemplate
from pcdet.models.backbones_2d import Unet


def meandice(pred, label):
    sumdice = 0
    smooth = 1e-6
    for i in range(1, 2):
        pred_bin = (pred == i) * 1
        label_bin = (label == i) * 1
        pred_bin = pred_bin.contiguous().view(pred_bin.shape[0], -1)
        label_bin = label_bin.contiguous().view(label_bin.shape[0], -1)
        intersection = (pred_bin * label_bin).sum()
        dice = (2. * intersection + smooth) / (pred_bin.sum() + label_bin.sum() + smooth)
        sumdice += dice
    return sumdice / 4

def dice_loss(pred,target,smooth=1.):
    num = pred.shape[0]
    pred = F.sigmoid(pred)
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum()
    score = (2. * intersection + smooth)/(pred.sum()+target.sum()+smooth)
    loss = (1-score.sum()/num)

    return loss


class PillarSeg(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        # self.Unet = Unet.UNet(n_channels=64, n_classes=4)
        # self.seg_loss = F.cross_entropy()
        # self.save_path = '/home/rpf/Pillarseg/OpenPCDet/output/' + model_cfg['NAME'].lower() + '/' + model_cfg['extra_tag']
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # batch_dict['logit'] = self.Unet(batch_dict['spatial_features'])


        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # return pred_dicts, recall_dicts


            # run demo need #####
            # save_out = ''
            frame_id = batch_dict['frame_id']
            seg_dicts = F.softmax(batch_dict['logit'],dim=1).argmax(dim=1)
            label_dicts = batch_dict['voxel_label']
            # vis_pred_label_bev(seg_dicts,label_dicts,self.save_path,frame_id)
            return seg_dicts, label_dicts, pred_dicts, recall_dicts

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        logit = batch_dict['logit']
        # pred_dicts = F.softmax(batch_dict['logit'], dim=1).argmax(dim=1)
        # voxel_label = Unet.get_one_hot(target,4)
        loss_rpn, tb_dict = self.dense_head.get_loss()

        #==========dice loss ===========#
        # loss_dice = 1 - meandice(pred_dicts,target)
        if logit.shape[1]==1:
            # 单类别loss
            target = batch_dict['voxel_label']
            seg_loss_rpn = F.binary_cross_entropy_with_logits(logit.squeeze(1), target)
            # seg_dice_loss = dice_loss(logit.squeeze(1), target)
        else:
            # 多类别loss
            target = batch_dict['voxel_label'].long()
            seg_loss_rpn = F.cross_entropy(logit, target)


        # ,tb_dict
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            # **tb_dict
        }
        loss = loss_rpn + seg_loss_rpn
        # loss = loss_rpn + seg_loss_rpn + 0.5 * seg_dice_loss
        return loss, tb_dict, disp_dict
