import cv2
import torch
import torch.nn.functional as F

from math import log
from torch import nn
from hisup.backbones import build_backbone
from hisup.utils.polygon import generate_polygon
from hisup.utils.polygon import get_pred_junctions
from skimage.measure import label, regionprops

from hisup.cross.cgb import CrossGeoBlock




def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)

    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])

    return loss.mean()

def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        t = ((mask == 1) | (mask == 2)).float()
        w = t.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(t/w)

    return loss.mean()

# Copyright (c) 2019 BangguWu, Qilong Wang
# Modified by Bowen Xu, Jiakun Xu, Nan Xue and Gui-song Xia
class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        C = channel

        t = int(abs((log(C, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        y = self.avg_pool(x1 + x2)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1 ,-2).unsqueeze(-1)
        y = self.sigmoid(y)

        out = self.out_conv(x2 * y.expand_as(x2))
        return out


class BuildingDetector(nn.Module):
    def __init__(self, cfg, test=False):
        super(BuildingDetector, self).__init__()
        self.backbone = build_backbone(cfg)
        self.backbone_name = cfg.MODEL.NAME
        self.junc_loss = nn.CrossEntropyLoss()
        self.test_inria = 'inria' in cfg.DATASETS.TEST[0]
        self.training = False
        if not test:
            # import sys
            # sys.path.append("..")
            from hisup.encoder import Encoder
            self.encoder = Encoder(cfg)
            self.training = True

        self.pred_height = cfg.DATASETS.TARGET.HEIGHT
        self.pred_width = cfg.DATASETS.TARGET.WIDTH
        self.origin_height = cfg.DATASETS.ORIGIN.HEIGHT
        self.origin_width = cfg.DATASETS.ORIGIN.WIDTH

        dim_in = cfg.MODEL.OUT_FEATURE_CHANNELS

        self.cross = CrossGeoBlock(dim=dim_in, num_heads=8, mlp_ratio=4,
                                qkv_bias=False, drop=0,
                                drop_path=0, act_layer=nn.ReLU6, window_size=16,
                                norm_layer=nn.BatchNorm2d)

        self.mask_head = self._make_conv(dim_in, dim_in, dim_in)
        self.jloc_head = self._make_conv(dim_in, dim_in, dim_in)
        self.afm_head = self._make_conv(dim_in, dim_in, dim_in)

        self.a2m_att = ECA(dim_in)
        self.a2j_att = ECA(dim_in)

        self.mask_predictor = self._make_predictor(dim_in, 2)
        self.jloc_predictor = self._make_predictor(dim_in, 3)
        self.afm_predictor = self._make_predictor(dim_in, 2)

        self.refuse_conv = self._make_conv(2, dim_in//2, dim_in)
        self.final_conv = self._make_conv(dim_in*2, dim_in, 2)

        self.train_step = 0
        
    def forward(self, image_a,image_b, annotation_a = None, annotation_b = None):
        if self.training:
            # print("hi")
            return self.forward_train(image_a,image_b, annotation_a, annotation_b)
        else:
            return self.forward_test(image_a, image_b, annotation_a, annotation_b)

    def forward_test(self, image_a,image_b, annotation_a = None, annotation_b = None):
        device = image_a.device
        outputs_a, features_a = self.backbone(image_a) # extract Fb
        outputs_b, features_b = self.backbone(image_b)

        en_features_a = self.cross(features_a,features_b)
        en_features_b = self.cross(features_b,features_a)


        mask_feature_a = self.mask_head(en_features_a) # extract Fseg
        jloc_feature_a = self.jloc_head(en_features_a) # extract Fver
        afm_feature_a = self.afm_head(en_features_a) # extract Fafm

        mask_feature_b = self.mask_head(en_features_b) # extract Fseg
        jloc_feature_b = self.jloc_head(en_features_b) # extract Fver
        afm_feature_b = self.afm_head(en_features_b) # extract Fafm


        mask_att_feature_a = self.a2m_att(afm_feature_a, mask_feature_a) # input Fseg into ECA-Net to output Fe_seg
        jloc_att_feature_a = self.a2j_att(afm_feature_a, jloc_feature_a) # input Fver into ECA-Net to output Fe_ver

        mask_att_feature_b = self.a2m_att(afm_feature_b, mask_feature_b) # input Fseg into ECA-Net to output Fe_seg
        jloc_att_feature_b = self.a2j_att(afm_feature_b, jloc_feature_b) # input Fver into ECA-Net to output Fe_ver



        mask_pred_a = self.mask_predictor(mask_feature_a + mask_att_feature_a) # add Fe_seg to Fseg
        jloc_pred_a = self.jloc_predictor(jloc_feature_a + jloc_att_feature_a) # add Fe_ver to Fver
        afm_pred_a = self.afm_predictor(afm_feature_a) # 

        mask_pred_b = self.mask_predictor(mask_feature_b + mask_att_feature_b) # add Fe_seg to Fseg
        jloc_pred_b = self.jloc_predictor(jloc_feature_b + jloc_att_feature_b) # add Fe_ver to Fver
        afm_pred_b = self.afm_predictor(afm_feature_b) # 


        afm_conv_a = self.refuse_conv(afm_pred_a) # extract F*afm  
        remask_pred_a = self.final_conv(torch.cat((en_features_a, afm_conv_a), dim=1)) # concatenate F*afm to Fb

        afm_conv_b = self.refuse_conv(afm_pred_b) # extract F*afm  
        remask_pred_b = self.final_conv(torch.cat((en_features_b, afm_conv_b), dim=1)) # concatenate F*afm to Fb


        joff_pred_a = outputs_a[:, :].sigmoid() - 0.5
        mask_pred_a = mask_pred_a.softmax(1)[:,1:]
        jloc_convex_pred_a = jloc_pred_a.softmax(1)[:, 2:3]
        jloc_concave_pred_a = jloc_pred_a.softmax(1)[:, 1:2]
        remask_pred_a = remask_pred_a.softmax(1)[:, 1:]

        joff_pred_b = outputs_b[:, :].sigmoid() - 0.5
        mask_pred_b = mask_pred_b.softmax(1)[:,1:]
        jloc_convex_pred_b = jloc_pred_b.softmax(1)[:, 2:3]
        jloc_concave_pred_b = jloc_pred_b.softmax(1)[:, 1:2]
        remask_pred_b = remask_pred_b.softmax(1)[:, 1:]


        scale_y = self.origin_height / self.pred_height
        scale_x = self.origin_width / self.pred_width

        batch_polygons = [[],[]]
        batch_masks = [[],[]]
        batch_scores = [[],[]]
        batch_juncs = [[],[]]

        for b in range(remask_pred_a.size(0)):
            mask_pred_per_im_a = cv2.resize(remask_pred_a[b][0].cpu().numpy(), (self.origin_width, self.origin_height))
            juncs_pred_a = get_pred_junctions(jloc_concave_pred_a[b], jloc_convex_pred_a[b], joff_pred_a[b])
            juncs_pred_a[:,0] *= scale_x
            juncs_pred_a[:,1] *= scale_y

            mask_pred_per_im_b = cv2.resize(remask_pred_b[b][0].cpu().numpy(), (self.origin_width, self.origin_height))
            juncs_pred_b = get_pred_junctions(jloc_concave_pred_b[b], jloc_convex_pred_b[b], joff_pred_b[b])
            juncs_pred_b[:,0] *= scale_x
            juncs_pred_b[:,1] *= scale_y

            if not self.test_inria:
                polys_a, scores_a = [], []
                polys_b, scores_b = [], []
                props_a = regionprops(label(mask_pred_per_im_a > 0.5))
                props_b = regionprops(label(mask_pred_per_im_b > 0.5))
                for prop in props_a:
                    poly, juncs_sa, edges_sa, score, juncs_index = generate_polygon(prop, mask_pred_per_im_a, \
                                                                            juncs_pred_a, 0, self.test_inria)
                    if juncs_sa.shape[0] == 0:
                        continue

                    polys_a.append(poly)
                    scores_a.append(score)

                for prop in props_b:
                    poly, juncs_sa, edges_sa, score, juncs_index = generate_polygon(prop, mask_pred_per_im_b, \
                                                                            juncs_pred_b, 0, self.test_inria)
                    if juncs_sa.shape[0] == 0:
                        continue

                    polys_b.append(poly)
                    scores_b.append(score)


                batch_scores[0].append(scores_a)
                batch_polygons[0].append(polys_a)

                batch_scores[1].append(scores_b)
                batch_polygons[1].append(polys_b)
            
            batch_masks[0].append(mask_pred_per_im_a)
            batch_juncs[0].append(juncs_pred_a)

            batch_masks[1].append(mask_pred_per_im_b)
            batch_juncs[1].append(juncs_pred_b)

        extra_info = {}
        output = {
            'polys_pred': batch_polygons,
            'mask_pred': batch_masks,
            'scores': batch_scores,
            'juncs_pred': batch_juncs
        }
        return output, extra_info

    def forward_train(self, image_a,image_b, annotation_a = None, annotation_b = None):
        self.train_step += 1

        device_a = image_a.device
        target_a, metas = self.encoder(annotation_a)
        target_b, metas = self.encoder(annotation_b)
        outputs_a, features_a = self.backbone(image_a) # extract Fb
        outputs_b, features_b = self.backbone(image_b)

        en_features_a = self.cross(features_a)
        en_features_b = self.cross(features_b)


        loss_dict = {
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_mask': 0.0,
            'loss_afm' : 0.0,
            'loss_remask': 0.0
        }

        mask_feature_a = self.mask_head(en_features_a) # extract Fseg
        jloc_feature_a = self.jloc_head(en_features_a) # extract Fver
        afm_feature_a = self.afm_head(en_features_a) # extract Fafm

        mask_feature_b = self.mask_head(en_features_b) # extract Fseg
        jloc_feature_b = self.jloc_head(en_features_b) # extract Fver
        afm_feature_b = self.afm_head(en_features_b) # extract Fafm

        

        mask_att_feature_a = self.a2m_att(afm_feature_a, mask_feature_a) # input Fseg into ECA-Net to output Fe_seg
        jloc_att_feature_a = self.a2j_att(afm_feature_a, jloc_feature_a) # input Fver into ECA-Net to output Fe_ver

        mask_att_feature_b = self.a2m_att(afm_feature_b, mask_feature_b) # input Fseg into ECA-Net to output Fe_seg
        jloc_att_feature_b = self.a2j_att(afm_feature_b, jloc_feature_b) # input Fver into ECA-Net to output Fe_ver

        # print(mask_att_feature.shape)
        # print(jloc_att_feature.shape)

        mask_pred_a = self.mask_predictor(mask_feature_a + mask_att_feature_a) # add Fe_seg to Fseg
        jloc_pred_a = self.jloc_predictor(jloc_feature_a + jloc_att_feature_a) # add Fe_ver to Fver
        afm_pred_a = self.afm_predictor(afm_feature_a) # 

        mask_pred_b = self.mask_predictor(mask_feature_b + mask_att_feature_b) # add Fe_seg to Fseg
        jloc_pred_b = self.jloc_predictor(jloc_feature_b + jloc_att_feature_b) # add Fe_ver to Fver
        afm_pred_b = self.afm_predictor(afm_feature_b) # 

        # print(mask_pred.shape)
        # print(jloc_pred.shape)

        afm_conv_a = self.refuse_conv(afm_pred_a) # extract F*afm  
        remask_pred_a = self.final_conv(torch.cat((en_features_a, afm_conv_a), dim=1)) # concatenate F*afm to Fb

        afm_conv_b = self.refuse_conv(afm_pred_b) # extract F*afm  
        remask_pred_b = self.final_conv(torch.cat((en_features_b, afm_conv_b), dim=1)) # concatenate F*afm to Fb

        alpha = 1.0  # adjust this value for target_a loss weight
        beta = 0.00001  # adjust this value for target_b loss weight

        if target_a is not None:
            loss_dict['loss_jloc'] += alpha* self.junc_loss(jloc_pred_a, target_a['jloc'].squeeze(dim=1))
            loss_dict['loss_joff'] += alpha* sigmoid_l1_loss(outputs_a[:, :], target_a['joff'], -0.5, target_a['jloc'])
            loss_dict['loss_mask'] += alpha* F.cross_entropy(mask_pred_a, target_a['mask'].squeeze(dim=1).long())
            loss_dict['loss_afm'] += alpha* F.l1_loss(afm_pred_a, target_a['afmap'])
            loss_dict['loss_remask'] += alpha* F.cross_entropy(remask_pred_a, target_a['mask'].squeeze(dim=1).long())

        if target_b is not None:
            loss_dict['loss_jloc'] += beta* self.junc_loss(jloc_pred_b, target_b['jloc'].squeeze(dim=1))
            loss_dict['loss_joff'] += beta* sigmoid_l1_loss(outputs_b[:, :], target_b['joff'], -0.5, target_b['jloc'])
            loss_dict['loss_mask'] += beta* F.cross_entropy(mask_pred_b, target_b['mask'].squeeze(dim=1).long())
            loss_dict['loss_afm'] += beta* F.l1_loss(afm_pred_b, target_b['afmap'])
            loss_dict['loss_remask'] += beta* F.cross_entropy(remask_pred_b, target_b['mask'].squeeze(dim=1).long())

        
        extra_info = {}

        return loss_dict, extra_info
    
    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        return layer

    def _make_predictor(self, dim_in, dim_out):
        m = int(dim_in / 4)
        layer = nn.Sequential(
                    nn.Conv2d(dim_in, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, dim_out, kernel_size=1),
                )
        return layer


def get_pretrained_model(cfg, dataset, device, pretrained=True):
    PRETRAINED = {
        'crowdai': 'https://github.com/XJKunnn/pretrained_model/releases/download/pretrained_model/crowdai_hrnet48_e100.pth',
        'inria': 'https://github.com/XJKunnn/pretrained_model/releases/download/pretrained_model/inria_hrnet48_e5.pth'
    }

    model = BuildingDetector(cfg, test=True)
    if pretrained:
        url = PRETRAINED[dataset]
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device, progress=True)
        state_dict = {k[7:]:v for k,v in state_dict['model'].items() if k[0:7] == 'module.'}
        model.load_state_dict(state_dict)
        model = model.eval()
        return model
    return model
