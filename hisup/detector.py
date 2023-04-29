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


class Baseline(nn.Module):
    def __init__(self, cfg, test=False):
        super(Baseline, self).__init__()

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
        
    def forward(self, target, outputs, en_features):
        if self.training:
            return self.forward_train(target, outputs, en_features)
        else:
            return self.forward_test(target, outputs, en_features)

    def forward_test(self, target, outputs, en_features):
        mask_feature = self.mask_head(en_features) # extract Fseg
        jloc_feature = self.jloc_head(en_features) # extract Fver
        afm_feature = self.afm_head(en_features) # extract Fafm

        mask_att_feature = self.a2m_att(afm_feature, mask_feature) # input Fseg into ECA-Net to output Fe_seg
        jloc_att_feature = self.a2j_att(afm_feature, jloc_feature) # input Fver into ECA-Net to output Fe_ver

        mask_pred = self.mask_predictor(mask_feature + mask_att_feature) # add Fe_seg to Fseg
        jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature) # add Fe_ver to Fver
        afm_pred = self.afm_predictor(afm_feature) #

        afm_conv = self.refuse_conv(afm_pred) # extract F*afm  
        remask_pred = self.final_conv(torch.cat((en_features, afm_conv), dim=1)) # concatenate F*afm to Fb

        joff_pred = outputs[:, :].sigmoid() - 0.5
        mask_pred = mask_pred.softmax(1)[:,1:]
        jloc_convex_pred = jloc_pred.softmax(1)[:, 2:3]
        jloc_concave_pred = jloc_pred.softmax(1)[:, 1:2]
        remask_pred = remask_pred.softmax(1)[:, 1:]

        scale_y = self.origin_height / self.pred_height
        scale_x = self.origin_width / self.pred_width

        batch_polygons, batch_masks, batch_scores, batch_juncs = [], [], [], []
        for b in range(remask_pred.size(0)):
            mask_pred_per_im = cv2.resize(remask_pred[b][0].cpu().numpy(), (self.origin_width, self.origin_height))
            juncs_pred = get_pred_junctions(jloc_concave_pred[b], jloc_convex_pred[b], joff_pred[b])
            juncs_pred[:,0] *= scale_x
            juncs_pred[:,1] *= scale_y

            if not self.test_inria:
                polys, scores = [], []
                props = regionprops(label(mask_pred_per_im > 0.5))
                for prop in props:
                    poly, juncs_sa, edges_sa, score, juncs_index = generate_polygon(prop, mask_pred_per_im, \
                                                                            juncs_pred, 0, self.test_inria)
                    if juncs_sa.shape[0] == 0:
                        continue

                    polys.append(poly)
                    scores.append(score)


                batch_scores.append(scores)
                batch_polygons.append(polys)
            
            batch_masks.append(mask_pred_per_im)
            batch_juncs.append(juncs_pred)

        output = {
            'polys_pred': batch_polygons,
            'mask_pred': batch_masks,
            'scores': batch_scores,
            'juncs_pred': batch_juncs
        }
        return output

    
    def forward_train(self, target, outputs, en_features):
        mask_feature = self.mask_head(en_features) # extract Fseg
        jloc_feature = self.jloc_head(en_features) # extract Fver
        afm_feature = self.afm_head(en_features) # extract Fafm        

        mask_att_feature = self.a2m_att(afm_feature, mask_feature) # input Fseg into ECA-Net to output Fe_seg
        jloc_att_feature = self.a2j_att(afm_feature, jloc_feature) # input Fver into ECA-Net to output Fe_ver

        # print(mask_att_feature.shape)
        # print(jloc_att_feature.shape)

        mask_pred = self.mask_predictor(mask_feature + mask_att_feature) # add Fe_seg to Fseg
        jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature) # add Fe_ver to Fver
        afm_pred = self.afm_predictor(afm_feature) #

        # print(mask_pred.shape)
        # print(jloc_pred.shape)

        afm_conv = self.refuse_conv(afm_pred) # extract F*afm  
        remask_pred = self.final_conv(torch.cat((en_features, afm_conv), dim=1)) # concatenate F*afm to Fb

        if target is not None:
            loss_dict = {
                'loss_jloc': self.junc_loss(jloc_pred, target['jloc'].squeeze(dim=1)),
                'loss_joff': sigmoid_l1_loss(outputs[:, :], target['joff'], -0.5, target['jloc']),
                'loss_mask': F.cross_entropy(mask_pred, target['mask'].squeeze(dim=1).long()),
                'loss_afm': F.l1_loss(afm_pred, target['afmap']),
                'loss_remask': F.cross_entropy(remask_pred, target['mask'].squeeze(dim=1).long())
            }
        else:
            loss_dict = {}
        return loss_dict

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


class BuildingDetector(nn.Module):
    def __init__(self, cfg, test=False):
        super(BuildingDetector, self).__init__()

        self.test_inria = 'inria' in cfg.DATASETS.TEST[0]
        self.training = False
        if not test:
            # import sys
            # sys.path.append("..")
            from hisup.encoder import Encoder
            self.encoder = Encoder(cfg)
            self.training = True
        self.backbone = build_backbone(cfg)
        self.backbone_name = cfg.MODEL.NAME

        dim_in = cfg.MODEL.OUT_FEATURE_CHANNELS
        self.cross = CrossGeoBlock(dim=dim_in)

        self.train_step = 0
        self.baseline = Baseline(cfg, test)
        
    def forward(self, image_a,image_b, annotation_a = None, annotation_b = None):
        if self.training:
            # print("hi")
            return self.forward_train(image_a,image_b, annotation_a, annotation_b)
        else:
            return self.forward_test(image_a, image_b, annotation_a, annotation_b)

    def forward_test(self, image_a, image_b, annotation_a = None, annotation_b = None):
        outputs_a, features_a = self.backbone(image_a) # extract Fb
        outputs_b, features_b = self.backbone(image_b)

        en_features_a, en_features_b = self.cross(features_a, features_b)

        output_a = self.baseline(outputs_a, en_features_a)
        output_b = self.baseline(outputs_b, en_features_b)

        output = {}
        for key in output_a.keys():
            output[key] = (output_a[key], output_b[key])
        extra_info = {}

        return output, extra_info

    def forward_train(self, image_a, image_b, annotation_a = None, annotation_b = None):
        # print(type(image_a), type(image_b), type(annotation_a), type(annotation_b))
        self.train_step += 1

        # print(annotation_a.keys())
        target_a, _ = self.encoder(annotation_a)
        target_b, _ = self.encoder(annotation_b)
        outputs_a, features_a = self.backbone(image_a)
        outputs_b, features_b = self.backbone(image_b)

        en_features_a, en_features_b = self.cross(features_a,features_b)

        loss_dict_a = self.baseline(target_a, outputs_a, en_features_a)
        loss_dict_b = self.baseline(target_b, outputs_b, en_features_b)

        loss_dict = {}
        alpha, beta = 1.0, 0.00001  # adjust this value for `target_a` and `target_b` loss weight
        for key in loss_dict_a.keys():
            loss_dict[key] = alpha*loss_dict_a[key] + beta*loss_dict_b[key]
        extra_info = {}

        return loss_dict, extra_info


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
