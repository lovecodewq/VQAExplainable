import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import ResNet101_Weights
from torchvision.models.detection._utils import BoxCoder
from fast_rcnn.config import cfg

from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class BottomUpAttention(nn.Module):
    def __init__(self, num_objects=1601, num_attributes=401, embed_dim=300):
        super().__init__()
        # Backbone: conv4 & conv5
        resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.backbone = IntermediateLayerGetter(
            resnet, return_layers={'layer3':'0','layer4':'1'}
        )

        # RPN on conv4
        anchor_gen = AnchorGenerator(
            sizes=((128,256,512),), aspect_ratios=((0.5,1.0,2.0),)
        )
        rpn_head = RPNHead(in_channels=1024, num_anchors=9)
        self.rpn = RegionProposalNetwork(
            anchor_gen, rpn_head,
            fg_iou_thresh=cfg.TRAIN.RPN_POSITIVE_OVERLAP,
            bg_iou_thresh=cfg.TRAIN.RPN_NEGATIVE_OVERLAP,
            batch_size_per_image=cfg.TRAIN.RPN_BATCHSIZE,
            positive_fraction=cfg.TRAIN.RPN_FG_FRACTION,
            pre_nms_top_n={
                'training': cfg.TRAIN.RPN_PRE_NMS_TOP_N,
                'testing':  cfg.TEST.RPN_PRE_NMS_TOP_N
            },
            post_nms_top_n={
                'training': cfg.TRAIN.RPN_POST_NMS_TOP_N,
                'testing':  cfg.TEST.RPN_POST_NMS_TOP_N
            },
            nms_thresh=cfg.TRAIN.RPN_NMS_THRESH
        )
        # RoIAlign on conv5
        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=['1'], output_size=14, sampling_ratio=0
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        # Detection heads
        self.cls_score = nn.Linear(2048, num_objects)
        self.bbox_pred = nn.Linear(2048, num_objects * 4)

        # Attribute head
        self.cls_embed = nn.Embedding(num_objects, embed_dim)
        self.fc_attr   = nn.Linear(2048 + embed_dim, 512)
        self.relu_attr = nn.ReLU(inplace=True)
        self.attr_score= nn.Linear(512, num_attributes)

        for m in (self.cls_score, self.bbox_pred,
                  self.fc_attr, self.attr_score):
            m.apply(_init_weights)

        # BoxCoder for targets (match Caffe defaults if you like)
        self.box_coder = BoxCoder(weights=(10.,10.,5.,5.))
        # Inference filters
        self.score_thresh = 0.2
        self.max_regions  = 36
        self.roi_heads = RoIHeads(
            box_roi_pool       = self.roi_pool,
            box_head           = None,   # unused by the sampling functions
            box_predictor      = None,   # unused by the sampling functions
            fg_iou_thresh      = 0.5,    # Faster‑RCNN defaults
            bg_iou_thresh      = 0.5,
            batch_size_per_image = 128,
            positive_fraction  = 0.25,
            bbox_reg_weights   = (10.,10.,5.,5.),
            score_thresh       = self.score_thresh,
            nms_thresh         = 0.7,
            detections_per_img = self.max_regions
        )

    def forward(self, images, gt_targets=None):
        # 1) Backbone
        feats = self.backbone(images.tensors)
        conv4, conv5 = feats['0'], feats['1']

        # 2) RPN
        proposals, rpn_losses = self.rpn(
            images, {'0': conv4},
            gt_targets if self.training else None
        )

        # 3) Sample and label our proposals exactly like the original Faster RCNN:
        #    - append GTs, match IoU, subsample 25% pos / 75% neg
        # this will sample batch_size_per_image = 128 proposals
        if self.training:
            proposals, matched_idxs_list, labels_list, regression_targets_list = \
                self.roi_heads.select_training_samples(proposals, gt_targets)
        else:
            # at test time we use all RPN proposals, and no labels/reg targets
            matched_idxs_list = labels = regression_targets = None
        # 4) RoIAlign + global pool
        roi_feats = self.roi_pool({'1': conv5}, proposals, images.image_sizes)
        x = self.global_pool(roi_feats).flatten(1) # [M, 2048]
        # 5) Detection head predictions
        cls_logits = self.cls_score(x)      # [M, 1601]
        box_regs   = self.bbox_pred(x)      # [M, 1601*4]

        losses = {}
        if self.training:
            # 6a) RPN losses
            losses['rpn_obj'] = rpn_losses['loss_objectness']
            losses['rpn_box'] = rpn_losses['loss_rpn_box_reg']

            # 6b) Faster‐RCNN classification & box‐regression on sampled ROIs
            loss_cls, loss_box = fastrcnn_loss(
                cls_logits, box_regs,
                labels_list, regression_targets_list
            )
            losses['cls'] = loss_cls
            losses['box'] = loss_box
            # 7) Attribute loss on the *positive* ROIs only (labels > 0)
            labels = torch.cat(labels_list, dim=0).to(x.device)
            pos_inds = (labels > 0).nonzero(as_tuple=False).squeeze(1)
            if pos_inds.numel() > 0:
                # gather each image's attrs by its matched idxs, then flatten
                #`matched_idxs_list` is a list per image
                attr_targets = []
                for tgt, midx in zip(gt_targets, matched_idxs_list):
                    attr_targets.append(
                        tgt['attributes'][midx.to(tgt['attributes'].device)]
                    )
                attr_targets = torch.cat(attr_targets, dim=0).to(x.device)
                pos_attrs = attr_targets[pos_inds]
                # Now run the attribute head on the *positive* subset
                emb         = self.cls_embed(labels[pos_inds])           # [#pos, embed_dim]
                feat_attr   = torch.cat([x[pos_inds], emb], dim=1)       # [#pos, 2048+embed_dim]
                h           = self.relu_attr(self.fc_attr(feat_attr))   # [#pos,512]
                attr_logits = self.attr_score(h)                         # [#pos,401]
                losses['attr'] = F.cross_entropy(attr_logits, pos_attrs)
            else:
                losses['attr'] = torch.tensor(0., device=x.device)

            return losses

        # Inference
        probs = F.softmax(cls_logits, dim=1)[:,1:]
        max_s, lbls = probs.max(dim=1)
        keep = max_s > self.score_thresh
        keep_idxs = torch.nonzero(keep).squeeze(1)
        topk = keep_idxs[torch.argsort(max_s[keep], descending=True)][:self.max_regions]

        # 1) shape everything back into [N, C, 4]
        C      = cls_logits.size(1)
        deltas = box_regs.view(-1, C, 4)                   # [N, C, 4]

        # 2) pick the delta for the single most‐likely class for each kept ROI
        picked_labels = lbls[topk]                         # [K]
        picked_deltas = deltas[topk, picked_labels]        # [K,4]

        # 3) the “reference” RPN boxes for *this* image are proposals[0]
        ref_boxes = proposals[0][topk]                     # [K,4]

        # 4) decode: BoxCoder wants rel_codes:T[K,4], boxes:List[Tensor[K,4]]
        decoded = self.box_coder.decode(
            picked_deltas,    # Tensor[K,4]
            [ref_boxes]       # note the list!
        )                                                 # returns Tensor[K,1,4]
        decoded = decoded.squeeze(1)                     # → [K,4]

        out = {
            'cls_score': cls_logits[topk],  # [K, C]
            'boxes':     decoded            # [K, 4]
        }
        pred_cls = out['cls_score'].argmax(dim=1)
        emb_inf  = self.cls_embed(pred_cls)
        feat_inf = torch.cat([x[topk], emb_inf], dim=1)
        a_inf    = self.relu_attr(self.fc_attr(feat_inf))
        out['attr_score'] = self.attr_score(a_inf)
        out['pool5_flat'] = x[topk]
        return out