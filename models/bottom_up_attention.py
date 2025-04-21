import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import ResNet101_Weights
from torchvision.models.detection._utils import BoxCoder
from fast_rcnn.config import cfg  # your config

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

    def forward(self, images, im_info, gt_targets=None):
        # Backbone
        feats = self.backbone(images.tensors)
        conv4, conv5 = feats['0'], feats['1']
        print("-Debug- conv4:", conv4.shape)
        print("-Debug- conv5:", conv5.shape)

        # RPN
        proposals, rpn_losses = self.rpn(
            images, {'0': conv4},
            gt_targets if self.training else None
        )
        # Debug prints for proposals
        for i in range(len(proposals)):
            print("-Debug- proposals {i} shape:", proposals[i].shape)

        # --- PREPEND GT BOXES to each proposal list! ---
        if self.training:
            print("-Debug- gt_targets:")
            for target in gt_targets:
                print("\t-Debug- target['boxes'] shape:", target['boxes'].shape)
                print("\t-Debug- target['labels'] shape:", target['labels'].shape)
            proposals_with_gt = []
            for props, tgt in zip(proposals, gt_targets):
                proposals_with_gt.append(
                    torch.cat([tgt['boxes'], props], dim=0)
                )
            proposals = proposals_with_gt

        # Debug prints for proposals after GT
        for proposal in proposals:
            print("-Debug- proposals after injecting gt boxes:", proposal.shape)

        # RoIAlign + pool → [sum(N_i),2048]
        roi_feats = self.roi_pool({'1': conv5}, proposals, images.image_sizes)
        print("-Debug- roi_feats shape:", roi_feats.shape)
        x = self.global_pool(roi_feats).flatten(1)
        print("-Debug- x shape:", x.shape)

        # Heads
        cls_logits = self.cls_score(x)          # [sum(N_i), num_obj]
        print("-Debug- cls_logits shape:", cls_logits.shape)
        box_regs   = self.bbox_pred(x)          # [sum(N_i), num_obj*4]
        print("-Debug- box_regs shape:", box_regs.shape)
        if self.training:
            # 1) RPN losses
            losses = {
                'rpn_obj': rpn_losses['loss_objectness'],
                'rpn_box': rpn_losses['loss_rpn_box_reg']
            }

            # 2) Compute flatten‐offsets for each image
            lengths = [p.shape[0] for p in proposals]
            offsets = [0] + torch.cumsum(
                torch.tensor(lengths[:-1], device=x.device), dim=0
            ).tolist()

            # 3) Gather all GT labels & proposal‐indices
            pos_idxs = []
            cls_lbls = []
            for i, tgt in enumerate(gt_targets):
                G = tgt['labels'].size(0)
                start = offsets[i]
                idxs = torch.arange(start, start + G, device=x.device)
                pos_idxs.append(idxs)
                cls_lbls.append(tgt['labels'])

            pos_idxs = torch.cat(pos_idxs, dim=0)   # [ΣGᵢ]
            cls_lbls = torch.cat(cls_lbls, dim=0)   # [ΣGᵢ]

            # 4) Classification loss on GT‐matched indices
            cls_pos = cls_logits[pos_idxs]          # [ΣGᵢ, num_obj]
            losses['cls'] = F.cross_entropy(cls_pos, cls_lbls)

            # 5) Box regression targets via BoxCoder
            #    prepare lists of length B
            ref_boxes     = [t['boxes']               for t in gt_targets]
            matched_props = [proposals[i][:len(t['boxes'])] 
                             for i, t in enumerate(gt_targets)]
            # returns Tensor [ΣGᵢ, 4]
            deltas = self.box_coder.encode(ref_boxes, matched_props)

            # 6) Gather predicted deltas for the GT classes
            C = cls_logits.size(1)
            box_pred = box_regs[pos_idxs]            # [ΣGᵢ, C*4]
            box_pred = box_pred.view(-1, C, 4)      # [ΣGᵢ, C, 4]
            box_pred = box_pred[
                torch.arange(cls_lbls.size(0), device=x.device), 
                cls_lbls
            ]                                       # [ΣGᵢ, 4]
            losses['box'] = F.smooth_l1_loss(box_pred, deltas)

            # 7) Attribute loss (optional)
            if 'attributes' in gt_targets[0]:
                attr_logits, attr_gts = [], []
                for i, tgt in enumerate(gt_targets):
                    G = tgt['attributes'].size(0)
                    start = offsets[i]
                    idxs  = torch.arange(start, start + G, device=x.device)
                    emb   = self.cls_embed(tgt['labels'])
                    feat  = torch.cat([x[idxs], emb], dim=1)
                    attr_logits.append(self.attr_score(self.relu_attr(self.fc_attr(feat))))
                    attr_gts.append(tgt['attributes'])
                attr_logits = torch.cat(attr_logits, dim=0)
                attr_gts     = torch.cat(attr_gts,     dim=0)
                losses['attr'] = F.cross_entropy(attr_logits, attr_gts, ignore_index=-1)

            return losses

        # Inference (unchanged) …
        probs = F.softmax(cls_logits, dim=1)[:,1:]
        max_s, _ = probs.max(dim=1)
        keep = max_s > self.score_thresh
        keep_idxs = torch.nonzero(keep).squeeze(1)
        topk = keep_idxs[torch.argsort(max_s[keep], descending=True)][:self.max_regions]

        out = {
            'cls_score': cls_logits[topk],
            'bbox_pred': box_regs[topk]
        }
        pred_cls = out['cls_score'].argmax(dim=1)
        emb_inf  = self.cls_embed(pred_cls)
        feat_inf = torch.cat([x[topk], emb_inf], dim=1)
        a_inf    = self.relu_attr(self.fc_attr(feat_inf))
        out['attr_score'] = self.attr_score(a_inf)
        out['pool5_flat'] = x[topk]
        return out