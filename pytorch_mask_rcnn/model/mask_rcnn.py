from collections import OrderedDict
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc

#import sys 
#sys.path.append("/home/glados/unix-Documents/AstroSignals/mightee_dino")
#print(sys.path)
#from mightee_dino import utils as dino_utils

from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer


class MaskRCNN(nn.Module):
    """
    Implements Mask R-CNN.

    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor], containing the classification and regression losses 
    for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format, 
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)
        
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        num_classes (int): number of output classes of the model (including the background).
        
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals during training of the 
            classification head
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_num_detections (int): maximum number of detections, for all classes.
        
    """
    
    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.1, box_nms_thresh=0.6, box_num_detections=100):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        
        #------------- RPN --------------------------
        anchor_sizes = (128, 256, 512)
        anchor_ratios = (0.5, 1, 2)
        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        #------------ RoIHeads --------------------------
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)
        
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)
        
        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)
        
        self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)
        
        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)
        
        #------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=800, max_size=1333, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225])
        
    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]
        
        image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]
        feature = self.backbone(image)
        print(f"feature shape: {feature.shape}")
        
        proposal, rpn_losses = self.rpn(feature, image_shape, target)
        result, roi_losses = self.head(feature, proposal, image_shape, target)
        
        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result
        
        
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta        
    
    
class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """
        
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features
        
        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                
    
class ResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        
        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
                
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256
        
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        for module in self.body.values():
            x = module(x)
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x

    
def maskrcnn_resnet50(pretrained, num_classes, pretrained_backbone=True):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.
    
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """
    
    if pretrained:
        backbone_pretrained = False
        
    backbone = ResBackbone('resnet50', pretrained_backbone)
    model = MaskRCNN(backbone, num_classes)
    
    if pretrained:
        model_urls = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])
        
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        for i, name in enumerate(msd):
            if i in skip_list:
                continue
            msd[name].copy_(pretrained_msd[i])
            
        model.load_state_dict(msd)
    
    return model

# def get_DINO_teacher(args):
#     # ============ building network ... ============
#     if "vit" in args.arch:
#         model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
#         print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
#     elif "xcit" in args.arch:
#         model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)

#     elif args.arch in torchvision_models.__dict__.keys():
#             model = torchvision_models.__dict__[args.arch](num_classes=0)
#             model.fc = nn.Identity()
#         else:
#             print(f"Architecture {args.arch} non supported")
#             sys.exit(1)
#     model.cuda()
#     utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
#     return model

# def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
#     if os.path.isfile(pretrained_weights):
#         state_dict = torch.load(pretrained_weights, map_location="cpu")
#         if checkpoint_key is not None and checkpoint_key in state_dict:
#             print(f"Take key {checkpoint_key} in provided checkpoint dict")
#             state_dict = state_dict[checkpoint_key]
#         # remove `module.` prefix
#         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#         # remove `backbone.` prefix induced by multicrop wrapper
#         state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
#         msg = model.load_state_dict(state_dict, strict=False)
#         print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
#     else:
#         print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
#         url = None
#         if model_name == "vit_small" and patch_size == 16:
#             url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
#         elif model_name == "vit_small" and patch_size == 8:
#             url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
#         elif model_name == "vit_base" and patch_size == 16:
#             url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
#         elif model_name == "vit_base" and patch_size == 8:
#             url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
#         elif model_name == "xcit_small_12_p16":
#             url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
#         elif model_name == "xcit_small_12_p8":
#             url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
#         elif model_name == "xcit_medium_24_p16":
#             url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
#         elif model_name == "xcit_medium_24_p8":
#             url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
#         elif model_name == "resnet50":
#             url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
#         if url is not None:
#             print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
#             state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
#             model.load_state_dict(state_dict, strict=True)
#         else:
#             print("There is no reference weights available for this model => We use random weights.")



# def maskrcnn_DINO(dino_args, pretrained, num_classes, pretrained_backbone=True):
#     """
#     Constructs a Mask R-CNN model with a DINO teacher(ResNet-50) backbone.
    
#     Arguments:
#         pretrained (bool): If True, returns a model pre-trained on COCO train2017.
#         num_classes (int): number of classes (including the background).
#     """
    
#     if pretrained:
#         backbone_pretrained = False
        
#     ## load DINO teacher
#     if "vit" in dino_args.arch:
#         backbone = vits.__dict__[dino_args.arch](patch_size=dino_args.patch_size, num_classes=0)
#         print(f"Model {dino_args.arch} {dino_args.patch_size}x{dino_args.patch_size} built.")
#     elif "xcit" in dino_args.arch:
#         backbone = torch.hub.load('facebookresearch/xcit:main', dino_args.arch, num_classes=0)
#     elif dino_args.arch in models.__dict__.keys():
#             backbone = models.__dict__[dino_args.arch](num_classes=0)
#             backbone.fc = nn.Identity()
#     else:
#         print(f"Architecture {dino_args.arch} non supported")
#         sys.exit(1)
#     backbone.cuda()
#     load_pretrained_weights(backbone, dino_args.pretrained_weights, dino_args.checkpoint_key, dino_args.arch, dino_args.patch_size)

#     model = MaskRCNN(backbone, num_classes)
    
#     if pretrained:
#         model_urls = {
#             'maskrcnn_resnet50_fpn_coco':
#                 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
#         }
#         model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])
        
#         pretrained_msd = list(model_state_dict.values())
#         del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
#         for i, del_idx in enumerate(del_list):
#             pretrained_msd.pop(del_idx - i)

#         msd = model.state_dict()
#         skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
#         if num_classes == 91:
#             skip_list = [271, 272, 273, 274]
#         for i, name in enumerate(msd):
#             if i in skip_list:
#                 continue
#             msd[name].copy_(pretrained_msd[i])
            
#         model.load_state_dict(msd)
    
#     return model