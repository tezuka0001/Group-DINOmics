import torch
from torch.nn import Transformer
import torch.nn as nn
import torch.nn.functional as F
from util.utils import *
from models.backbone import *

class Ball_detect_model(nn.Module):
    def __init__(self, args):
        super(Ball_detect_model, self).__init__()
        self.ball_pred = args.ball_pred
        self.flow_pred = args.flow_pred
        self.backbone = args.backbone
        
        self.temp_mask = args.temporal_mask
        self.future_mask = args.future_mask
        self.test_time_mask = args.test_time_mask
        self.mlp_comp = args.mlp_comp
        self.trans_comp = args.trans_comp
        
        self.dataset = args.dataset
        self.num_class = args.num_activities
        self.head_list = args.head_list
        self.num_frame = args.num_frame
        if self.dataset == 'volleyball':
            self.num_boxes = 12
        elif self.dataset == 'nba':
            self.num_boxes = 10

        H, W = args.image_height, args.image_width
        
        self.loc_guide = args.loc_guide
        self.comp_dim = args.comp_dim
        
        self.supervised = args.supervised
        self.fix_model = args.fix_model
        
        self.input_dim = args.hidden_size
        self.dino_head = len(self.head_list)
        
        if self.backbone == 'dinov2':
            self.image_encoder = dinov2_learnable_last_layer(args)
        elif self.backbone == 'dinov3':
            self.image_encoder = dinov3_learnable_last_layer(args)
        elif self.backbone == 'resnet50':
            self.image_encoder = ResNet50(args)
        elif self.backbone == 'vgg16':
            self.image_encoder = VGG16(args)
        elif self.backbone == 'vgg19':
            self.image_encoder = VGG19(args)
        elif self.backbone == 'clip':
            self.image_encoder = clip_vitl14_learnable_last_layer(args)
        elif self.backbone == 'ViT':
            self.image_encoder = vitl16_learnable_last_layer(args)
        elif self.backbone == 'MAE':
            self.image_encoder = mae_vitl_learnable_last_layer(args)
        elif self.backbone == 'dino':
            self.image_encoder = dino_vitb_learnable_last_layer(args)
        elif self.backbone == "siglip":
            self.image_encoder = siglip_vitl16_learnable_last_layer(args)
        elif self.backbone == 'siglip2':
            self.image_encoder = siglip2_vitl16_learnable_last_layer(args)
        elif self.backbone == 'franca':
            self.image_encoder = franca_learnable_last_layer(args)
        
        if self.ball_pred:
            self.bbox_head = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, 2),
            )
            self.cond_bbox_head = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, 2),
            )
        
        if self.flow_pred:
            self.spatial_mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
            )
            self.flow_head = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, 2),
            )
            self.cond_flow_head = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, 2),
            )
        
        if args.dataset == 'volleyball':
            self.temporal_transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=4, batch_first=True, dropout=0.0),
                num_layers=1,
            )
        elif args.dataset == 'nba':
            self.temporal_transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=4, batch_first=True, dropout=0.0),
                num_layers=1,
            )
        
        
        self.mask_prob = getattr(args, 'mask_prob', 0.2)
        
        self.tem_enc = positionalencoding1d(self.input_dim, 100).cuda()
        self.pos_enc_ind = positionalencoding2d(self.input_dim, 720, 1280).cuda()

        if self.dataset == 'volleyball':
            self.temporal_transformer_mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
            )
        elif self.dataset == 'nba':
            self.temporal_transformer_mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
            )
        
        if self.future_mask:
            self.future_bbox = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, 2*args.num_frame),
            )
        
        if self.mlp_comp:
            tem_enc_comp = positionalencoding1d(self.comp_dim*self.num_frame, 100)
            self.register_buffer('tem_enc_comp', tem_enc_comp)
            self.frame_comp = nn.Sequential(
                nn.Linear(self.input_dim, self.comp_dim),
                nn.ReLU(),
                nn.Linear(self.comp_dim, self.comp_dim),
            )
            self.cond_bbox_mlp = nn.Sequential(
                nn.Linear(self.comp_dim*self.num_frame, self.comp_dim*self.num_frame),
                nn.ReLU(),
                nn.Linear(self.comp_dim*self.num_frame, 2),
            )
        elif self.trans_comp:
            self.cls_temp_token = nn.Parameter(torch.zeros(1, 1, self.input_dim))
            nn.init.trunc_normal_(self.cls_temp_token, std=0.02)
            tem_enc_trans_comp = positionalencoding1d(self.input_dim, 100)
            self.register_buffer('tem_enc_trans_comp', tem_enc_trans_comp)
            self.trans_frame_comp = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=4, batch_first=True, dropout=0.0),num_layers=1)
            
        if self.supervised:
            self.classifier = nn.Linear(self.input_dim, self.num_class)
            if self.fix_model:
                for param in self.parameters():
                    param.requires_grad = False
                for p in self.classifier.parameters():
                    p.requires_grad = True
        
    def forward(self, input_data):
        ret_dic = {}
        pred_bbox_future = None
        x = input_data['images']
        boxes_in = input_data['bboxes']
        
        B, T, C, H, W = x.shape
        N = self.num_boxes
        images = x.reshape(B * T, C, H, W)
        cls_token = self.image_encoder(images)
        
        frame_feature = cls_token.view(B, T, self.input_dim)
        video_feature, _ = frame_feature.max(dim=1)
                
        ret_dic['video_features'] = video_feature
                    
        return ret_dic