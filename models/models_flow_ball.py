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
        self.linear_probing = args.linear_probing
        self.spatial_backbone_mlp = args.spatial_backbone_mlp
        self.spatial_mlp_ball = args.spatial_mlp_ball
        self.spatial_mlp_flow = args.spatial_mlp_flow
        self.temp_mlp_flow = args.temp_mlp_flow
        
        self.temp_mask = args.temporal_mask
        self.future_mask = args.future_mask
        self.test_time_mask = args.test_time_mask
        self.mlp_comp = args.mlp_comp
        self.trans_comp = args.trans_comp
        
        self.dataset = args.dataset
        self.num_class = args.num_activities
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
        
        if self.backbone == 'dinov3':
            if args.use_lora:
                self.image_encoder = dinov3_with_lora(args)
            else:
                self.image_encoder = dinov3_learnable_last_layer(args)
        elif self.backbone == 'dinov2':
            self.image_encoder = dinov2_learnable_last_layer(args)
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
        
        if self.linear_probing:
            self.linear_probing_head = nn.Linear(self.input_dim, self.input_dim)
        
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
        
        self.backbone_mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
            )
        
        self.spatial_mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
            )
        
        self.spatial_ball_mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
            )
        
        if self.flow_pred:
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
            self.temporal_mlp_flow = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
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
        elif args.dataset == 'jrdb':
            self.temporal_transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=4, batch_first=True, dropout=0.0),
                num_layers=1,
            )
        
        self.mask_prob = getattr(args, 'mask_prob', 0.2)
        
        self.tem_enc = positionalencoding1d(self.input_dim, 100).cuda()
        if self.dataset == 'volleyball' or self.dataset == 'nba':
            self.pos_enc_ind = positionalencoding2d(self.input_dim, 720, 1280).cuda()
        else :
            self.pos_enc_ind = positionalencoding2d(self.input_dim, 480, 3760).cuda()

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
        elif self.dataset == 'jrdb':
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
        
        if self.linear_probing:
            cls_token = self.linear_probing_head(cls_token)
        elif self.spatial_backbone_mlp:
            cls_token = self.backbone_mlp(cls_token)
        
        frame_feature = cls_token.view(B, T, self.input_dim)
            
        if self.ball_pred:
            if self.spatial_mlp_ball:
                cls_token_mlp = self.spatial_ball_mlp(cls_token)
                pred_bbox_spatial = self.bbox_head(cls_token_mlp)
            else:
                pred_bbox_spatial = self.bbox_head(cls_token)
            ret_dic['pred_bbox_spatial'] = pred_bbox_spatial
        
        if self.flow_pred:
            if self.spatial_mlp_flow:
                spatial_cls = self.spatial_mlp(cls_token)
            else:
                spatial_cls = cls_token
            spatial_cls_expand = spatial_cls.unsqueeze(1).expand(-1, N, -1)
            spatial_cls_expand = spatial_cls_expand.view(B, T, N, self.input_dim)
            
            boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])//2
            boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])//2
            boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
            boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
            pos_enc_ind = self.pos_enc_ind
            ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
            ind_loc_feat = ind_loc_feat.view(B, T, N, self.input_dim)
            
            if self.loc_guide == 'spatial_temporal_loc' or self.loc_guide == 'spatial_loc':
                ind_feat = spatial_cls_expand + ind_loc_feat
            elif self.loc_guide == 'temporal_loc' or self.loc_guide == 'none':
                ind_feat = spatial_cls_expand
            pred_flow_spatial = self.flow_head(ind_feat)
            ret_dic['pred_flow_spatial'] = pred_flow_spatial
        
        tem_enc_select = self.tem_enc[:T, :].expand(B, T, -1).cuda()
        frame_feature = frame_feature + tem_enc_select
        
        if self.temp_mask:
            if self.training:
                random_mask = (torch.rand(B, T) < self.mask_prob).bool().cuda()
            else:
                random_mask = None
            masked_frame_feature = self.temporal_transformer_encoder(frame_feature, src_key_padding_mask=random_mask)
        elif self.future_mask:
            if self.test_time_mask:
                attn_mask = Transformer.generate_square_subsequent_mask(T).cuda()
                masked_frame_feature = self.temporal_transformer_encoder(frame_feature, mask=attn_mask, is_causal=True)
                pred_bbox_future = self.future_bbox(masked_frame_feature)
                ret_dic['pred_bbox_future'] = pred_bbox_future
            else:
                if self.training:
                    attn_mask = Transformer.generate_square_subsequent_mask(T).cuda()
                    masked_frame_feature = self.temporal_transformer_encoder(frame_feature, mask=attn_mask, is_causal=True)
                else:
                    masked_frame_feature = self.temporal_transformer_encoder(frame_feature)
                pred_bbox_future = self.future_bbox(masked_frame_feature)
                ret_dic['pred_bbox_future'] = pred_bbox_future
        else:
            masked_frame_feature = self.temporal_transformer_encoder(frame_feature)
            
        if self.dataset == 'volleyball':
            masked_frame_feature = self.temporal_transformer_mlp(masked_frame_feature)
        elif self.dataset == 'nba':
            masked_frame_feature = self.temporal_transformer_mlp(masked_frame_feature)
        else:
            masked_frame_feature = self.temporal_transformer_mlp(masked_frame_feature)
        
        if self.mlp_comp:
            masked_frame_feature_comp = self.frame_comp(masked_frame_feature)
            video_feature = masked_frame_feature_comp.view(B, -1)
            video_feature_expand = video_feature.unsqueeze(1).expand(-1, T, -1)
            tem_enc_comp_select = self.tem_enc_comp[:T, :].expand(B, T, -1).cuda()
            cond_video_feature = video_feature_expand + tem_enc_comp_select
            cond_video_feature = cond_video_feature.view(B*T, self.comp_dim*T)
            pred_bbox_temporal = self.cond_bbox_mlp(cond_video_feature)
        elif self.trans_comp:
            cls_temp_tokens = self.cls_temp_token.expand(B, -1, -1).cuda()
            frame_features = torch.cat((cls_temp_tokens, masked_frame_feature), dim=1)
            tem_enc_trans_comp_select = self.tem_enc_trans_comp[:T+1, :].expand(B, T+1, -1).cuda()
            frame_features = frame_features + tem_enc_trans_comp_select
            frame_features = self.trans_frame_comp(frame_features)
            video_feature = frame_features[:, 0, :]
        else:
            video_feature, _ = masked_frame_feature.max(dim=1)
            
        if self.ball_pred:
            video_feature_expand = video_feature.unsqueeze(1).expand(-1, T, -1)
            if self.loc_guide == 'spatial_temporal_loc' or self.loc_guide == 'temporal_loc':
                cond_video_feature = video_feature_expand + tem_enc_select
            elif self.loc_guide == 'spatial_loc' or self.loc_guide == 'none':
                cond_video_feature = video_feature_expand
            cond_video_feature = cond_video_feature.contiguous().view(B*T, self.input_dim)
            pred_bbox_temporal = self.cond_bbox_head(cond_video_feature)
            ret_dic['pred_bbox_temporal'] = pred_bbox_temporal
        if self.flow_pred:
            if self.temp_mlp_flow:
                video_feature_mlp = self.temporal_mlp_flow(video_feature)
                video_feature_expand = video_feature_mlp.unsqueeze(1).expand(-1, T*N, -1)
            else:
                video_feature_expand = video_feature.unsqueeze(1).expand(-1, T*N, -1)
            video_feature_expand = video_feature_expand.view(B, T, N, self.input_dim)
            tem_enc_select_expand = tem_enc_select.unsqueeze(2).expand(-1, -1, N, -1)
            if self.loc_guide == 'spatial_temporal_loc':
                cond_video_feature = video_feature_expand + tem_enc_select_expand + ind_loc_feat
            elif self.loc_guide == 'temporal_loc':
                cond_video_feature = video_feature_expand + tem_enc_select_expand
            elif self.loc_guide == 'spatial_loc':
                cond_video_feature = video_feature_expand + ind_loc_feat
            elif self.loc_guide == 'none':
                cond_video_feature = video_feature_expand
            pred_flow_temporal = self.cond_flow_head(cond_video_feature)
            ret_dic['pred_flow_temporal'] = pred_flow_temporal
        
        if self.supervised:
            activites_score = self.classifier(video_feature)
            ret_dic['activities_score'] = activites_score
                
        ret_dic['video_features'] = video_feature
                    
        return ret_dic