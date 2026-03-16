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
        self.person_recon = args.person_recon
        self.patch_maxpool = args.patch_maxpool
        self.patch_avgpool = args.patch_avgpool
        self.patch_cnn = args.patch_cnn
        self.patch_concat = args.patch_concat
        self.pooling_method = args.pooling_method
        self.cls_path = args.cls_path
        self.patch_path = args.patch_path
        self.backbone = args.backbone
        self.ViT_Blocks = args.ViT_Blocks
        self.linear_probing = args.linear_probing
        self.spatial_mlp_flow = args.spatial_mlp_flow
        
        self.flow_patch = args.flow_patch
        
        self.temp_mask = args.temporal_mask
        self.future_mask = args.future_mask
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
        elif self.dataset == 'jrdb':
            self.num_boxes = 60

        H, W = args.image_height, args.image_width
        
        self.comp_dim = args.comp_dim
        
        self.input_dim = args.hidden_size
        self.dino_head = len(self.head_list)
        
        self.supervised = args.supervised
        self.fix_model = args.fix_model
        
        if self.backbone == 'dinov2':
            if self.patch_path or self.patch_concat or self.ViT_Blocks > 0:
                self.image_encoder = dinov2_learnable_last_layer_patch(args)
            else:
                self.image_encoder = dinov2_learnable_last_layer(args)
        elif self.backbone == 'dinov3':
            if self.patch_path or self.patch_concat or self.ViT_Blocks > 0 or self.person_recon or self.flow_patch:
                self.image_encoder = dinov3_learnable_last_layer_patch(args)
            else:
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
            
        if self.ViT_Blocks > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=1024,
                nhead=8,
                dim_feedforward=int(self.input_dim * 4),
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.ViT_Block = nn.TransformerEncoder(encoder_layer, num_layers=self.ViT_Blocks)
            if (H, W) == (252, 448):
                self.featmap_h, self.featmap_w = 18, 32
            elif (H, W) == (288, 512) and self.backbone == 'dinov3':
                self.featmap_h, self.featmap_w = 18, 32
            elif (H, W) == (224, 224):
                if self.backbone == 'dinov2' or self.backbone == 'clip':
                    self.featmap_h, self.featmap_w = 16, 16
                elif self.backbone == 'dinov3' or self.backbone == 'ViT' or self.backbone == 'MAE':
                    self.featmap_h, self.featmap_w = 14, 14
            self.pos_enc_patch = positionalencoding2d(1024, self.featmap_h, self.featmap_w).cuda()
            
        if self.linear_probing:
            self.linear_probing_head = nn.Linear(self.input_dim, self.input_dim)
        
        if self.ball_pred:
            self.bbox_head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, 2),
                ),
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, 2),
                )
            ])
            self.cond_bbox_head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, 2),
                ),
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, 2),
                )
            ])
            
        if self.flow_pred:
            self.spatial_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, self.input_dim),
                ),
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, self.input_dim),
                )
            ])
            self.flow_head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, 2),
                ),
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, 2),
                )
            ])
            self.cond_flow_head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, 2),
                ),
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, 2),
                )
            ])
            
        if self.flow_patch:
            self.flow_patch_conv = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(512, 2, kernel_size=1),
            )
                
            
        if self.person_recon:
            self.cond_recon_head = nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, self.input_dim),
                )
            
        self.cls_proj = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_dim),
        )
        
        self.temporal_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=4, batch_first=True, dropout=0.0), num_layers=1,),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=4, batch_first=True, dropout=0.0), num_layers=1),
        ])
        
        self.mask_prob = getattr(args, 'mask_prob', 0.2)
        
        self.tem_enc = positionalencoding1d(self.input_dim, 100).cuda()
        if self.dataset == 'volleyball' or self.dataset == 'nba':
            self.pos_enc_ind = positionalencoding2d(self.input_dim, 720, 1280).cuda()
        else :
            self.pos_enc_ind = positionalencoding2d(self.input_dim, 480, 3760).cuda()

        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
            ),
            nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim),
            )
        ])
        
        if self.future_mask:
            self.future_bbox = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, 2*args.num_frame),
                ),
                nn.Sequential(
                    nn.Linear(self.input_dim, self.input_dim),
                    nn.ReLU(),
                    nn.Linear(self.input_dim, 2*args.num_frame),
                )
            ])
        
        if self.mlp_comp:
            tem_enc_comp = positionalencoding1d(self.comp_dim*self.num_frame, 100)
            self.register_buffer('tem_enc_comp', tem_enc_comp)
            self.frame_comp = nn.Modluelist([
                nn.Sequential(
                    nn.Linear(self.input_dim, self.comp_dim),
                    nn.ReLU(),
                    nn.Linear(self.comp_dim, self.comp_dim),
                ),
                nn.Sequential(
                    nn.Linear(self.input_dim, self.comp_dim),
                    nn.ReLU(),
                    nn.Linear(self.comp_dim, self.comp_dim),
                )
            ])
            self.cond_bbox_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.comp_dim*self.num_frame, self.comp_dim*self.num_frame),
                    nn.ReLU(),
                    nn.Linear(self.comp_dim*self.num_frame, 2),
                ),
                nn.Sequential(
                    nn.Linear(self.comp_dim*self.num_frame, self.comp_dim*self.num_frame),
                    nn.ReLU(),
                    nn.Linear(self.comp_dim*self.num_frame, 2),
                )
            ])
        elif self.trans_comp:
            self.cls_temp_token = nn.Parameter(torch.zeros(1, 1, self.input_dim))
            nn.init.trunc_normal_(self.cls_temp_token, std=0.02)
            self.cls_temp_token_patch = nn.Parameter(torch.zeros(1, 1, self.input_dim))
            nn.init.trunc_normal_(self.cls_temp_token_patch, std=0.02)
            tem_enc_trans_comp = positionalencoding1d(self.input_dim, 100)
            self.register_buffer('tem_enc_trans_comp', tem_enc_trans_comp)
            self.trans_frame_comp = nn.Modluelist([
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=4, batch_first=True, dropout=0.0),num_layers=1),
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=4, batch_first=True, dropout=0.0),num_layers=1),
            ])
            
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
        if self.patch_path or self.patch_concat or self.ViT_Blocks > 0 or self.person_recon or self.flow_patch:
            cls_token, patch_tokens = self.image_encoder(images)
            if self.ViT_Blocks > 0:
                patch_tokens = patch_tokens + self.pos_enc_patch
                patch_tokens = patch_tokens.reshape(B*T, -1, 1024)
                cls_token = cls_token.unsqueeze(1)
                feat_tokens = torch.cat([cls_token, patch_tokens], dim=1)
                feat_tokens = self.ViT_Block(feat_tokens)
                cls_token = feat_tokens[:, 0, :]
                patch_tokens = feat_tokens[:, 1:, :]
                patch_tokens = patch_tokens.reshape(B*T, -1, self.featmap_h, self.featmap_w)
            if self.flow_patch:
                esti_flow_patch = self.flow_patch_conv(patch_tokens)
                ret_dic['esti_flow_patch'] = esti_flow_patch
            elif self.patch_path or self.patch_concat:
                if self.patch_cnn:
                    patch_features = self.patch_cnn_net(patch_tokens)
                elif self.pooling_method == 'max':
                    patch_tokens = patch_tokens.reshape(B*T, -1, 1024)
                    patch_features = torch.max(patch_tokens, dim=1)[0]
                elif self.pooling_method == 'avg':
                    patch_tokens = patch_tokens.reshape(B*T, -1, 1024)
                    patch_features = torch.mean(patch_tokens, dim=1)
                cls_token = torch.cat([cls_token, patch_features], dim=1)    
        else:
            cls_token = self.image_encoder(images)
            if self.input_dim != 1024:
                cls_token = self.cls_proj(cls_token)
                
        if self.linear_probing:
            cls_token = self.linear_probing_head(cls_token)
                
        
        frame_feature = cls_token.view(B, T, self.input_dim)
        
        if self.cls_path:
            if self.ball_pred:
                pred_bbox_spatial = self.bbox_head[0](cls_token)
                ret_dic['pred_bbox_spatial'] = pred_bbox_spatial
            
            if self.flow_pred:
                if self.spatial_mlp_flow:
                    spatial_cls = self.spatial_mlp[0](cls_token)
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
                
                ind_feat = spatial_cls_expand + ind_loc_feat
                pred_flow_spatial = self.flow_head[0](ind_feat)
                ret_dic['pred_flow_spatial'] = pred_flow_spatial
            
            tem_enc_select = self.tem_enc[:T, :].expand(B, T, -1).cuda()
            frame_feature = frame_feature + tem_enc_select
            
            if self.temp_mask:
                if self.training:
                    random_mask = (torch.rand(B, T) < self.mask_prob).bool().cuda()
                else:
                    random_mask = None
                masked_frame_feature = self.temporal_transformer_encoder[0](frame_feature, src_key_padding_mask=random_mask)
            elif self.future_mask:
                attn_mask = Transformer.generate_square_subsequent_mask(T).cuda()
                masked_frame_feature = self.temporal_transformer_encoder[0](frame_feature, mask=attn_mask, is_causal=True)
                pred_bbox_future = self.future_bbox[0](masked_frame_feature)
                ret_dic['pred_bbox_future'] = pred_bbox_future
            else:
                masked_frame_feature = self.temporal_transformer_encoder[0](frame_feature)
                
            masked_frame_feature = self.temporal_transformer_mlp[0](masked_frame_feature)
            
            if self.mlp_comp:
                masked_frame_feature_comp = self.frame_comp[0](masked_frame_feature)
                video_feature = masked_frame_feature_comp.view(B, -1)
                video_feature_expand = video_feature.unsqueeze(1).expand(-1, T, -1)
                tem_enc_comp_select = self.tem_enc_comp[:T, :].expand(B, T, -1).cuda()
                cond_video_feature = video_feature_expand + tem_enc_comp_select
                cond_video_feature = cond_video_feature.view(B*T, self.comp_dim*T)
                pred_bbox_temporal = self.cond_bbox_mlp[0](cond_video_feature)
            elif self.trans_comp:
                cls_temp_tokens = self.cls_temp_token.expand(B, -1, -1).cuda()
                frame_features = torch.cat((cls_temp_tokens, masked_frame_feature), dim=1)
                tem_enc_trans_comp_select = self.tem_enc_trans_comp[:T+1, :].expand(B, T+1, -1).cuda()
                frame_features = frame_features + tem_enc_trans_comp_select
                frame_features = self.trans_frame_comp[0](frame_features)
                video_feature = frame_features[:, 0, :]
            else:
                video_feature, _ = masked_frame_feature.max(dim=1)
                
            if self.ball_pred:
                video_feature_expand = video_feature.unsqueeze(1).expand(-1, T, -1)
                cond_video_feature = video_feature_expand + tem_enc_select
                cond_video_feature = cond_video_feature.view(B*T, self.input_dim)
                pred_bbox_temporal = self.cond_bbox_head[0](cond_video_feature)
                ret_dic['pred_bbox_temporal'] = pred_bbox_temporal
            if self.flow_pred:
                video_feature_expand = video_feature.unsqueeze(1).expand(-1, T*N, -1)
                video_feature_expand = video_feature_expand.view(B, T, N, self.input_dim)
                tem_enc_select_expand = tem_enc_select.unsqueeze(2).expand(-1, -1, N, -1)
                cond_video_feature = video_feature_expand + tem_enc_select_expand + ind_loc_feat
                pred_flow_temporal = self.cond_flow_head[0](cond_video_feature)
                ret_dic['pred_flow_temporal'] = pred_flow_temporal
            if self.person_recon:
                boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])//2
                boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])//2
                boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
                boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
                pos_enc_ind = self.pos_enc_ind
                ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
                ind_loc_feat = ind_loc_feat.view(B, T, N, self.input_dim)
                
                video_feature_expand = video_feature.unsqueeze(1).expand(-1, T*N, -1)
                video_feature_expand = video_feature_expand.view(B, T, N, self.input_dim)
                tem_enc_select_expand = tem_enc_select.unsqueeze(2).expand(-1, -1, N, -1)
                cond_video_feature = video_feature_expand + tem_enc_select_expand + ind_loc_feat
                pred_person_recon = self.cond_recon_head(cond_video_feature)
                ret_dic['pred_person_recon'] = pred_person_recon
                ret_dic['patch_tokens'] = patch_tokens
                
            ret_dic['video_features'] = video_feature
            
        if self.patch_path:
            if self.patch_cnn:
                frame_token = self.patch_cnn_net(patch_tokens)
            elif self.patch_maxpool:
                patch_tokens = patch_tokens.reshape(B*T, -1, 1024)
                frame_token = torch.max(patch_tokens, dim=1)[0]
            elif self.patch_avgpool:
                patch_tokens = patch_tokens.reshape(B*T, -1, 1024)
                frame_token = torch.mean(patch_tokens, dim=1)
                
            frame_feature_patch = frame_token.view(B, T, -1)
            
            if self.ball_pred:
                pred_bbox_spatial_patch = self.bbox_head[1](frame_token)
                ret_dic['pred_bbox_spatial'] = pred_bbox_spatial_patch
            
            if self.flow_pred:
                if self.spatial_mlp_flow:
                    spatial_cls_patch = self.spatial_mlp[1](frame_token)
                else:
                    spatial_cls_patch = frame_token
                spatial_cls_patch_expand = spatial_cls_patch.unsqueeze(1).expand(-1, N, -1)
                spatial_cls_patch_expand = spatial_cls_patch_expand.view(B, T, N, self.input_dim)
                
                boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])//2
                boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])//2
                boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
                boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
                pos_enc_ind = self.pos_enc_ind
                ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
                ind_loc_feat = ind_loc_feat.view(B, T, N, self.input_dim)
                
                ind_feat_patch = spatial_cls_patch_expand + ind_loc_feat
                pred_flow_spatial_patch = self.flow_head[1](ind_feat_patch)
                ret_dic['pred_flow_spatial'] = pred_flow_spatial_patch
            elif self.patch_en_decoder:
                ret_dic['flow_recon_spatial'] = flow_recon
            
            tem_enc_select = self.tem_enc[:T, :].expand(B, T, -1).cuda()
            frame_feature_patch = frame_feature_patch + tem_enc_select
            
            if self.temp_mask:
                if self.training:
                    random_mask = (torch.rand(B, T) < self.mask_prob).bool().cuda()
                else:
                    random_mask = None
                masked_frame_feature_patch = self.temporal_transformer_encoder[1](frame_feature_patch, src_key_padding_mask=random_mask)
            elif self.future_mask:
                attn_mask = Transformer.generate_square_subsequent_mask(T).cuda()
                masked_frame_feature_patch = self.temporal_transformer_encoder[1](frame_feature, mask=attn_mask, is_causal=True)
                pred_bbox_future_patch = self.future_bbox[1](masked_frame_feature_patch)
                ret_dic['pred_bbox_future'] = pred_bbox_future_patch
            else:
                masked_frame_feature_patch = self.temporal_transformer_encoder[1](frame_feature_patch)
                
            masked_frame_feature_patch = self.temporal_transformer_mlp[1](masked_frame_feature_patch)
            
            if self.mlp_comp:
                masked_frame_feature_comp_patch = self.frame_comp[1](masked_frame_feature_patch)
                video_feature_patch = masked_frame_feature_comp_patch.view(B, -1)
                video_feature_expand_patch = video_feature_patch.unsqueeze(1).expand(-1, T, -1)
                tem_enc_comp_select = self.tem_enc_comp[:T, :].expand(B, T, -1).cuda()
                cond_video_feature_patch = video_feature_expand_patch + tem_enc_comp_select
                cond_video_feature_patch = cond_video_feature_patch.view(B*T, self.comp_dim*T)
                pred_bbox_temporal_patch = self.cond_bbox_mlp[1](cond_video_feature_patch)
            elif self.trans_comp:
                cls_temp_tokens_patch = self.cls_temp_token_patch.expand(B, -1, -1).cuda()
                frame_features_patch = torch.cat((cls_temp_tokens_patch, masked_frame_feature_patch), dim=1)
                tem_enc_trans_comp_select = self.tem_enc_trans_comp[:T+1, :].expand(B, T+1, -1).cuda()
                frame_features_patch = frame_features_patch + tem_enc_trans_comp_select
                frame_features_patch = self.trans_frame_comp[1](frame_features_patch)
                video_feature_patch = frame_features_patch[:, 0, :]
            else:
                video_feature_patch, _ = masked_frame_feature_patch.max(dim=1)
                
            if self.ball_pred:
                video_feature_expand_patch = video_feature_patch.unsqueeze(1).expand(-1, T, -1)
                cond_video_feature_patch = video_feature_expand_patch + tem_enc_select
                cond_video_feature_patch = cond_video_feature_patch.view(B*T, self.input_dim)
                pred_bbox_temporal_patch = self.cond_bbox_head[1](cond_video_feature_patch)
                ret_dic['pred_bbox_temporal'] = pred_bbox_temporal_patch
            if self.flow_pred:
                video_feature_expand_patch = video_feature_patch.unsqueeze(1).expand(-1, T*N, -1)
                video_feature_expand_patch = video_feature_expand_patch.view(B, T, N, self.input_dim)
                tem_enc_select_expand = tem_enc_select.unsqueeze(2).expand(-1, -1, N, -1)
                cond_video_feature_patch = video_feature_expand_patch + tem_enc_select_expand + ind_loc_feat
                pred_flow_temporal_patch = self.cond_flow_head[1](cond_video_feature_patch)
                ret_dic['pred_flow_temporal'] = pred_flow_temporal_patch

            ret_dic['video_features'] = video_feature_patch
                    
        return ret_dic