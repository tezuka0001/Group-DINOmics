import torch
from torch.nn import Transformer
import torch.nn as nn
import torch.nn.functional as F
from util.utils import *
import torchvision.models as models
import loralib as lora
from transformers import AutoModel

# nn.Linear → lora.Linear
def _to_lora_linear(linear: nn.Linear, r=16, alpha=None, dropout=0.0) -> lora.Linear:
    alpha = r if alpha is None else alpha
    new = lora.Linear(
        linear.in_features,
        linear.out_features,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=(linear.bias is not None),
    )
    with torch.no_grad():
        new.weight.copy_(linear.weight)
        if linear.bias is not None:
            new.bias.copy_(linear.bias)
    return new

# inject LoRA
def inject_lora_into_dinov3(
    backbone: nn.Module,
    indices,
    *,
    r=16,
    alpha=None,
    dropout=0.0,
    attn_q=True, attn_k=True, attn_v=True, attn_o=False,
    mlp_up=False, mlp_down=False,
    unfix_last=False,
    learnable_layers=1,
    verbose=True,
):
    assert hasattr(backbone, "layer")
    layers = backbone.layer
    n = len(layers)

    for p in backbone.parameters():
        p.requires_grad = False

    normed = []
    for i in indices:
        j = i + n if i < 0 else i
        if 0 <= j < n:
            normed.append(j)
        elif verbose:
            print(f"[LoRA] skip invalid index {i} (norm={j})")
    target = sorted(set(normed))
    if verbose:
        print(f"[LoRA] target blocks: {target} / total={n}")

    for i in target:
        blk = layers[i]

        # Attention
        if hasattr(blk, "attention"):
            attn = blk.attention
            if attn_q and isinstance(getattr(attn, "q_proj", None), nn.Linear):
                attn.q_proj = _to_lora_linear(attn.q_proj, r, alpha, dropout)
                if verbose: print(f"[LoRA] layer.{i}.attention.q_proj")
            if attn_k and isinstance(getattr(attn, "k_proj", None), nn.Linear):
                attn.k_proj = _to_lora_linear(attn.k_proj, r, alpha, dropout)
                if verbose: print(f"[LoRA] layer.{i}.attention.k_proj")
            if attn_v and isinstance(getattr(attn, "v_proj", None), nn.Linear):
                attn.v_proj = _to_lora_linear(attn.v_proj, r, alpha, dropout)
                if verbose: print(f"[LoRA] layer.{i}.attention.v_proj")
            if attn_o and isinstance(getattr(attn, "o_proj", None), nn.Linear):
                attn.o_proj = _to_lora_linear(attn.o_proj, r, alpha, dropout)
                if verbose: print(f"[LoRA] layer.{i}.attention.o_proj")

        # MLP
        if hasattr(blk, "mlp"):
            mlp = blk.mlp
            if mlp_up and isinstance(getattr(mlp, "up_proj", None), nn.Linear):
                mlp.up_proj = _to_lora_linear(mlp.up_proj, r, alpha, dropout)
                if verbose: print(f"[LoRA] layer.{i}.mlp.up_proj")
            if mlp_down and isinstance(getattr(mlp, "down_proj", None), nn.Linear):
                mlp.down_proj = _to_lora_linear(mlp.down_proj, r, alpha, dropout)
                if verbose: print(f"[LoRA] layer.{i}.mlp.down_proj")

    # lora_A / lora_B
    for name, p in backbone.named_parameters():
        p.requires_grad = ("lora_A" in name) or ("lora_B" in name)

    if unfix_last:
        k = max(1, min(learnable_layers, n))
        targets = set(range(n - k, n))
        print(f"→ backbone_learnable=True : train last {k} block(s) → {sorted(targets)}")

        for name, param in backbone.named_parameters():
            in_last_blocks = any(name.startswith(f"layer.{i}") for i in targets)
            is_lora = ("lora_A" in name) or ("lora_B" in name)
            param.requires_grad = in_last_blocks or is_lora

        if verbose:
            print(f"[Unfix] layer.{sorted(targets)} are also trainable in addition to LoRA parameters")

class dinov3_with_lora(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")

        if getattr(args, "use_lora", True):
            inject_lora_into_dinov3(
                self.backbone,
                indices=getattr(args, "lora_blocks", list(range(0, 24))),
                r=getattr(args, "lora_rank", 16),
                alpha=getattr(args, "lora_alpha", None),      # None→alpha=r
                dropout=getattr(args, "lora_dropout", 0.0),
                attn_q=getattr(args, "lora_q", True),
                attn_k=getattr(args, "lora_k", False),
                attn_v=getattr(args, "lora_v", True),
                attn_o=getattr(args, "lora_o", False),
                mlp_up=getattr(args, "lora_mlp_up", False),
                mlp_down=getattr(args, "lora_mlp_down", False),
                unfix_last=getattr(args, "backbone_learnable", False),
                learnable_layers=getattr(args, "backbone_learnable_layers", 1),
                verbose=getattr(args, "verbose", True),
            )
        elif args.backbone_learnable:
            num_layers = len(self.backbone.layer)
            for name, p in self.backbone.named_parameters():
                p.requires_grad = name.startswith(f"layer.{num_layers-1}")

    def forward(self, x):
        out = self.backbone(x)
        return out.last_hidden_state[:, 0]  # CLS

class ResNet50(nn.Module):
    def __init__(self, args):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Identity()
        self.linear = nn.Linear(2048, args.hidden_size)
        
    def forward(self, image):
        features = self.resnet50(image)
        features = self.linear(features)
        return features

class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier = nn.Identity()
        self.linear = nn.Linear(25088, args.hidden_size)
        
    def forward(self, image):
        features = self.vgg16(image)
        features = self.linear(features)
        return features

class VGG19(nn.Module):
    def __init__(self, args):
        super(VGG19, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        self.vgg19.classifier = nn.Identity()
        self.linear = nn.Linear(25088, args.hidden_size)
        
    def forward(self, image):
        features = self.vgg19(image)
        features = self.linear(features)
        return features

class dinov3_learnable_last_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone_learnable       = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable  = getattr(args, "backbone_full_learnable", False)
        self.last_k_trainable     = getattr(args, "backbone_learnable_layers", 1)
        self.vit_arch = getattr(args, "ViT_arch", "vit-l")

        print("Loading DINOv3 backbone from Hugging Face")
        if self.vit_arch == "vit-l":
            self.dinov3_backbone = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
        elif self.vit_arch == "vit-b":
            self.dinov3_backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")

        encoder_layers = self.dinov3_backbone.layer
        num_layers = len(encoder_layers)

        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : ALL parameters trainable")
            for p in self.dinov3_backbone.parameters():
                p.requires_grad = True

        else:
            for p in self.dinov3_backbone.parameters():
                p.requires_grad = False

            if self.backbone_learnable:
                k = max(1, min(self.last_k_trainable, num_layers))
                targets = set(range(num_layers - k, num_layers))
                print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {sorted(targets)}")

                for name, param in self.dinov3_backbone.named_parameters():
                    train_this = any(name.startswith(f"layer.{i}") for i in targets)
                    param.requires_grad = train_this
            else:
                print("→ backbone_learnable=False : ALL parameters frozen")

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W)
        Returns:
            cls_token: (B, hidden_dim)
        """
        outputs = self.dinov3_backbone(image)
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token
    
class dinov3_learnable_last_layer_patch(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone_learnable       = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable  = getattr(args, "backbone_full_learnable", False)
        self.last_k_trainable     = getattr(args, "backbone_learnable_layers", 1)
        self.vit_arch = getattr(args, "ViT_arch", "vit-l")
        self.patch_height = args.image_height //16
        self.patch_width = args.image_width //16

        print("Loading DINOv3 backbone from Hugging Face")
        if self.vit_arch == "vit-l":
            self.dinov3_backbone = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
        elif self.vit_arch == "vit-b":
            self.dinov3_backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")

        encoder_layers = self.dinov3_backbone.layer
        num_layers = len(encoder_layers)

        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : ALL parameters trainable")
            for p in self.dinov3_backbone.parameters():
                p.requires_grad = True

        else:
            for p in self.dinov3_backbone.parameters():
                p.requires_grad = False

            if self.backbone_learnable:
                k = max(1, min(self.last_k_trainable, num_layers))
                targets = set(range(num_layers - k, num_layers))
                print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {sorted(targets)}")

                for name, param in self.dinov3_backbone.named_parameters():
                    train_this = any(name.startswith(f"layer.{i}") for i in targets)
                    param.requires_grad = train_this
            else:
                print("→ backbone_learnable=False : ALL parameters frozen")

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W)
        Returns:
            cls_token: (B, hidden_dim)
        """
        outputs = self.dinov3_backbone(image)
        cls_token = outputs.last_hidden_state[:, 0]
        patch_tokens = outputs.last_hidden_state[:, 5:]
        patch_tokens = patch_tokens.view(patch_tokens.size(0), -1, self.patch_height, self.patch_width)
        
        return cls_token, patch_tokens
        
class dinov2_learnable_last_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone_learnable       = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable  = getattr(args, "backbone_full_learnable", False)
        self.last_k_trainable     = getattr(args, "backbone_learnable_layers", 1)
        self.vit_arch = getattr(args, "ViT_arch", "vit-l")

        print("Loading DINOv2 backbone from torch.hub")
        if self.vit_arch == "vit-l":
            self.dinov2_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        elif self.vit_arch == "vit-b":
            self.dinov2_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        elif self.vit_arch == "vit-b-In21k":
            self.dinov2_backbone = torch.hub.load('valeoai/Franca', 'franca_vitb14', weights='DINOV2_IN21K', use_rasa_head=False)
        
        encoder_layers = self.dinov2_backbone.blocks
        num_layers = len(encoder_layers)

        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : ALL parameters trainable")
            for p in self.dinov2_backbone.parameters():
                p.requires_grad = True

        else:
            for p in self.dinov2_backbone.parameters():
                p.requires_grad = False

            if self.backbone_learnable:
                k = max(1, min(self.last_k_trainable, num_layers))
                targets = set(range(num_layers - k, num_layers))
                print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {sorted(targets)}")

                for name, param in self.dinov2_backbone.named_parameters():
                    train_this = any(name.startswith(f"blocks.{i}") for i in targets)
                    param.requires_grad = train_this
            else:
                print("→ backbone_learnable=False : ALL parameters frozen")

    def forward(self, image):
        features = self.dinov2_backbone.get_intermediate_layers(
            image,
            n=1,
            reshape=True,
            return_class_token=True
        )[0]
        cls_token    = features[1]
        return cls_token

class clip_vitl14_learnable_last_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone_learnable       = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable  = getattr(args, "backbone_full_learnable", False)
        self.last_k_trainable     = getattr(args, "backbone_learnable_layers", 1)
        self.vit_arch = getattr(args, "ViT_arch", "vit-l")
        
        from transformers import CLIPModel
        if self.vit_arch == "vit-l":
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        elif self.vit_arch == "vit-b":
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True)

        encoder_layers = self.clip_model.vision_model.encoder.layers
        num_layers = len(encoder_layers)

        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : ALL parameters trainable")
            for p in self.clip_model.vision_model.parameters():
                p.requires_grad = True

        else:
            for p in self.clip_model.vision_model.parameters():
                p.requires_grad = False

            if self.backbone_learnable:
                k = max(1, min(self.last_k_trainable, num_layers))
                targets = set(range(num_layers - k, num_layers))
                print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {sorted(targets)}")

                for name, param in self.clip_model.vision_model.named_parameters():
                    train_this = any(name.startswith(f"encoder.layers.{i}") for i in targets)
                    param.requires_grad = train_this
            else:
                print("→ backbone_learnable=False : ALL parameters frozen")

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W)
        Returns:
            cls_token: (B, hidden_dim)
        """
        outputs = self.clip_model.vision_model(pixel_values=image)
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token
    
class siglip2_vitl16_learnable_last_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone_learnable       = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable  = getattr(args, "backbone_full_learnable", False)
        self.last_k_trainable     = getattr(args, "backbone_learnable_layers", 1)
        self.vit_arch = getattr(args, "ViT_arch", "vit-l")
        
        from transformers import SiglipModel
        if self.vit_arch == "vit-l":
            self.siglip2_model = SiglipModel.from_pretrained("google/siglip2-large-patch16-256")
        elif self.vit_arch == "vit-b":
            self.siglip2_model = SiglipModel.from_pretrained("google/siglip2-base-patch16-256")

        encoder_layers = self.siglip2_model.vision_model.encoder.layers
        num_layers = len(encoder_layers)

        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : ALL parameters trainable")
            for p in self.siglip2_model.vision_model.parameters():
                p.requires_grad = True

        else:
            for p in self.siglip2_model.vision_model.parameters():
                p.requires_grad = False

            if self.backbone_learnable:
                k = max(1, min(self.last_k_trainable, num_layers))
                targets = set(range(num_layers - k, num_layers))
                print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {sorted(targets)}")

                for name, param in self.siglip2_model.vision_model.named_parameters():
                    train_this = any(name.startswith(f"encoder.layers.{i}") for i in targets)
                    param.requires_grad = train_this
            else:
                print("→ backbone_learnable=False : ALL parameters frozen")

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W)
        Returns:
            cls_token: (B, hidden_dim)
        """
        outputs = self.siglip2_model.vision_model(image)
        cls_token = outputs.pooler_output
        return cls_token
    
class siglip_vitl16_learnable_last_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone_learnable       = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable  = getattr(args, "backbone_full_learnable", False)
        self.last_k_trainable     = getattr(args, "backbone_learnable_layers", 1)
        self.vit_arch = getattr(args, "ViT_arch", "vit-l")
        
        from transformers import SiglipModel
        if self.vit_arch == "vit-l":
            self.siglip_model = SiglipModel.from_pretrained("google/siglip-large-patch16-256")
        elif self.vit_arch == "vit-b":
            self.siglip_model = SiglipModel.from_pretrained("google/siglip-base-patch16-256")

        encoder_layers = self.siglip_model.vision_model.encoder.layers
        num_layers = len(encoder_layers)

        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : ALL parameters trainable")
            for p in self.siglip_model.vision_model.parameters():
                p.requires_grad = True

        else:
            for p in self.siglip_model.vision_model.parameters():
                p.requires_grad = False

            if self.backbone_learnable:
                k = max(1, min(self.last_k_trainable, num_layers))
                targets = set(range(num_layers - k, num_layers))
                print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {sorted(targets)}")

                for name, param in self.siglip_model.vision_model.named_parameters():
                    train_this = any(name.startswith(f"encoder.layers.{i}") for i in targets)
                    param.requires_grad = train_this
            else:
                print("→ backbone_learnable=False : ALL parameters frozen")

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W)
        Returns:
            cls_token: (B, hidden_dim)
        """
        outputs = self.siglip_model.vision_model(image)
        cls_token = outputs.pooler_output
        return cls_token
    
class vitl16_learnable_last_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone_learnable       = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable  = getattr(args, "backbone_full_learnable", False)
        self.last_k_trainable     = getattr(args, "backbone_learnable_layers", 1)
        self.vit_arch = getattr(args, "ViT_arch", "vit-l")

        from transformers import ViTModel
        if self.vit_arch == "vit-l":
            self.vit_model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
        elif self.vit_arch == "vit-b":
            self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        encoder_layers = self.vit_model.encoder.layer
        num_layers = len(encoder_layers)

        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : ALL parameters trainable")
            for p in self.vit_model.encoder.parameters():
                p.requires_grad = True

        else:
            for p in self.vit_model.encoder.parameters():
                p.requires_grad = False

            if self.backbone_learnable:
                k = max(1, min(self.last_k_trainable, num_layers))
                targets = set(range(num_layers - k, num_layers))
                print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {sorted(targets)}")

                for name, param in self.vit_model.encoder.named_parameters():
                    train_this = any(name.startswith(f"layer.{i}") for i in targets)
                    param.requires_grad = train_this
            else:
                print("→ backbone_learnable=False : ALL parameters frozen")

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W)
        Returns:
            cls_token: (B, hidden_dim)
        """
        outputs = self.vit_model(image)
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token
    
class mae_vitl_learnable_last_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone_learnable       = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable  = getattr(args, "backbone_full_learnable", False)
        self.last_k_trainable     = getattr(args, "backbone_learnable_layers", 1)
        self.vit_arch = getattr(args, "ViT_arch", "vit-l")

        from transformers import ViTMAEModel
        if self.vit_arch == "vit-l":
            self.mae_model = ViTMAEModel.from_pretrained("facebook/vit-mae-large", use_safetensors=True)
        elif self.vit_arch == "vit-b":
            self.mae_model = ViTMAEModel.from_pretrained("facebook/vit-mae-base", use_safetensors=True)

        encoder_layers = self.mae_model.encoder.layer
        num_layers = len(encoder_layers)
        
        self.mae_model.config.mask_ratio = 0.0

        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : ALL parameters trainable")
            for p in self.mae_model.encoder.parameters():
                p.requires_grad = True

        else:
            for p in self.mae_model.encoder.parameters():
                p.requires_grad = False

            if self.backbone_learnable:
                k = max(1, min(self.last_k_trainable, num_layers))
                targets = set(range(num_layers - k, num_layers))
                print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {sorted(targets)}")

                for name, param in self.mae_model.encoder.named_parameters():
                    train_this = any(name.startswith(f"layer.{i}") for i in targets)
                    param.requires_grad = train_this
            else:
                print("→ backbone_learnable=False : ALL parameters frozen")

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W)
        Returns:
            cls_token: (B, hidden_dim)
        """
        outputs = self.mae_model(image)
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token
    
class dino_vitb_learnable_last_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone_learnable       = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable  = getattr(args, "backbone_full_learnable", False)
        self.last_k_trainable     = getattr(args, "backbone_learnable_layers", 1)

        print("Loading DINO backbone from torch.hub")
        self.dino_backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        
        encoder_layers = self.dino_backbone.blocks
        num_layers = len(encoder_layers)

        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : ALL parameters trainable")
            for p in self.dino_backbone.parameters():
                p.requires_grad = True

        else:
            for p in self.dino_backbone.parameters():
                p.requires_grad = False

            if self.backbone_learnable:
                k = max(1, min(self.last_k_trainable, num_layers))
                targets = set(range(num_layers - k, num_layers))
                print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {sorted(targets)}")

                for name, param in self.dino_backbone.named_parameters():
                    train_this = any(name.startswith(f"blocks.{i}") for i in targets)
                    param.requires_grad = train_this
            else:
                print("→ backbone_learnable=False : ALL parameters frozen")

    def forward(self, image):
        features = self.dino_backbone.get_intermediate_layers(image,n=1,)[0]
        cls_token    = features[:, 0, :]
        return cls_token
    
class franca_learnable_last_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone_learnable       = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable  = getattr(args, "backbone_full_learnable", False)
        self.last_k_trainable     = getattr(args, "backbone_learnable_layers", 1)
        self.vit_arch = getattr(args, "ViT_arch", "vit-l")

        print("Loading franca backbone from torch.hub")
        if self.vit_arch == "vit-l":
            self.franca_backbone = torch.hub.load('valeoai/Franca', 'franca_vitl14', weights='LAION', use_rasa_head=True)
        elif self.vit_arch == "vit-b-In21k":
            self.franca_backbone = torch.hub.load('valeoai/Franca', 'franca_vitb14', use_rasa_head=True)
        
        encoder_layers = self.franca_backbone.blocks
        num_layers = len(encoder_layers)

        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : ALL parameters trainable")
            for p in self.franca_backbone.parameters():
                p.requires_grad = True

        else:
            for p in self.franca_backbone.parameters():
                p.requires_grad = False

            if self.backbone_learnable:
                k = max(1, min(self.last_k_trainable, num_layers))
                targets = set(range(num_layers - k, num_layers))
                print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {sorted(targets)}")

                for name, param in self.franca_backbone.named_parameters():
                    train_this = any(name.startswith(f"blocks.{i}") for i in targets)
                    param.requires_grad = train_this
            else:
                print("→ backbone_learnable=False : ALL parameters frozen")

    def forward(self, image):
        features = self.franca_backbone.get_intermediate_layers(
            image,
            n=1,
            reshape=True,
            return_class_token=True
        )[0]
        cls_token    = features[1]
        return cls_token
    
class vjepa2_learnable_last_layer(nn.Module):
    """
    V-JEPA 2 backbone wrapper (Hugging Face Transformers).
    - freeze all params by default
    - optionally unfreeze only last-k encoder blocks
    - forward returns a single vector per video by default (pooler_output)
    """

    def __init__(self, args):
        super().__init__()

        self.backbone_learnable      = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable = getattr(args, "backbone_full_learnable", False)
        self.last_k_trainable        = getattr(args, "backbone_learnable_layers", 1)

        self.hf_repo = getattr(args, "vjepa2_repo", "facebook/vjepa2-vitl-fpc64-256")

        self.skip_predictor = getattr(args, "vjepa2_skip_predictor", True)

        # ---- load model ----
        self.vjepa2_model = AutoModel.from_pretrained(self.hf_repo)

        # ---- locate encoder layers safely ----
        layers = self._get_encoder_layers(self.vjepa2_model)
        num_layers = len(layers)

        # ---- requires_grad 設定 ----
        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : ALL parameters trainable")
            for p in self.vjepa2_model.parameters():
                p.requires_grad = True

        else:
            for p in self.vjepa2_model.parameters():
                p.requires_grad = False

            if self.backbone_learnable:
                k = max(1, min(self.last_k_trainable, num_layers))
                targets = list(range(num_layers - k, num_layers))
                print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {targets}")

                for i in targets:
                    for p in layers[i].parameters():
                        p.requires_grad = True
            else:
                print("→ backbone_learnable=False : ALL parameters frozen")

    @staticmethod
    def _get_encoder_layers(model: nn.Module) -> nn.ModuleList:
        backbone = getattr(model, "vjepa2", None)
        if backbone is None:
            backbone = model

        encoder = getattr(backbone, "encoder", None)
        if encoder is None:
            raise RuntimeError("Could not find encoder module (expected something like model.vjepa2.encoder).")

        layers = getattr(encoder, "layer", None)
        if layers is None:
            layers = getattr(encoder, "layers", None)

        if layers is None or not hasattr(layers, "__len__"):
            raise RuntimeError("Could not find encoder layers (expected encoder.layer or encoder.layers).")

        return layers

    def forward(self, pixel_values_videos: torch.Tensor, return_tokens: bool = False):
        outputs = self.vjepa2_model(
            pixel_values_videos=pixel_values_videos,
            skip_predictor=self.skip_predictor,
        )

        if return_tokens:
            return outputs.last_hidden_state  # (B, N, D)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output  # (B, D)

        return outputs.last_hidden_state.mean(dim=1)
    
class sam2_image_encoder_learnable_last_layers(nn.Module):

    def __init__(self, args):
        super().__init__()

        # ---- flags ----
        self.backbone_learnable = getattr(args, "backbone_learnable", True)
        self.backbone_full_learnable = getattr(args, "backbone_full_learnable", False)

        self.last_k_trainable = int(getattr(args, "backbone_learnable_layers", 2))
        self.last_n_stages = int(getattr(args, "backbone_learnable_stages", 1))
        self.neck_learnable = bool(getattr(args, "neck_learnable", True))

        self.sam2_model_id = getattr(args, "sam2_model_id", "facebook/sam2.1-hiera-large")
        
        self.pooling = getattr(args, "pooling", "fpn_all")
        # self.pooling = getattr(args, "pooling", "fpn_last")
        # self.pooling = getattr(args, "pooling", "backbone_last")

        # ---- load SAM2 from Hugging Face ----
        self.sam2 = AutoModel.from_pretrained(self.sam2_model_id)
        self.vision = self.sam2.vision_encoder  # Sam2VisionModel (backbone + neck)

        # ---- modules ----
        self.backbone = getattr(self.vision, "backbone", None)  # Sam2HieraDetModel
        self.neck = getattr(self.vision, "neck", None)          # Sam2VisionNeck
        if self.backbone is None or self.neck is None:
            raise AttributeError(
                "Could not find backbone or neck in SAM2 vision encoder. Please check the structure of self.sam2 and adjust the code if needed."
            )

        self._set_requires_grad()

    @staticmethod
    def _freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def _unfreeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    @staticmethod
    def _stage_ranges(stage_ends):
        starts = [0] + [e + 1 for e in stage_ends[:-1]]
        ends = list(stage_ends)
        return list(zip(starts, ends))

    @staticmethod
    def _tail_indices(start, end, k):
        if k <= 0:
            return list(range(start, end + 1))
        total = end - start + 1
        kk = max(1, min(k, total))
        return list(range(end - kk + 1, end + 1))

    def _set_requires_grad(self):
        self._freeze(self.vision)

        if self.backbone_full_learnable:
            print("→ backbone_full_learnable=True : vision_encoder ALL parameters trainable")
            self._unfreeze(self.vision)
            return

        if not self.backbone_learnable:
            print("→ backbone_learnable=False : vision_encoder backbone frozen")
            return

        # Regardless of backbone_learnable, control neck (FPN) learnability with neck_learnable flag.
        if self.neck_learnable:
            self._unfreeze(self.neck)
            print("→ neck_learnable=True : neck(FPN) trainable")
        else:
            print("→ neck_learnable=False : neck(FPN) frozen")

        blocks = getattr(self.backbone, "blocks", None)
        stage_ends = getattr(self.backbone, "stage_ends", None)
        if blocks is None or stage_ends is None:
            raise AttributeError("Could not find backbone.blocks or backbone.stage_ends. Please check the structure of self.sam2 and adjust the code if needed.")

        ranges = self._stage_ranges(stage_ends)  # 0-based stages
        num_stages = len(ranges)

        n = max(0, min(self.last_n_stages, num_stages))
        if n == 0:
            print("→ backbone_learnable_stages=0 : backbone all stages frozen")
            return

        target_stage_ids = list(range(num_stages - n, num_stages))
        train_block_indices = []

        for sid in target_stage_ids:
            s, e = ranges[sid]
            train_block_indices += self._tail_indices(s, e, self.last_k_trainable)

        train_block_indices = sorted(set(train_block_indices))
        for i in train_block_indices:
            self._unfreeze(blocks[i])

        stage_1based = [s + 1 for s in target_stage_ids]
        print(
            f"→ backbone_learnable=True : train backbone last {n} stage(s)={stage_1based}, "
            f"last_k_trainable={self.last_k_trainable} blocks={train_block_indices}"
        )

    def forward(self, pixel_values: torch.Tensor):
        out = self.vision(pixel_values=pixel_values)
        fpn_feats = out.fpn_hidden_states  # tuple of (B, 256, H_i, W_i)

        if self.pooling == "backbone_last":
            global_vec = out.last_hidden_state.mean(dim=(1, 2))  # (B, C)

        elif self.pooling == "fpn_last":
            global_vec = fpn_feats[-1].mean(dim=(2, 3))          # (B, 256)

        elif self.pooling == "fpn_all":
            pooled = [f.mean(dim=(2, 3)) for f in fpn_feats]     # list of (B, 256)
            global_vec = torch.cat(pooled, dim=1)                 # (B, 256*L)

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # return global_vec, fpn_feats
        return global_vec