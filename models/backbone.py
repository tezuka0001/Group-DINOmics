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
        if ("lora_A" in name) or ("lora_B" in name):
            p.requires_grad = True

    if unfix_last:
        k = max(1, min(learnable_layers, n))
        targets = set(range(n - k, n))
        print(f"→ backbone_learnable=True : train ONLY last {k} block(s) → {sorted(targets)}")

        for name, param in backbone.named_parameters():
            train_this = any(name.startswith(f"layer.{i}") for i in targets)
            param.requires_grad = train_this
                
        if verbose:
            print(f"[Unfix] layer.{sorted(targets)} are learnable")

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