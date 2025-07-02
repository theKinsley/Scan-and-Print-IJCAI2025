import torch
import torch.nn as nn
from torchvision import transforms
import timm
from model.helper.positional_encoding import PositionalEncode
from model.helper.sampling import sample
from einops import rearrange, repeat
import logging

logger = logging.getLogger(__name__)

class BaselineGeneralSeqDecoder(nn.Module):
    def __init__(self,
                 num_decoder_layers,
                 d_label,
                 d_model,
                 d_model_forward,
                 num_heads=8,
                 init_weight=False,
                 ):
        super(BaselineGeneralSeqDecoder, self).__init__()
        
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model,
                                       nhead=num_heads,
                                       batch_first=True,
                                       norm_first=True,
                                       dim_feedforward=d_model_forward),
            num_layers=num_decoder_layers,
        )
        self.embed = nn.Embedding(d_label, d_model)
        self.pos_embed = PositionalEncode("1D", d_model=d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_label, bias=False),
        )
        if init_weight:
            self.init_weight()
        
    def init_weight(self):
        logger.info(f"Initializing weights of {self.__class__.__name__} with Xavier uniform")
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for module in self.head:
            if isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, tgt, memory, tgt_key_padding_mask):
        tgt = self.embed(tgt)
        tgt = self.pos_embed(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
        tgt = self.transformer(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_is_causal=True,
        )
        outputs = self.head(tgt)
        
        return outputs
        

class BaselineGeneralSeqGenerator(nn.Module):
    def __init__(self,
                 backbone_model_name,
                 backbone_ckpt,
                 use_suppl,
                 max_token_length,
                 num_encoder_layers,
                 num_decoder_layers,
                 d_label,
                 d_model=384,
                 d_model_forward=2048,
                 tokenizer=None,
                 init_weight=False,
                 use_layout_encoder=True,
                 ):
        super(BaselineGeneralSeqGenerator, self).__init__()
        self._names = ['vit', 'layout_token_encoder', 'origin_suppl_fusion', 'layout_token_decoder']
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        
        # backbone vit model
        self.vit = timm.create_model(backbone_model_name, pretrained=False)
        if backbone_ckpt is not None:
            self.vit.load_state_dict(torch.load(backbone_ckpt, weights_only=True))
        for layer_i in range(num_encoder_layers, len(self.vit.blocks)):
            self.vit.blocks[layer_i] = nn.Identity()
        self.vit.head = nn.Identity()
        # self.pos_embed = PositionalEncode("2D", d_model=d_model) # not required for vit has positional encoding inherently
        if use_suppl:
            self.origin_suppl_fusion = nn.Conv2d(6, 3, 3, padding='same')
        else:
            self.origin_suppl_fusion = nn.Identity()
        
        cfg = self.vit.default_cfg
        self.vit_transforms = transforms.Compose([
            transforms.Resize(cfg['input_size'][1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['mean'], std=cfg['std'])
        ])
        
        # Layout token encoder
        if use_layout_encoder:
            self.layout_token_encoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.layout_token_encoder = nn.Identity()
        
        # Layout token decoder
        self.layout_token_decoder = BaselineGeneralSeqDecoder(num_decoder_layers, d_label, d_model, d_model_forward, init_weight=init_weight)
        
        self.loss = {
            'nll_loss': nn.CrossEntropyLoss(
                label_smoothing=0.1,
                ignore_index=self.tokenizer.special_token_to_id('pad'),
            )
        }
        
    def preprocess(self, x, device):
        gt_seq = self.tokenizer.encode(x['layout'])
        x["gt_seq"] = {
            "seq": gt_seq["seq"][:, :-1].to(device),
            "mask": gt_seq["mask"][:, :-1].to(device),
        }
        x["image"] = x["image"].to(device)
        y_true = gt_seq["seq"][:, 1:].to(device) # ignore bos token
        
        return x, y_true
    
    def train_loss(self, x, y_true):
        y_pred = self(x)
        losses = {}
        for loss_name, loss_fn in self.loss.items():
            loss = loss_fn(rearrange(y_pred, 'b n d -> b d n'), y_true)
            losses[loss_name] = loss
        return y_pred, losses
    
    def postprocess(self, y_pred):
        invalid = repeat(~self.tokenizer.token_mask[:y_pred.shape[1]], "n d -> b n d", b=y_pred.shape[0])
        y_pred[invalid] = -float("Inf")
        y_pred = y_pred.argmax(dim=-1)
        layout_pred = self.tokenizer.decode(y_pred)
        return layout_pred
    
    def forward_features(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)[:, 1:, :] # get rid of cls token
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        return x
    
    def forward(self, x):
        image = x['image']
        tgt = x['gt_seq']['seq']
        tgt_key_padding_mask = ~x['gt_seq']['mask']
        
        features = self.origin_suppl_fusion(image)
        features = self.forward_features(features)
        features = self.layout_token_encoder(features)
        
        outputs = self.layout_token_decoder(tgt, features, tgt_key_padding_mask)
        return outputs
    
    @torch.no_grad()
    def sample(self, x, sampling_cfg):
        bos_token_id = self.tokenizer.special_token_to_id('bos')
        pad_token_id = self.tokenizer.special_token_to_id('pad')
        eos_token_id = self.tokenizer.special_token_to_id('eos')
        token_mask = self.tokenizer.token_mask
        
        image = x["image"]
        start = torch.full((x['image'].shape[0], 1), bos_token_id, device=x['image'].device)
        tgt = start
        
        features = self.origin_suppl_fusion(image)
        features = self.forward_features(features)
        features = self.layout_token_encoder(features)
        
        eos_recorder = torch.zeros(image.shape[0], device=image.device).bool()
        for i in range(self.max_token_length):
            tgt_key_padding_mask = (tgt == pad_token_id)
            logits = self.layout_token_decoder(tgt, features, tgt_key_padding_mask)
            
            logits = rearrange(logits[:, i : i + 1], "b 1 c -> b c")
            invalid = repeat(~token_mask[i : i + 1], "1 c -> b c", b=logits.size(0))
            logits[invalid] = -float("Inf")
            
            output = sample(logits, sampling_cfg)
            tgt = torch.cat([tgt, output], dim=1)
            eos_recorder |= (output.squeeze(-1) == eos_token_id)
            if eos_recorder.all() and (i + 1) % self.tokenizer.N_var_per_element == 0:
                break
            
        outputs = tgt[:, 1:] # remove bos token
        return outputs
    
    @torch.no_grad()
    def sample_with_condition(self, x, y_true, sampling_cfg, condition):
        raise NotImplementedError
