import torch
import torch.nn as nn
from torchvision import transforms
import timm
from model.helper.positional_encoding import PositionalEncode
from model.helper.sampling import sample
from model.baseline import BaselineGeneralSeqGenerator
from einops import rearrange, repeat
import logging

logger = logging.getLogger(__name__)

class FilterGeneralSeqGenerator(BaselineGeneralSeqGenerator):
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
        super(FilterGeneralSeqGenerator, self).__init__(
            backbone_model_name,
            backbone_ckpt,
            use_suppl,
            max_token_length,
            num_encoder_layers,
            num_decoder_layers,
            d_label,
            d_model,
            d_model_forward,
            tokenizer,
            init_weight,
            use_layout_encoder,
        )
        logger.info(f"Filtering model initialized with {max_token_length} tokens")
        
    def preprocess(self, x, device):
        x, y_true = super(FilterGeneralSeqGenerator, self).preprocess(x, device)
        x['patch_indices'] = x['patch_indices'].to(device)
        return x, y_true
    
    def forward_features(self, x, patch_indices):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)[:, 1:, :] # get rid of cls token
        x = torch.gather(x, 1, 
                patch_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        return x
    
    def forward(self, x):
        image = x['image']
        patch_indices = x['patch_indices']
        tgt = x['gt_seq']['seq']
        tgt_key_padding_mask = ~x['gt_seq']['mask']
        
        features = self.origin_suppl_fusion(image)
        features = self.forward_features(features, patch_indices)
        features = self.layout_token_encoder(features)
        
        outputs = self.layout_token_decoder(tgt, features, tgt_key_padding_mask)
        return outputs
    
    @torch.no_grad()
    def sample(self, x, sampling_cfg=None):
        bos_token_id = self.tokenizer.special_token_to_id('bos')
        pad_token_id = self.tokenizer.special_token_to_id('pad')
        eos_token_id = self.tokenizer.special_token_to_id('eos')
        token_mask = self.tokenizer.token_mask
        
        image = x["image"]
        patch_indices = x['patch_indices']
        start = torch.full((x['image'].shape[0], 1), bos_token_id, device=x['image'].device)
        tgt = start
        
        features = self.origin_suppl_fusion(image)
        features = self.forward_features(features, patch_indices)
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
        bos_token_id = self.tokenizer.special_token_to_id('bos')
        pad_token_id = self.tokenizer.special_token_to_id('pad')
        eos_token_id = self.tokenizer.special_token_to_id('eos')
        token_mask = self.tokenizer.token_mask
        
        image = x["image"]
        patch_indices = x['patch_indices']
        y_true = y_true.unsqueeze(-1)
        
        start = torch.full((x['image'].shape[0], 1), bos_token_id, device=x['image'].device)
        tgt = start
        
        features = self.origin_suppl_fusion(image)
        features = self.forward_features(features, patch_indices)
        features = self.layout_token_encoder(features)
        
        eos_recorder = torch.zeros(image.shape[0], device=image.device).bool()
        for i in range(self.max_token_length):
            if condition == "c":
                if i == y_true.shape[1]:
                    break
                if i % 3 == 0:
                    tgt = torch.cat([tgt, y_true[:, i]], dim=1)
                    continue
            elif condition == "partial":
                if i < 3:
                    tgt = torch.cat([tgt, y_true[:, i]], dim=1)
                    continue
            else:
                raise ValueError(f"Not supported condition: {condition}")
            
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
