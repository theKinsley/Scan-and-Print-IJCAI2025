if __name__ == "__main__":
    import sys
    sys.path.append('/home/xuxiaoyuan/Scan-and-Print')
    import model.helper.rich_log
    
import torch
from model.helper.bucketizer import LinearBucketizer
from copy import deepcopy
from einops import rearrange, repeat, reduce
import logging

logger = logging.getLogger(__name__)

class GeneralLayoutSeqTokenizer:
    def __init__(self,
                 class_feature,
                 var_order=('clses', 'center_x', 'center_y', 'width', 'height'),
                 special_tokens=('pad', 'bos', 'eos'),
                 share_vocab=False,
                 max_num_elements=10,
                 geo_quantization="Linear",
                 geo_quantization_num_bins=128,
                 geo_quantization_weight_path=None,
                 ):
        
        self.class_feature = class_feature
        self.var_order = var_order
        self.special_tokens = special_tokens
        self.share_vocab = share_vocab
        self.geo_quantization = geo_quantization
        
        assert 'pad' in self.special_tokens
        
        self.geo_quantization_num_bins = geo_quantization_num_bins
        if self.geo_quantization == "Linear":
            self.bucketizer = {key: LinearBucketizer(self.geo_quantization_num_bins) for key in self.var_order if key != 'clses'}
            logger.info(f"Initializing {self.__class__.__name__} with Linear quantization with {self.geo_quantization_num_bins} bins")
        elif self.geo_quantization == "KMeans":
            self.geo_quantization_weight_path = geo_quantization_weight_path
            # TODO: implement KMeans quantization
            raise NotImplementedError("KMeans quantization is not implemented")
            logger.info(f"Initializing {self.__class__.__name__} with KMeans quantization with {self.geo_quantization_num_bins} bins and weight path {self.geo_quantization_weight_path}")
        else:
            raise ValueError(f"Invalid geo_quantization: {self.geo_quantization}")
        
        self.N_var_per_element = len(self.var_order)
        self.N_class_tokens = len(self.class_feature.names)
        self.N_special_tokens = len(self.special_tokens)
        self.N_bbox_tokens = self.geo_quantization_num_bins if self.share_vocab else 4 * self.geo_quantization_num_bins
        self.N_total = self.N_class_tokens + self.N_bbox_tokens + self.N_special_tokens
        self.max_num_elements = max_num_elements
        self._sp_token_to_id = {token: i + self.N_class_tokens + self.N_bbox_tokens for i, token in enumerate(self.special_tokens)}
        self._sp_id_to_token = {v: k for k, v in self._sp_token_to_id.items()}
        sp = ", ".join([f"[{k}] {v}" for k, v in self._sp_token_to_id.items()])
        logger.info(f"N_total: {self.N_total}, (class, bbox, special): ({self.N_class_tokens}, {self.N_bbox_tokens}, {self.N_special_tokens} ({sp}))")
        
    def special_token_to_id(self, token):
        return self._sp_token_to_id[token]
    
    def special_id_to_token(self, token_id):
        return self._sp_id_to_token[token_id]
    
    def detect_oov(self, token_ids):
        """
        parameters:
            token_ids: Dict[str, torch.Tensor]
                - clses: (B, N)
                - center_x: (B, N)
                - center_y: (B, N)
                - width: (B, N)
                - height: (B, N)
        return:
            invalid: torch.Tensor[bool] (B, N)
        """
        valid = torch.full(token_ids["clses"].shape, True)
        for i, key in enumerate(self.var_order):
            if key == 'clses':
                valid = valid & (0 <= token_ids[key]) & (token_ids[key] < self.N_class_tokens)
            elif key in self.var_order:
                valid = valid & (0 <= token_ids[key]) & (token_ids[key] < self.geo_quantization_num_bins)
            else:
                raise ValueError(f"Invalid key: {key}")
        invalid = ~valid
        return invalid
    
    def detect_eos(self, token_ids):
        """
        parameters:
            token_ids: torch.Tensor (B, N)
        return:
            eos: torch.Tensor[bool] (B, N)
        """
        if "eos" in self.special_tokens:
            return torch.cumsum(token_ids == self.special_token_to_id('eos'), dim=1) > 0
        else:
            return torch.full(token_ids.shape, False)
        
    def encode(self, layouts):
        """
        parameters:
            layouts: Dict[str, torch.Tensor]
                - boxes: (B, N, 4)
                - clses: (B, N)
                - mask: (B, N)
        return:
            token_ids: Dict[str, torch.Tensor]
                - seq: (B, 5 * N + 1)
                - mask: (B, 5 * N + 1)
        """
        # transform boxes to center_x, center_y, width, height
        layouts['center_x'] = layouts['boxes'][:, :, 0]
        layouts['center_y'] = layouts['boxes'][:, :, 1]
        layouts['width'] = layouts['boxes'][:, :, 2]
        layouts['height'] = layouts['boxes'][:, :, 3]
        # layouts['clses'] = layouts['clses']
        
        data = {}
        for i, key in enumerate(self.var_order):
            if key == 'clses':
                data[key] = deepcopy(layouts[key])
            elif key in self.var_order:
                data[key] = self.bucketizer[key].encode(layouts[key])
                data[key] = data[key] + self.N_class_tokens
                if not self.share_vocab:
                    data[key] += (i - 1) * self.geo_quantization_num_bins
            else:
                raise ValueError(f"Invalid key: {key}")
            data[key][~layouts['mask']] = self.special_token_to_id('pad')
        
        data['mask'] = deepcopy(layouts['mask'])
        seq_len = reduce(data["mask"].int(), "b n -> b 1", reduction="sum")
        
        seq = torch.stack([data[key] for key in self.var_order], dim=-1)
        seq = rearrange(seq, "b n x -> b (n x)")
        mask = repeat(data["mask"], "b n -> b (n c)", c=self.N_var_per_element).clone()
        
        if 'bos' in self.special_tokens and 'eos' in self.special_tokens:
            indices = torch.arange(seq.shape[1]).unsqueeze(0)
            eos_mask = seq_len * self.N_var_per_element == indices
            seq[eos_mask] = self.special_token_to_id('eos')
            mask[eos_mask] = True
            
            bos = torch.full((seq.shape[0], 1), self.special_token_to_id('bos'))
            seq = torch.cat([bos, seq], dim=1)
            mask = torch.cat([torch.full((mask.shape[0], 1), True), mask], dim=1)
            
        return {
            'seq': seq,
            'mask': mask
        }
    
    def decode(self, seq):
        '''
        parameters:
            seq: torch.Tensor (B, 5 * N + 1)
        return:
            layouts: Dict[str, torch.Tensor]
                - boxes: (B, N, 4)
                - clses: (B, N)
                - mask: (B, N)
        '''
        
        seq = rearrange(
            deepcopy(seq),
            "b (s c) -> b s c",
            c=self.N_var_per_element,
        )
        outputs = {}
        for i, key in enumerate(self.var_order):
            outputs[key] = seq[:, :, i]
            if key == 'clses':
                continue
            else:
                outputs[key] = outputs[key] - self.N_class_tokens
                if not self.share_vocab:
                    outputs[key] = outputs[key] - (i - 1) * self.geo_quantization_num_bins
                outputs[key] = self.bucketizer[key].decode(outputs[key])
        
        # detect invalid tokens
        invalid = self.detect_eos(outputs['clses'])
        invalid = invalid | self.detect_oov(outputs)
        outputs['mask'] = torch.logical_not(invalid)
        
        for key in self.var_order:
            if key == 'clses':
                outputs[key][invalid] = -1
            else:
                outputs[key][invalid] = 0
        
        return outputs
    
    @property
    def token_mask(self):
        ng_tokens = ["bos", "mask"]  # shouldn't be predicted
        last = torch.tensor(
            [False if x in ng_tokens else True for x in self.special_tokens]
        )

        # get masks for geometry variables
        masks = {}
        if self.share_vocab:
            for key in self.var_order[1:]:
                masks[key] = torch.cat(
                    [
                        torch.full((self.N_class_tokens,), False),
                        torch.full((self.N_bbox_tokens,), True),
                        last,
                    ]
                )
        else:
            false_tensor = torch.full((self.N_bbox_tokens,), False)
            for i, key in enumerate(self.var_order[1:]):
                tensor = deepcopy(false_tensor)
                start, stop = (
                    i * self.geo_quantization_num_bins,
                    (i + 1) * self.geo_quantization_num_bins,
                )
                tensor[start:stop] = True
                masks[key] = torch.cat(
                    [torch.full((self.N_class_tokens,), False), tensor, last]
                )

        masks["clses"] = torch.cat(
            [
                torch.full((self.N_class_tokens,), True),
                torch.full((self.N_bbox_tokens,), False),
                last,
            ]
        )
        
        mask = torch.stack([masks[k] for k in self.var_order], dim=0)
        mask = repeat(mask, "x c -> (s x) c", s=self.max_num_elements)
        return mask
    
    def process_layout(self, outputs, preview_size=(1, 1)):
        processed = []
        for i in range(len(outputs['clses'])):
            layout = {}
            
            layout['clses'] = outputs['clses'][i][outputs['mask'][i]].tolist()
            if preview_size != (1, 1):
                layout['clses'] = self.class_feature.int2str(layout['clses'])
            
            center_x = outputs['center_x'][i][outputs['mask'][i]]
            center_y = outputs['center_y'][i][outputs['mask'][i]]
            
            width = outputs['width'][i][outputs['mask'][i]]
            height = outputs['height'][i][outputs['mask'][i]]
            layout['boxes'] = torch.stack([center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2], dim=1)
            layout['boxes'][:, ::2] *= preview_size[0]
            layout['boxes'][:, 1::2] *= preview_size[1]
            layout['boxes'] = layout['boxes'].tolist()
            
            processed.append(layout)
        return processed
    
def test():
    import datasets as ds
    from model.tokenizer import initialize_tokenizer
    name = "general"
    class_feature = ds.ClassLabel(names=["text", "logo", "underlay"])
    tokenizer = initialize_tokenizer(name,
                                     class_feature=class_feature,
                                     var_order=('clses', 'center_x', 'center_y', 'width', 'height'),
                                     special_tokens=('pad', 'bos', 'eos'),
                                     max_num_elements=10,
                                     share_vocab=False)
    print(tokenizer.token_mask.shape, tokenizer.token_mask)
    boxes = torch.tensor([[[0.5000, 0.4053, 1.0000, 0.2160],
         [0.5010, 0.3660, 0.8694, 0.1053],
         [0.5010, 0.4667, 0.7407, 0.0560],
         [0.2719, 0.9647, 0.4776, 0.0493],
         [0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.4990, 0.0813, 0.8304, 0.0560],
         [0.4873, 0.0847, 0.3002, 0.0493],
         [0.4961, 0.1867, 0.6959, 0.0987],
         [0.4990, 0.2933, 0.6589, 0.0880],
         [0.4951, 0.2913, 0.3821, 0.0573],
         [0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.2593, 0.1540, 0.4016, 0.2867],
         [0.1394, 0.3967, 0.1735, 0.2253],
         [0.1365, 0.3953, 0.1248, 0.2013],
         [0.3324, 0.4047, 0.0994, 0.1800],
         [0.1667, 0.8500, 0.2164, 0.0387],
         [0.1667, 0.8880, 0.2164, 0.0400],
         [0.1686, 0.9280, 0.2125, 0.0400]],

        [[0.2895, 0.0833, 0.5010, 0.0627],
         [0.2183, 0.1427, 0.3548, 0.0427],
         [0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000]]])
    clses = torch.tensor([[ 2,  0,  0,  0, -1, -1, -1],
        [ 2,  0,  0,  2,  0, -1, -1],
        [ 0,  2,  0,  0,  0,  0,  0],
        [ 0,  0, -1, -1, -1, -1, -1]])
    mask = torch.tensor([[ True,  True,  True,  True, False, False, False],
        [ True,  True,  True,  True,  True, False, False],
        [ True,  True,  True,  True,  True,  True,  True],
        [ True,  True, False, False, False, False, False]])
    layout = {
        'boxes': boxes,
        'clses': clses,
        'mask': mask
    }
    token_ids = tokenizer.encode(layout)
    print(token_ids['seq'].shape)
    decoded = tokenizer.decode(token_ids['seq'][:, 1:])
    print(layout)
    print(decoded)
    
if __name__ == "__main__":
    test()