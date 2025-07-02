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


class SEPointSeqTokenizer:
    def __init__(self,
                 class_feature,
                 var_order=('clses', 'x', 'y'),
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
        self.N_class_tokens = len(self.class_feature.names) * 2
        self.N_special_tokens = len(self.special_tokens)
        self.N_bbox_tokens = self.geo_quantization_num_bins if self.share_vocab else 2 * self.geo_quantization_num_bins
        self.N_total = self.N_class_tokens + self.N_bbox_tokens + self.N_special_tokens
        self.max_num_elements = max_num_elements * 2
        self._sp_token_to_id = {token: i + self.N_class_tokens + self.N_bbox_tokens for i, token in enumerate(self.special_tokens)}
        self._sp_id_to_token = {v: k for k, v in self._sp_token_to_id.items()}
        sp = ", ".join([f"[{k}] {v}" for k, v in self._sp_token_to_id.items()])
        logger.info(f"N_total: {self.N_total}, (class, bbox, special): ({self.N_class_tokens}, {self.N_bbox_tokens}, {self.N_special_tokens} ({sp}))")
        
        # for processing layout
        self.underlay_start_id = self.class_feature.str2int('underlay') * 2
        self.end_of_id = {2 * c : 2 * c + 1 for c in range(len(self.class_feature.names))}
        self.start_of_id = {v: k for k, v in self.end_of_id.items()}
        
    def special_token_to_id(self, token):
        return self._sp_token_to_id[token]
    
    def special_id_to_token(self, token_id):
        return self._sp_id_to_token[token_id]
    
    def detect_oov(self, token_ids):
        """
        parameters:
            token_ids: Dict[str, torch.Tensor]
                - clses: (B, N)
                - x: (B, N)
                - y: (B, N)
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
                - boxes: (B, N, 2)
                - clses: (B, N)
                - mask: (B, N)
        return:
            token_ids: Dict[str, torch.Tensor]
                - seq: (B, 3 * N + 1)
                - mask: (B, 3 * N + 1)
        """
        # transform boxes to x, y
        layouts['x'] = layouts['boxes'][:, :, 0]
        layouts['y'] = layouts['boxes'][:, :, 1]
        layouts['clses'] = layouts['clses']
        
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
            seq: torch.Tensor (B, 3 * N + 1)
        return:
            layouts: Dict[str, torch.Tensor]
                - boxes: (B, N, 2)
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
    
    def group_sepoints(self, clses):
        groups = []
        i = 0
        
        while i < len(clses):
            if clses[i] == self.underlay_start_id:
                nested_count = 0
                end_idx = i + 1
                while end_idx < len(clses):
                    if clses[end_idx] == self.underlay_start_id:
                        nested_count += 1
                    elif clses[end_idx] == self.end_of_id[self.underlay_start_id]:
                        if nested_count == 0:
                            break
                        nested_count -= 1
                    end_idx += 1
                
                nested_group = [self.underlay_start_id] + self.group_sepoints(clses[i+1:end_idx]) + [self.end_of_id[self.underlay_start_id]]
                groups.append(nested_group)
                    
                i = end_idx + 1
            else:
                groups.append(clses[i])
                i += 1
                
        return groups

    def arrange_elements(self, clses, x, y):
        layout = {'clses': [], 'boxes': []}
        stack = []
        i = 0 # index of clses
        j = 0 # index of x, y
        while i < len(clses):
            try:
                if isinstance(clses[i], list):
                    nested_layout = self.arrange_elements(clses[i][1:-1], x[j+1:j+len(clses[i])-1], y[j+1:j+len(clses[i])-1])
                    
                    if x[j] > x[j+len(clses[i])-1] or y[j] > y[j+len(clses[i])-1]:
                        raise ValueError(f"Box is invalid: {x[j]}, {y[j]}, {x[j+len(clses[i])-1]}, {y[j+len(clses[i])-1]}")
                    
                    b = [[x[j], y[j], x[j+len(clses[i])-1], y[j+len(clses[i])-1]]] + nested_layout['boxes']
                    c = [clses[i][0] // 2] + nested_layout['clses']
                    layout['boxes'].extend(b)
                    layout['clses'].extend(c)
                    j += len(clses[i])
                elif isinstance(clses[i], int):
                    if clses[i] in self.start_of_id.values():
                        stack.append((clses[i], j))
                    elif clses[i] in self.end_of_id.values():
                        for start_idx in range(len(stack)):
                            if (stack[start_idx][0] == self.start_of_id[clses[i]]) and \
                                (x[stack[start_idx][1]] < x[j] and y[stack[start_idx][1]] < y[j]):
                                break
                        else:
                            raise ValueError(f"Match failed, skip this element: {clses[i]}")
                        
                        b = [[x[stack[start_idx][1]], y[stack[start_idx][1]], x[j], y[j]]]
                        c = [stack[start_idx][0] // 2]
                        layout['boxes'].extend(b)
                        layout['clses'].extend(c)
                        stack.pop(start_idx)
                    else:
                        raise ValueError(f"Invalid class: {clses[i]}")
                    j += 1
                i += 1
            except Exception as e:
                logger.error(f"Error when arranging elements: {e}")
                if isinstance(clses[i], list):
                    j += len(clses[i])
                else:
                    j += 1
                i += 1
                continue

        return layout
    
    def process_layout(self, outputs, preview_size=(1, 1)):
        processed = []
        for i in range(len(outputs['clses'])):
            layout = {}
            
            clses = outputs['clses'][i][outputs['mask'][i]].tolist()
            x = outputs['x'][i][outputs['mask'][i]] * preview_size[0]
            x = x.tolist()
            y = outputs['y'][i][outputs['mask'][i]] * preview_size[1]
            y = y.tolist()
            
            clses = self.group_sepoints(clses)
            layout = self.arrange_elements(clses, x, y)
            
            if preview_size != (1, 1):
                layout['clses'] = self.class_feature.int2str(layout['clses'])
            
            processed.append(layout)
        return processed
    
def test():
    import datasets as ds
    from model.tokenizer import initialize_tokenizer
    name = "sepoint"
    class_feature = ds.ClassLabel(names=["text", "logo", "underlay"])
    tokenizer = initialize_tokenizer(name,
                                     class_feature=class_feature,
                                     var_order=('clses', 'x', 'y'),
                                     special_tokens=('pad', 'bos', 'eos'),
                                     max_num_elements=10,
                                     share_vocab=False)
    print(tokenizer.token_mask.shape, tokenizer.token_mask)
    boxes = torch.tensor([[[0.0643, 0.7893],
            [0.5380, 0.8293],
            [0.0487, 0.8467],
            [0.0916, 0.8640],
            [0.3489, 0.8973],
            [0.3899, 0.9200],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000]],

            [[0.8012, 0.1920],
            [0.9708, 0.2240],
            [0.4172, 0.2467],
            [0.9766, 0.2867],
            [0.6764, 0.3067],
            [0.6823, 0.3200],
            [0.9805, 0.3480],
            [0.9844, 0.3600],
            [0.0000, 0.0000],
            [0.0000, 0.0000]],

            [[0.0682, 0.7987],
            [0.9006, 0.8533],
            [0.0604, 0.8933],
            [0.5789, 0.8907],
            [0.3821, 0.9387],
            [0.9006, 0.9413],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000]],

            [[0.0897, 0.6427],
            [0.9084, 0.7227],
            [0.0975, 0.7333],
            [0.3314, 0.7333],
            [0.0994, 0.7680],
            [0.3509, 0.7680],
            [0.3138, 0.8080],
            [0.8538, 0.8133],
            [0.3314, 0.8360],
            [0.9064, 0.8360]]])
    clses = torch.tensor([[ 0,  1,  4,  0,  1,  5, -1, -1, -1, -1],
            [ 0,  1,  0,  1,  4,  0,  1,  5, -1, -1],
            [ 0,  1,  0,  0,  1,  1, -1, -1, -1, -1],
            [ 0,  1,  4,  4,  0,  0,  1,  1,  5,  5]])
    mask = torch.tensor([[ True,  True,  True,  True,  True,  True, False, False, False, False],
            [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
            [ True,  True,  True,  True,  True,  True, False, False, False, False],
            [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]])
    layout = {
        'boxes': boxes,
        'clses': clses,
        'mask': mask
    }
    token_ids = tokenizer.encode(layout)
    print(token_ids['seq'].shape)
    decoded = tokenizer.decode(token_ids['seq'][:, 1:])
    print('original layout:', layout)
    print('decoded layout:', decoded)

if __name__ == "__main__":
    test()