import torch
import torch.optim as optim

import datasets as ds
import argparse
import sys
import os
import time
import json
import logging

logger = logging.getLogger(__name__)

def str2tuple(s):
    s = eval(s)
    assert isinstance(s, tuple)
    return s

def str2bool(s):
    s = eval(s)
    assert isinstance(s, bool)
    return s

class BaseCfg(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
class EvalArgumentCfg(BaseCfg):
    def __init__(self, **kwargs):
        super(EvalArgumentCfg, self).__init__(**kwargs)
        self.inference = True
        for key, value in self.parse_args().__dict__.items():
            setattr(self, key, value)
        
        sys.path.append(self.root)
        import model.helper.rich_log
        from model.utils import set_seed
        set_seed(self.seed)
        self.set_seed = set_seed
        
        with open(os.path.join(self.save_path, 'config.json'), 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            if key in ['root', 'gpu', 'seed', 'temperature', 'top_k']: # enable to override
                continue
            setattr(self, key, value)
        
        config = {}
        config["dataset"] = ['dataset_root', 'dataset', 'suppl_type']
        config["backbone"] = ['model_name', 'backbone_model_name']
        config["inference"] = ['temperature', 'top_k', 'num_samples', 'condition']
        config["tokenizer"] = ['num_bin_geometry', 'tokenizer_name', 'var_order', 'special_tokens', 'share_vocab']
        
        if 'augment' not in self.__dict__:
            self.augment = False
            
        if self.augment:
            config["augmentation_setting"] = ['num_augment', 'similarity_type']
            self.filtering = True
            config["filter_setting"] = ['patch_indices_dir', 'topk', 'pick_topk', 'P']
        elif self.model_name == "filtering":
            self.filtering = True
            config["filter_setting"] = ['patch_indices_dir', 'topk', 'pick_topk', 'P']
        else:
            self.filtering = False
            
        for name, keys in config.items():
            self.preprocess_args(name, keys)
    
    def refresh_seed(self, increment=0):
        if increment > 0:
            self.set_seed(self.seed + increment)
        self.save_pkl_path = self.save_pkl_paths[increment]
        logger.info(f"Using seed {increment + 1}: {self.seed + increment}")
    
    def preprocess_args(self, name, keys):
        logger.info(f"{name.capitalize()} config".center(120, '='))
        info = "\n".join([f"{key}: {getattr(self, key)}" for key in keys])
        logger.info(info)
        
        if name == "dataset":
            if self.dataset == 'pku':
                self.num_element_classes = 3
                self.class_feature = ds.ClassLabel(names=["text", "logo", "underlay"])
                self.original_size = (513, 750)
            elif self.dataset == 'cgl':
                self.num_element_classes = 4
                self.class_feature = ds.ClassLabel(names=["logo", "text", "underlay", "embellishment"])
                self.original_size = (514, 751)
            else:
                raise ValueError(f"Invalid dataset: {self.dataset}")
        elif name == "backbone":
            if self.suppl_type == "None":
                self.suppl_type = None
            self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
            self.backbone_ckpt = None
            self.init_weight = False
        elif name == "inference":
            if self.condition == "uncond":
                self.eval_dir = "eval"
            else:
                self.eval_dir = "eval_" + self.condition
                
            os.makedirs(os.path.join(self.save_path, self.eval_dir), exist_ok=True)
                
            checkpoint_dir = os.path.join(self.save_path, 'checkpoint')
            weight = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')][0]
            weight_epoch = weight.split('_')[1]
            self.weight_path = os.path.join(checkpoint_dir, weight)
            
            self.save_pkl_paths = [
                os.path.join(self.save_path, self.eval_dir, f"{weight_epoch}_seed_{i}_temp_{self.temperature}_topk_{self.top_k}_"+"{}.pkl")
                for i in range(self.num_samples)
            ]
            self.sampling_cfg = SamplingCfg(temperature=self.temperature, top_k=self.top_k)
        elif name == "tokenizer":
            from model.tokenizer import initialize_tokenizer
            self.tokenizer = initialize_tokenizer(self.tokenizer_name,
                                                  class_feature=self.class_feature,
                                                  var_order=self.var_order,
                                                  special_tokens=self.special_tokens,
                                                  share_vocab=self.share_vocab,
                                                  max_num_elements=self.max_num_elements,
                                                  geo_quantization_num_bins=self.num_bin_geometry)
            self.max_token_length = self.tokenizer.max_num_elements * self.tokenizer.N_var_per_element
            self.d_label = self.tokenizer.N_total
        elif name == "filter_setting":
            pass
        elif name == "augmentation_setting":
            pass
        else:
            raise ValueError(f"Invalid config name: {name}")
        
    def load_model(self):
        from model.load_model import load_model
        model = load_model(self)
        model.to(self.device)
        
        logger.info(f"Loading model {self.model_name} to {self.device} with checkpoint {self.weight_path}")
        model.load_state_dict(torch.load(self.weight_path, weights_only=True))
        
        return model
    
    def load_dataloader(self):
        from dataloader.init import initialize_dataloader
        LayoutDataset, train_collate_fn = initialize_dataloader(self)
        if self.filtering:
            from dataloader.filter_setting import load_filter_setting
            LayoutDataset, train_collate_fn, test_collate_fn = load_filter_setting(LayoutDataset, train_collate_fn)
            if self.augment:
                from dataloader.augment_setting import load_augment_setting
                augment_train_collate_fn = load_augment_setting(self, train_collate_fn)
                return LayoutDataset, augment_train_collate_fn, train_collate_fn, test_collate_fn
            else:
                return LayoutDataset, train_collate_fn, train_collate_fn, test_collate_fn
        else:
            return LayoutDataset, train_collate_fn, train_collate_fn, None
    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--root', type=str, default='/home/xuxiaoyuan/Scan-and-Print')
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--save_path', type=str, required=True)
        
        # inference config
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--top_k', type=int, default=5)
        parser.add_argument('--num_samples', type=int, default=3)
        parser.add_argument('--condition', type=str, default='uncond', choices=['uncond', 'c', 'partial'])
        
        return parser.parse_args()
    
        
class TrainArgumentCfg(BaseCfg):
    def __init__(self, **kwargs):
        super(TrainArgumentCfg, self).__init__(**kwargs)
        for key, value in self.parse_args().__dict__.items():
            setattr(self, key, value)
            
        if self.restart:
            assert self.save_path is not None, "save_path is required when restart is True"
            assert os.path.exists(self.save_path), f"save_path does not exist: {self.save_path}"
            
            with open(os.path.join(self.save_path, 'config.json'), 'r') as f:
                config = json.load(f)
            for key, value in config.items():
                if key in ['gpu']: # enable to override
                    continue
                setattr(self, key, value)
            
        sys.path.append(self.root)
        import model.helper.rich_log
        from model.utils import set_seed
        set_seed(self.seed)
        
        if not self.restart:
            self.save_path = os.path.join(self.root, 'history', f"{self.model_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}", self.dataset)    
            os.makedirs(self.save_path, exist_ok=True)
            logger.info(f"Save path: {self.save_path}")
            self.save_config()
        else:
            logger.info(f"Restarting training from save path: {self.save_path}")
        
        config = {}
        config["dataset"] = ['dataset_root', 'dataset', 'suppl_type', 'max_num_elements']
        config["backbone"] = ['model_name', 'backbone_model_name', 'backbone_ckpt', 'num_encoder_layers', 'num_decoder_layers', 'init_weight', 'use_layout_encoder']
        config["training"] = ['num_epochs', 'batch_size', 'lr', 'vis_preview']
        config["inference"] = ['temperature', 'top_k']
        config["tokenizer"] = ['num_bin_geometry', 'tokenizer_name', 'var_order', 'special_tokens', 'share_vocab']
        if self.augment:
            config["augmentation_setting"] = ['num_augment', 'similarity_type']
            self.filtering = True
            config["filter_setting"] = ['patch_indices_dir', 'topk', 'pick_topk', 'P']
        elif self.model_name == "filtering":
            self.filtering = True
            config["filter_setting"] = ['patch_indices_dir', 'topk', 'pick_topk', 'P']
        else:
            self.filtering = False
        
        for name, keys in config.items():
            self.preprocess_args(name, keys)
        
    def save_config(self):
        config_path = os.path.join(self.save_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def preprocess_args(self, name, keys):
        logger.info(f"{name.capitalize()} config".center(120, '='))
        info = "\n".join([f"{key}: {getattr(self, key)}" for key in keys])
        logger.info(info)
        
        if name == "dataset":
            if self.dataset == 'pku':
                self.num_element_classes = 3
                self.class_feature = ds.ClassLabel(names=["text", "logo", "underlay"])
                self.original_size = (513, 750)
            elif self.dataset == 'cgl':
                self.num_element_classes = 4
                self.class_feature = ds.ClassLabel(names=["logo", "text", "underlay", "embellishment"])
                self.original_size = (514, 751)
            else:
                raise ValueError(f"Invalid dataset: {self.dataset}")
        elif name == "backbone":
            if self.suppl_type == "None":
                self.suppl_type = None
            self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        elif name == "training":
            os.makedirs(os.path.join(self.save_path, 'checkpoint'), exist_ok=True)
            if self.vis_preview:
                self.preview_size = (240, 350)
                os.makedirs(os.path.join(self.save_path, 'vis'), exist_ok=True)
        elif name == "inference":
            self.sampling_cfg = SamplingCfg(temperature=self.temperature, top_k=self.top_k)
        elif name == "tokenizer":
            from model.tokenizer import initialize_tokenizer
            self.tokenizer = initialize_tokenizer(self.tokenizer_name,
                                                  class_feature=self.class_feature,
                                                  var_order=self.var_order,
                                                  special_tokens=self.special_tokens,
                                                  share_vocab=self.share_vocab,
                                                  max_num_elements=self.max_num_elements,
                                                  geo_quantization_num_bins=self.num_bin_geometry
                                                  )
            self.max_token_length = self.tokenizer.max_num_elements * self.tokenizer.N_var_per_element
            self.d_label = self.tokenizer.N_total
        elif name == "filter_setting":
            pass
        elif name == "augmentation_setting":
            pass
        else:
            raise ValueError(f"Invalid config name: {name}")
    
    def load_model_and_optimizer(self):
        from model.load_model import load_model
        logger.info(f"Loading model {self.model_name} to {self.device}")
        model = load_model(self)
        model.to(self.device)
        vit_params = list(filter(lambda name: name[0].startswith('vit') or name[0].startswith('module.vit'), model.named_parameters()))
        assert len(vit_params) >= 1, "There should be at least one vit module"
        other_params = list(filter(lambda name: not name[0].startswith('vit') and not name[0].startswith('module.vit'), model.named_parameters()))
        
        optimizer = optim.AdamW(
            [
                {"params": [p for _, p in vit_params], "lr": self.lr * 0.1, "weight_decay": 1e-2},
                {"params": [p for _, p in other_params], "lr": self.lr}
            ]
        )
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14], gamma=0.1)
        
        return model, optimizer, scheduler
    
    def load_dataloader(self):
        from dataloader.init import initialize_dataloader
        LayoutDataset, train_collate_fn = initialize_dataloader(self)
        if self.filtering:
            from dataloader.filter_setting import load_filter_setting
            LayoutDataset, train_collate_fn, test_collate_fn = load_filter_setting(LayoutDataset, train_collate_fn)
            if self.augment:
                from dataloader.augment_setting import load_augment_setting
                augment_train_collate_fn = load_augment_setting(self, train_collate_fn)
                return LayoutDataset, augment_train_collate_fn, train_collate_fn, test_collate_fn
            else:
                return LayoutDataset, train_collate_fn, train_collate_fn, test_collate_fn
        else:
            return LayoutDataset, train_collate_fn, train_collate_fn, None
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--root', type=str, default='/home/xuxiaoyuan/Scan-and-Print')
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--seed', type=int, default=42)
        
        # restart training: be careful with this
        parser.add_argument('--restart', type=str2bool, default=False)
        parser.add_argument('--save_path', type=str, default=None)
        
        # dataset config
        parser.add_argument('--dataset_root', type=str, default='/home/xuxiaoyuan/calg_dataset/')
        parser.add_argument('--dataset', type=str, default='pku')
        parser.add_argument('--suppl_type', type=str, default='saliency')
        parser.add_argument('--max_num_elements', type=int, default=10)
        
        # backbone config
        parser.add_argument('--model_name', type=str, default='baseline')
        parser.add_argument('--backbone_model_name', type=str, default='deit3_small_patch16_224.fb_in22k_ft_in1k')
        parser.add_argument('--num_encoder_layers', type=int, default=8)
        parser.add_argument('--num_decoder_layers', type=int, default=4)
        parser.add_argument('--init_weight', type=str2bool, default=True)
        parser.add_argument('--d_model', type=int, default=384)
        parser.add_argument('--d_model_forward', type=int, default=2048)
        parser.add_argument('--use_layout_encoder', type=str2bool, default=True)
        
        # training config
        parser.add_argument('--num_epochs', type=int, default=15)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--vis_preview', type=str2bool, default=False)
        
        # inference config
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--top_k', type=int, default=5)
        
        # tokenizer config
        parser.add_argument('--num_bin_geometry', type=int, default=128)
        parser.add_argument('--tokenizer_name', type=str, default='general')
        parser.add_argument('--var_order', type=str2tuple, default=('clses', 'center_x', 'center_y', 'width', 'height'))
        parser.add_argument('--special_tokens', type=str2tuple, default=('pad', 'bos', 'eos'))
        parser.add_argument('--share_vocab', type=str2bool, default=False)
        
        # filter setting config
        parser.add_argument('--topk', type=int, default=96)
        parser.add_argument('--pick_topk', type=int, default=96)
        parser.add_argument('--P', type=int, default=14)
        
        # augment setting config
        parser.add_argument('--augment', type=str2bool, default=False)
        parser.add_argument('--num_augment', type=int, default=32)
        parser.add_argument('--similarity_type', type=str, default='cosine')
        
        args = parser.parse_args()
        args.backbone_ckpt = os.path.join(args.root, "cache", args.backbone_model_name + ".bin")
        args.patch_indices_dir = os.path.join(args.root, "cache", "topk_patch_indices")
        
        return args
        
class SamplingCfg(BaseCfg):
    def __init__(self, **kwargs):
        super(SamplingCfg, self).__init__(name='top_k', **kwargs)
        
def test():
    # TrainArgumentCfg()
    args = EvalArgumentCfg()
    args.load_model()
    
if __name__ == "__main__":
    test()