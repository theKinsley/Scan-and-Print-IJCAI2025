from model.baseline import BaselineGeneralSeqGenerator
from model.filtering import FilterGeneralSeqGenerator
from model.utils import compute_params

def load_model(args):
    if args.model_name == 'baseline':
        return BaselineGeneralSeqGenerator(
            backbone_model_name=args.backbone_model_name,
            backbone_ckpt=args.backbone_ckpt,
            use_suppl=True if args.suppl_type is not None else False,
            max_token_length=args.max_token_length,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_label=args.d_label,
            d_model=args.d_model,
            d_model_forward=args.d_model_forward,
            tokenizer=args.tokenizer,
            init_weight=args.init_weight,
            use_layout_encoder=args.use_layout_encoder,
            )
    elif args.model_name == 'filtering':
        assert args.filtering, "Filtering is not enabled?"
        return FilterGeneralSeqGenerator(
            backbone_model_name=args.backbone_model_name,
            backbone_ckpt=args.backbone_ckpt,
            use_suppl=True if args.suppl_type is not None else False,
            max_token_length=args.max_token_length,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_label=args.d_label,
            d_model=args.d_model,
            d_model_forward=args.d_model_forward,
            tokenizer=args.tokenizer,
            init_weight=args.init_weight,
            use_layout_encoder=args.use_layout_encoder,
            )
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")
    
    logger.info(f"Model parameters: {compute_params(model):.3f}M")
    return model
