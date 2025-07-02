import logging

logger = logging.getLogger(__name__)

def initialize_dataloader(args):
    if args.tokenizer_name == "general":
        if args.suppl_type == "saliency" or args.suppl_type == "density":
            from dataloader.general_seq import ImageSupplLayoutDataset, train_collate_fn
            logger.info(f"Using ImageSupplLayoutDataset: {args.suppl_type}")
            return ImageSupplLayoutDataset, train_collate_fn
        elif args.suppl_type == None or args.suppl_type == "None":
            from dataloader.general_seq import ImageLayoutDataset, train_collate_fn
            logger.info("Using ImageLayoutDataset")
            return ImageLayoutDataset, train_collate_fn
        else:
            raise NotImplementedError(f"Invalid suppl_type: {args.suppl_type}")
    elif args.tokenizer_name == "sepoint":
        if args.suppl_type == "saliency" or args.suppl_type == "density":
            from dataloader.sepoint_seq import ImageSupplLayoutDataset, train_collate_fn
            logger.info(f"Using ImageSupplLayoutDataset: {args.suppl_type}")
            return ImageSupplLayoutDataset, train_collate_fn
        elif args.suppl_type == None or args.suppl_type == "None":
            from dataloader.sepoint_seq import ImageLayoutDataset, train_collate_fn
            logger.info("Using ImageLayoutDataset")
            return ImageLayoutDataset, train_collate_fn
        else:
            raise NotImplementedError(f"Invalid suppl_type: {args.suppl_type}")
    else:
        raise NotImplementedError(f"Invalid tokenizer_name: {args.tokenizer_name}")