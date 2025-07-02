def initialize_tokenizer(tokenizer_name, **kwargs):
    if tokenizer_name == "general":
        from model.tokenizers.general import GeneralLayoutSeqTokenizer
        return GeneralLayoutSeqTokenizer(**kwargs)
    elif tokenizer_name == "sepoint":
        from model.tokenizers.sepoint import SEPointSeqTokenizer
        return SEPointSeqTokenizer(**kwargs)
    elif tokenizer_name == "sepoint_n":
        from model.tokenizers.sepoint_n import SEPointNSeqTokenizer
        return SEPointNSeqTokenizer(**kwargs)
    else:
        raise ValueError(f"Invalid tokenizer: {tokenizer_name}")
