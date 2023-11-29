#===================== Encoders =====================

def sima_commons():
    opts = [
        'MODEL.ENCODER.TYPE', 'sima',
        'MODEL.ENCODER.MLP_RATIO', '2',
        'MODEL.ENCODER.QKV_BIAS', 'False',
    ]
    return opts

def sima_mini():
    opts = sima_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'sima _mini',
        'MODEL.ENCODER.EMBED_DIM', '24',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.DEPTHS', '(3, 6, 12, 3)'
    ])
    return opts

def sima_tiny():
    opts = sima_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'sima_tiny',
        'MODEL.ENCODER.EMBED_DIM', '24',
        'MODEL.ENCODER.NUM_HEADS', '(1, 2, 4, 8)',
        'MODEL.ENCODER.DEPTHS', '(1, 3, 7, 1)'
    ])
    return opts

def sima_micro():
    opts = sima_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'sima_micro',
        'MODEL.ENCODER.EMBED_DIM', '12',
        'MODEL.ENCODER.NUM_HEADS', '(1, 2, 4, 8)',
        'MODEL.ENCODER.DEPTHS', '(1, 3, 7, 1)'
    ])
    return opts

def sima_nano():
    opts = sima_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'sima_nano',
        'MODEL.ENCODER.EMBED_DIM', '8',
        'MODEL.ENCODER.NUM_HEADS', '(1, 2, 4, 8)',
        'MODEL.ENCODER.DEPTHS', '(1, 1, 3, 1)'
    ])
    return opts

#===================== Decoders =====================

def symmetric():
    opts = [
        'MODEL.DECODER.TYPE', 'symmetric',
        'MODEL.DECODER.SKIP_TYPE', 'cat',
        'MODEL.DECODER.EXPAND_FIRST', False,
    ]
    return opts

def atrous():
    opts = [
        'MODEL.DECODER.TYPE', 'atrous',
        'MODEL.DECODER.NAME', 'segformer',
        'MODEL.DECODER.EMBED_DIM', 24,
        'MODEL.DECODER.DEPTH', 4,
        'MODEL.DECODER.FUSE_OP', 'cat',
    ]
    return opts
