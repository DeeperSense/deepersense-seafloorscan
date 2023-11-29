def cswin_commons():
    opts = [
        'MODEL.BACKBONE.TYPE', 'cswin',
        'MODEL.BACKBONE.SPLIT_SIZE', '(1, 2, 8, 8)',
        'MODEL.BACKBONE.MLP_RATIO', '4',
        'MODEL.BACKBONE.QKV_BIAS', 'False',
    ]
    return opts

def cswin_small():
    opts = cswin_commons()
    opts.extend([
        'MODEL.BACKBONE.NAME', 'cswin_small',
        'MODEL.BACKBONE.EMBED_DIM', '64',
        'MODEL.BACKBONE.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.BACKBONE.DEPTHS', '(2, 4, 32, 2)'
    ])
    return opts

def cswin_mini():
    opts = cswin_commons()
    opts.extend([
        'MODEL.BACKBONE.NAME', 'cswin_mini',
        'MODEL.BACKBONE.EMBED_DIM', '24',
        'MODEL.BACKBONE.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.BACKBONE.DEPTHS', '(1, 2, 21, 1)'
    ])
    return opts

#----------------------------------------------------

def lsda_commons():
    opts = [
        'MODEL.BACKBONE.TYPE', 'lsda',
        'MODEL.BACKBONE.GROUP_SIZE', '8',
        'MODEL.BACKBONE.MLP_RATIO', '2',
        'MODEL.BACKBONE.QKV_BIAS', 'False',
    ]
    return opts

def lsda_mini():
    opts = lsda_commons()
    opts.extend([
        'MODEL.BACKBONE.NAME', 'lsda_mini',
        'MODEL.BACKBONE.EMBED_DIM', '24',
        'MODEL.BACKBONE.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.BACKBONE.DEPTHS', '(3, 6, 12, 3)'
    ])
    return opts

#----------------------------------------------------

def sima_commons():
    opts = [
        'MODEL.BACKBONE.TYPE', 'sima',
        'MODEL.BACKBONE.MLP_RATIO', '2',
        'MODEL.BACKBONE.QKV_BIAS', 'False',
    ]
    return opts

def sima_mini():
    opts = sima_commons()
    opts.extend([
        'MODEL.BACKBONE.NAME', 'sima _mini',
        'MODEL.BACKBONE.EMBED_DIM', '24',
        'MODEL.BACKBONE.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.BACKBONE.DEPTHS', '(3, 6, 12, 3)'
    ])
    return opts

def sima_tiny():
    opts = sima_commons()
    opts.extend([
        'MODEL.BACKBONE.NAME', 'sima_tiny',
        'MODEL.BACKBONE.EMBED_DIM', '24',
        'MODEL.BACKBONE.NUM_HEADS', '(1, 2, 4, 8)',
        'MODEL.BACKBONE.DEPTHS', '(1, 3, 7, 1)'
    ])
    return opts

def sima_micro():
    opts = sima_commons()
    opts.extend([
        'MODEL.BACKBONE.NAME', 'sima_micro',
        'MODEL.BACKBONE.EMBED_DIM', '12',
        'MODEL.BACKBONE.NUM_HEADS', '(1, 2, 4, 8)',
        'MODEL.BACKBONE.DEPTHS', '(1, 3, 7, 1)'
    ])
    return opts

def sima_nano():
    opts = sima_commons()
    opts.extend([
        'MODEL.BACKBONE.NAME', 'sima_nano',
        'MODEL.BACKBONE.EMBED_DIM', '8',
        'MODEL.BACKBONE.NUM_HEADS', '(1, 2, 4, 8)',
        'MODEL.BACKBONE.DEPTHS', '(1, 1, 3, 1)'
    ])
    return opts
    