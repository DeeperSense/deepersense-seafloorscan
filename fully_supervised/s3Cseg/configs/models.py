def ecnet():
    opts = [
        'MODEL.TYPE', 'ecnet',
    ]
    return opts


def dcnet():
    opts = [
        'MODEL.TYPE', 'dcnet',
    ]
    return opts


def rtseg():
    opts = [
        'MODEL.TYPE', 'rtseg',
        'MODEL.EXPANSION_FACTOR', 6,
    ]
    return opts
