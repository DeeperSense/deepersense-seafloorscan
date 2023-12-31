from yacs.config import CfgNode

_C = CfgNode()

_C.DATA = CfgNode()
_C.DATA.IMAGE_SIZE = 256
_C.DATA.IN_CHANS = 3
_C.DATA.NUM_CLASSES = 0
_C.DATA.MEAN = 0.
_C.DATA.STD = 1.

_C.AUG = CfgNode()

_C.AUG.CUTMIX = CfgNode()
_C.AUG.CUTMIX.ALPHA = 1.0
_C.AUG.CUTMIX.MINMAX = None

_C.AUG.MULTI_CROP = CfgNode()
_C.AUG.MULTI_CROP.GLOBAL_CROPS_SIZE = 224
_C.AUG.MULTI_CROP.GLOBAL_CROPS_SCALE = [0.4, 1]
_C.AUG.MULTI_CROP.LOCAL_CROPS_SIZE = 96
_C.AUG.MULTI_CROP.LOCAL_CROPS_SCALE = [0.05, 0.4]
_C.AUG.MULTI_CROP.LOCAL_CROPS_NUMBER = 10

_C.MODEL = CfgNode()

_C.MODEL.HEAD = CfgNode()
_C.MODEL.HEAD.OUT_DIM = 1024
_C.MODEL.HEAD.MLP_RATIO = 2
_C.MODEL.HEAD.BOTTLENECK_RATIO = 0.5
_C.MODEL.HEAD.NORM_LAST_LAYER = True
_C.MODEL.HEAD.USE_BN = False
_C.MODEL.HEAD.FREEZE_LAST_LAYER = 1

_C.MODEL.BACKBONE = CfgNode()
_C.MODEL.BACKBONE.TYPE = None
_C.MODEL.BACKBONE.NAME = None
_C.MODEL.BACKBONE.PATCH_SIZE = 4
_C.MODEL.BACKBONE.SPLIT_SIZE = None
_C.MODEL.BACKBONE.GROUP_SIZE = None
_C.MODEL.BACKBONE.DEPTHS = None
_C.MODEL.BACKBONE.NUM_HEADS = None
_C.MODEL.BACKBONE.EMBED_DIM = None
_C.MODEL.BACKBONE.MLP_RATIO = None
_C.MODEL.BACKBONE.QK_SCALE = None
_C.MODEL.BACKBONE.QKV_BIAS = None
_C.MODEL.BACKBONE.DROP_RATE = 0.0
_C.MODEL.BACKBONE.ATTN_DROP_RATE = 0.0
_C.MODEL.BACKBONE.DROP_PATH_RATE = 0.1
_C.MODEL.BACKBONE.USE_GHOST_FFN = False
_C.MODEL.BACKBONE.USE_MULTI_MERGE = False
_C.MODEL.BACKBONE.DENSE_PREDICTION = True

_C.TRAIN = CfgNode()
_C.TRAIN.EPOCHS = 300
_C.TRAIN.LR = 5e-4
_C.TRAIN.WARMUP_EPOCHS = 10
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.WEIGHT_DECAY = 0.04
_C.TRAIN.WEIGHT_DECAY_END = 0.4
_C.TRAIN.MOMENTUM_TEACHER = 0.996
_C.TRAIN.TEACHER_TEMP = 0.07
_C.TRAIN.WARMUP_TEACHER_TEMP = 0.04
_C.TRAIN.WARMUP_TEACHER_TEMP_EPOCHS = 30
_C.TRAIN.CLIP_GRAD = 3.0
_C.TRAIN.DEBUG = False

_C.EVAL = CfgNode()
_C.EVAL.TYPE = 'knn'
_C.EVAL.CLUSTERS = 1
_C.EVAL.NEIGHBORS = 20
_C.EVAL.TEMP = 0.7
