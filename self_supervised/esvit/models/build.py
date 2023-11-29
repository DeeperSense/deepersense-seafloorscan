from models.registry import backbone_entrypoints
from models.wrappers import MultiCropWrapper
from models.heads import ProjHead


def build_model(config):
    """Helper function to build the appropriate student/teacher architecture
    as per user specifications.
    """

    if config.MODEL.BACKBONE.TYPE not in backbone_entrypoints:
        raise ValueError(f'Unknown Architecture: {config.MODEL.BACKBONE.TYPE}')
    head_config = config.MODEL.HEAD

    student = backbone_entrypoints.get(config.MODEL.BACKBONE.TYPE)(config)
    student = MultiCropWrapper(
        student,
        ProjHead(student.num_features, head_config.OUT_DIM,
                 head_config.MLP_RATIO, head_config.BOTTLENECK_RATIO,
                 head_config.USE_BN, head_config.NORM_LAST_LAYER),
        ProjHead(student.num_features, head_config.OUT_DIM,
                 head_config.MLP_RATIO, head_config.BOTTLENECK_RATIO,
                 head_config.USE_BN, head_config.NORM_LAST_LAYER)
    )

    teacher = backbone_entrypoints.get(config.MODEL.BACKBONE.TYPE)(config)
    teacher = MultiCropWrapper(
        teacher,
        ProjHead(teacher.num_features, head_config.OUT_DIM, 
                 head_config.MLP_RATIO, head_config.BOTTLENECK_RATIO,
                 head_config.USE_BN),
        ProjHead(teacher.num_features, head_config.OUT_DIM,
                 head_config.MLP_RATIO, head_config.BOTTLENECK_RATIO,
                 head_config.USE_BN)
    )

    return student, teacher
