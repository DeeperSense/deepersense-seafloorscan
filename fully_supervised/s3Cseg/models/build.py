from models.registry import arch_entrypoints


def build_model(config):
    """Helper function to build the appropriate encoder/decoder architecture
    as per user specifications.
    """
    
    if config.MODEL.TYPE not in arch_entrypoints:
        raise ValueError(f'Unknown Architecture: {config.MODEL.TYPE}')
    model = arch_entrypoints.get(config.MODEL.TYPE)(config)

    return model
