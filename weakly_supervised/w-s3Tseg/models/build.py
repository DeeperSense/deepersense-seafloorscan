import math
import torch
import torch.nn as nn

from models.classifier import Classifier

from models.registry import encoder_entrypoints
from models.registry import decoder_entrypoints


class Model(nn.Module):

    def __init__(self, encoder, decoder, classifier):
        super(Model, self).__init__()
        self.classifier = classifier
        self.encoder = encoder
        self.decoder = decoder

    def module_groups(self):
        return self.classifier, self.encoder, self.decoder

    def forward(self, x, mode='train', isClassificationOnly=False):
        out = {"cls": None, "cams": None, "seg": None}
        
        features, trace = self.encoder(x)
        
        B, L, C = features.shape
        H = W = int(math.sqrt(L))
        features_= features.view(B, H, W, C).permute(0, 3, 1, 2)
        
        if isClassificationOnly:
            if mode != 'train':
                out["cams"] = self.classifier.calculate_cam(features_)
            out["cls"] = self.classifier(features_)
        else:
            if mode != 'infer':
                if mode == 'eval':
                    out["cams"] = self.classifier.calculate_cam(features_)
                out["cls"] = self.classifier(features_)
            out["seg"] = self.decoder(features, trace[::-1])
            
        return out

    def forward_cam(self, x):
        # input: list of 2 tensors of shape (B,C,H,W)
        # output: tensor of shape (B,K,h,w)
        with torch.set_grad_enabled(False):
            features = [self.encoder(x_)[0] for x_ in x]

            B, L, C = features[0].shape
            h = w = int(math.sqrt(L))
            
            features = [f.view(B, h, w, C).permute(0, 3, 1, 2) for f in features]
            return self.classifier.calculate_cam(features)


def build_model(config):
    """Helper function to build the appropriate encoder/decoder architecture
    as per user specifications.
    """

    if config.MODEL.ENCODER.TYPE not in encoder_entrypoints:
        raise ValueError(f'Unknown Encoder: {config.MODEL.ENCODER.TYPE}')
    encoder = encoder_entrypoints.get(config.MODEL.ENCODER.TYPE)(config)
    
    config.defrost()
    if config.MODEL.DECODER.TYPE == 'symmetric':
        config.MODEL.DECODER.TYPE = config.MODEL.ENCODER.TYPE
        config.MODEL.DECODER.NAME = config.MODEL.ENCODER.NAME
    config.MODEL.DECODER.IN_FEATURES = encoder.num_features
    config.freeze()

    if config.MODEL.DECODER.TYPE not in decoder_entrypoints:
        raise ValueError(f'Unknown Decoder: {config.MODEL.DECODER.TYPE}')
    decoder = decoder_entrypoints.get(config.MODEL.DECODER.TYPE)(config)
    
    classifier = Classifier(encoder.num_features, config.DATA.NUM_CLASSES)

    model = Model(encoder, decoder, classifier)
    return model
