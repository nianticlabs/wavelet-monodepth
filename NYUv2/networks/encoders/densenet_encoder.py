import torch.nn as nn


class DenseEncoder(nn.Module):
    def __init__(self, normalize_input=True, num_layers=161, pretrained=False):
        super(DenseEncoder, self).__init__()
        import torchvision.models as models

        model_dict = {161: models.densenet161,
                      121: models.densenet121,
                      201: models.densenet201,
                      169: models.densenet169}

        assert num_layers in model_dict, "Can't use any number of layers, should use from 121, 161, 169, 201"

        self.original_model = models.densenet161( pretrained=pretrained )
        self.normalize_input = normalize_input

        import numpy as np
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.num_ch_enc = [2208, 384, 192, 96, 96]
        self.num_ch_enc.reverse()

    def forward(self, x):
        if self.normalize_input:
            for t, m, s in zip(x, self.mean, self.std):
                t.sub(m).div(s)

        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features[3], features[4], features[6], features[8], features[11]


