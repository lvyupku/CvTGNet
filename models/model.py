from .gcn import *
from .cls_cvt import *
from tools import *
from torch.nn import Parameter



class CvtGNet(nn.Module):
    def __init__(self, config, num_classes, in_channel=300, t=0, adj_file=None, word_vec=None, sig=0.3):
        super(CvtGNet, self).__init__()
        msvit_spec = config.MODEL.SPEC
        self.features = ConvolutionalVisionTransformer(
            in_chans=3,
            num_classes=2048,
            act_layer=QuickGELU,
            norm_layer=partial(LayerNorm, eps=1e-5),
            init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
            spec=msvit_spec
        )

#         if config.MODEL.INIT_WEIGHTS:
#             self.feature.init_weights(
#                 config.MODEL.PRETRAINED,
#                 config.MODEL.PRETRAINED_LAYERS,
#                 config.VERBOSE
#             )

        self.num_classes = num_classes

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.5)
        #label fusion
        self.labelFusion = nn.Linear(num_classes, 2048)

        self.inp = torch.tensor(get_vec(word_vec)).cuda(non_blocking=True)
        _adj = gen_A(num_classes, adj_file, t, sig)
        self.A = Parameter(torch.from_numpy(_adj).float())

        self.fusion = nn.Linear(2048, 1024)
        self.classifity = nn.Linear(1024, num_classes)
        self.relu1 = nn.LeakyReLU(0.3)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature):
        feature = self.features(feature)
        #feature = self.pooling(feature)
        #feature = feature.view(feature.size(0), -1)


        # inp = inp[0]
        adj = gen_adj(self.A).detach()
        x = self.gc1(self.inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        x = self.labelFusion(x)

        #skip connection
        x = self.fusion(feature + x)
        x = self.relu1(x)
        x = self.classifity(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


class CvtG(nn.Module):
    def __init__(self, config, num_classes, in_channel=300, t=0, adj_file=None, word_vec=None, sig=0.3):
        super(CvtG, self).__init__()
        msvit_spec = config.MODEL.SPEC
        self.features = ConvolutionalVisionTransformer(
            in_chans=3,
            num_classes=2048,
            act_layer=QuickGELU,
            norm_layer=partial(LayerNorm, eps=1e-5),
            init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
            spec=msvit_spec
        )

#         if config.MODEL.INIT_WEIGHTS:
#             self.feature.init_weights(
#                 config.MODEL.PRETRAINED,
#                 config.MODEL.PRETRAINED_LAYERS,
#                 config.VERBOSE
#             )

        self.num_classes = num_classes

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.5)

        self.inp = torch.tensor(get_vec(word_vec)).cuda(non_blocking=True)
        _adj = gen_A(num_classes, adj_file, t, sig)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature):
        feature = self.features(feature)
        #feature = self.pooling(feature)
        #feature = feature.view(feature.size(0), -1)


        adj = gen_adj(self.A).detach()
        x = self.gc1(self.inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]



#单独Cvt
def getCvt(config):
    msvit_spec = config.MODEL.SPEC
    msvit = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=config.MODEL.NUM_CLASSES,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
        spec=msvit_spec
    )

#     if config.MODEL.INIT_WEIGHTS:
#         msvit.init_weights(
#             config.MODEL.PRETRAINED,
#             config.MODEL.PRETRAINED_LAYERS,
#             config.VERBOSE
#         )

    return msvit

#Cvt配GCN
def getCvtG(config):
    model = CvtG(config=config, num_classes=config.MODEL.NUM_CLASSES, t=0.4, adj_file=config.DATA.ADJ_PATH, word_vec=config.DATA.VEC_PATH)
    return model

def getCvtGNet(config):
    model = CvtGNet(config=config, num_classes=config.MODEL.NUM_CLASSES, t=0.4, adj_file=config.DATA.ADJ_PATH, word_vec=config.DATA.VEC_PATH)
    return model