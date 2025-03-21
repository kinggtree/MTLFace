import torch.nn.functional as F
import torch
import tqdm
from torch import nn

import torchvision.models as models


###########
from .irse import IResNet


class SPPModule(nn.Module):
    def __init__(self, pool_mode='avg', sizes=(1, 2, 3, 6)):
        super().__init__()
        if pool_mode == 'avg':
            pool_layer = nn.AdaptiveAvgPool2d
        elif pool_mode == 'max':
            pool_layer = nn.AdaptiveMaxPool2d
        else:
            raise NotImplementedError

        self.pool_blocks = nn.ModuleList([
            nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.pool_blocks]
        x = torch.cat(xs, dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x


class AttentionModule(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super(AttentionModule, self).__init__()
        kernel_size = 7
        pool_size = (1, 2, 3)
        self.avg_spp = SPPModule('avg', pool_size)
        self.max_spp = SPPModule('max', pool_size)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                      dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

        _channels = channels * int(sum([x ** 2 for x in pool_size]))
        self.channel = nn.Sequential(
            nn.Conv2d(
                _channels, _channels // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                _channels // reduction, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_input = self.avg_spp(x) + self.max_spp(x)
        channel_scale = self.channel(channel_input)

        spatial_input = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        spatial_scale = self.spatial(spatial_input)

        x_age = (x * channel_scale + x * spatial_scale) * 0.5

        x_id = x - x_age

        return x_id, x_age
    



class AIResNet(IResNet):
    def __init__(self, input_size, num_layers, mode='ir', **kwargs):
        super(AIResNet, self).__init__(input_size, num_layers, mode)
        self.fsm = AttentionModule()
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(512 * (input_size // 16) ** 2, 512),
            nn.BatchNorm1d(512))
        self._initialize_weights()

    def forward(self, x, return_age=False, return_shortcuts=False):
        x_1 = self.input_layer(x)
        x_2 = self.block1(x_1)
        x_3 = self.block2(x_2)
        x_4 = self.block3(x_3)
        x_5 = self.block4(x_4)
        x_id, x_age = self.fsm(x_5)
        embedding = self.output_layer(x_id)
        if return_shortcuts:
            return x_1, x_2, x_3, x_4, x_5, x_id, x_age
        if return_age:
            return embedding, x_id, x_age
        return embedding
    

class DenseNetAIFR(nn.Module):
    def __init__(self, input_size=224, drop_rate=0.4):
        super(DenseNetAIFR, self).__init__()
        # 导入预训练的DenseNet121
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features

        # 因为DenseNet121的特征图尺寸为7x7时特征通道数为1024
        final_feature_map_size = (input_size // 32) ** 2
        final_channels = 1024

        # 注意力模块
        self.attention = AttentionModule(channels=final_channels)  

        # 全连接层，这里我们保持与AIResNet类似的结构
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(final_channels),
            nn.Dropout(drop_rate),
            nn.Flatten(),
            nn.Linear(final_channels * final_feature_map_size, 512),
            nn.BatchNorm1d(512))

        self._initialize_weights()

    def forward(self, x, return_age=False, return_shortcuts=False):
        features = self.features(x)
        x_id, x_age = self.attention(features)
        embedding = self.output_layer(x_id)

        if return_shortcuts:
            # 这里我们无法提供类似ResNet的shortcut，因此返回None
            return None, None, None, None, features, x_id, x_age
        if return_age:
            return embedding, x_id, x_age
        return embedding

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()





class AgeEstimationModule(nn.Module):
    def __init__(self, input_size, age_group, dist=False):
        super(AgeEstimationModule, self).__init__()
        out_neurons = 101
        self.age_output_layer = nn.Sequential(
            nn.BatchNorm2d(1024),  # 修改为1024以匹配DenseNet的输出
            nn.Flatten(),
            nn.Linear(1024 * (input_size // 32) ** 2, 512),  # 调整为1024，并考虑DenseNet的缩减因子
            nn.LeakyReLU(0.2, inplace=True) if dist else nn.ReLU(inplace=True),
            nn.Linear(512, out_neurons),
        )
        self.group_output_layer = nn.Linear(out_neurons, age_group)

    def forward(self, x_age):
        x_age = self.age_output_layer(x_age)
        x_group = self.group_output_layer(x_age)
        return x_age, x_group



from functools import partial

backbone_dict = {
    'ir34': partial(AIResNet, num_layers=[3, 4, 6, 3], mode="ir"),
    'ir50': partial(AIResNet, num_layers=[3, 4, 14, 3], mode="ir"),
    'ir64': partial(AIResNet, num_layers=[3, 4, 10, 3], mode="ir"),
    'ir101': partial(AIResNet, num_layers=[3, 13, 30, 3], mode="ir"),
    'irse101': partial(AIResNet, num_layers=[3, 13, 30, 3], mode="ir_se"),
    'densenet': partial(DenseNetAIFR, input_size=224)

}