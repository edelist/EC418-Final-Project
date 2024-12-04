# import torch
# import torch.nn.functional as F


# def spatial_argmax(logit):
#     """
#     Compute the soft-argmax of a heatmap
#     :param logit: A tensor of size BS x H x W
#     :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
#     """
#     weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
#     return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
#                         (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


# class Planner(torch.nn.Module):
#     def __init__(self):

#       super().__init__()

#       layers = []
#       layers.append(torch.nn.Conv2d(3,16,5,2,2))
#       layers.append(torch.nn.ReLU())
#       layers.append(torch.nn.Conv2d(16,1,1,1))

#       self._conv = torch.nn.Sequential(*layers)



#     def forward(self, img):
#         """
#         Your code here
#         Predict the aim point in image coordinate, given the supertuxkart image
#         @img: (B,3,96,128)
#         return (B,2)
#         """
#         x = self._conv(img)
#         #print(img.shape)
#         #print(x.shape)
#         return spatial_argmax(x[:, 0])
#         # return self.classifier(x.mean(dim=[-2, -1]))


# def save_model(model):
#     from torch import save
#     from os import path
#     if isinstance(model, Planner):
#         return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
#     raise ValueError("model type '%s' not supported!" % str(type(model)))


# def load_model():
#     from torch import load
#     from os import path
#     r = Planner()
#     r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
#     return r


# if __name__ == '__main__':
#     from controller import control
#     from utils import PyTux
#     from argparse import ArgumentParser


#     def test_planner(args):
#         # Load model
#         planner = load_model().eval()
#         pytux = PyTux()
#         for t in args.track:
#             steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
#             print(steps, how_far)
#         pytux.close()


#     parser = ArgumentParser("Test the planner")
#     parser.add_argument('track', nargs='+')
#     parser.add_argument('-v', '--verbose', action='store_true')
#     args = parser.parse_args()
#     test_planner(args)

import torch
import torch.nn as nn
import torch.nn.functional as F

# CBAM Block
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply channel attention
        ca = self.channel_attention(x)
        x = x * ca
        # Apply spatial attention
        sa = self.spatial_attention(torch.cat(
            (x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]), dim=1))
        x = x * sa
        return x


# Spatial Argmax Function
def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap.
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2, the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


# Planner Model with CBAM
class PlannerCBAM(nn.Module):
    def __init__(self, channels=[16, 32, 64]):
        super(PlannerCBAM, self).__init__()
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAMBlock(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        self.cbam2 = CBAMBlock(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)
        self.cbam3 = CBAMBlock(channels[2])
        self.final_conv = nn.Conv2d(channels[2], 1, kernel_size=1)

    def forward(self, img):
        x = self.conv1(img)
        x = self.cbam1(x)
        x = self.conv2(x)
        x = self.cbam2(x)
        x = self.conv3(x)
        x = self.cbam3(x)
        x = self.final_conv(x)
        return spatial_argmax(x[:, 0])


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, PlannerCBAM):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner_cbam.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    model = PlannerCBAM()
    model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner_cbam.th'), map_location='cpu'))
    return model


if __name__ == '__main__':
    from controller import control
    from utils import PyTux
    from argparse import ArgumentParser

    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for track in args.track:
            steps, how_far = pytux.rollout(track, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser("Test the planner with CBAM")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
