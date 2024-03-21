import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, n_input_pts, num_class=-1,normal_channel=True, latent_dim=256):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(256, num_class)

        # decoder
        self.decoder_fc1 = nn.Linear(latent_dim, 512)
        self.decoder_bn1 = nn.BatchNorm1d(512)
        self.decoder_drop1 = nn.Dropout(0.3)
        self.decoder_fc2 = nn.Linear(512, 1024)
        self.decoder_bn2 = nn.BatchNorm1d(1024)
        self.decoder_drop2 = nn.Dropout(0.3)
        self.decoder_out = nn.Linear(1024, 3*n_input_pts)
        self.n_pts = n_input_pts

    def forward(self, xyz):
        """
        Reconstructs input PC
        [B, 3, N_pts] --> [B, N_pts, 3]
        also returns latent AE feat [B, 256]
        """
        B, _, n_pts = xyz.shape
        assert n_pts == self.n_pts

        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # pointnet encoder
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x)))) # [..., 256]
        latent_feats = x
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)

        # decoder to recon
        x = self.decoder_drop1(F.relu(self.decoder_bn1(self.decoder_fc1(x))))
        x = self.decoder_drop2(F.relu(self.decoder_bn2(self.decoder_fc2(x))))
        x = self.decoder_out(x).reshape(B, n_pts, 3)
        return x, latent_feats



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
