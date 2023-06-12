import torch.nn as nn
import torch.nn.init as init
import torch

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(*self.shape)

class EMGModel(nn.Module):
    def __init__(self, num_channels):
        super(EMGModel, self).__init__()

        self.emg_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
                nn.MaxPool1d(2),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
                nn.MaxPool1d(5),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
                nn.MaxPool1d(5),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
                nn.MaxPool1d(2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                # nn.MaxPool1d(2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                Reshape(-1, 64 * 1),
                nn.Linear(64 * 1, 3)
            ) for _ in range(num_channels)
        ])

    def forward(self, x):
        emg_outputs = [emg_layer(x).unsqueeze(dim=1) for emg_layer in self.emg_layers]
        return emg_outputs


class FMGModel(nn.Module):
    def __init__(self, num_channels):
        super(FMGModel, self).__init__()

        self.fmg_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
                nn.MaxPool1d(2),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
                nn.MaxPool1d(5),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
                nn.MaxPool1d(5),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
                nn.MaxPool1d(2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                # nn.MaxPool1d(2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                Reshape(-1, 64 * 1),
                nn.Linear(64 * 1, 3)
            ) for _ in range(num_channels)
        ])

    def forward(self, x):
        fmg_outputs = [fmg_layer(x).unsqueeze(dim=1) for fmg_layer in self.fmg_layers]
        return fmg_outputs

# class FusionModel(nn.Module):
#     def __init__(self, num_channels):
#         super(FusionModel, self).__init__()

#         self.fusion_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
#                 nn.MaxPool1d(2),
#                 nn.BatchNorm1d(16),
#                 nn.ReLU(),
#                 nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
#                 nn.MaxPool1d(5),
#                 nn.BatchNorm1d(64),
#                 nn.ReLU(),
#                 nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
#                 nn.MaxPool1d(5),
#                 nn.BatchNorm1d(256),
#                 nn.ReLU(),
#                 nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
#                 nn.MaxPool1d(2),
#                 nn.BatchNorm1d(128),
#                 nn.ReLU(),
#                 nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
#                 nn.MaxPool1d(2),
#                 nn.BatchNorm1d(64),
#                 nn.ReLU(),
#                 Reshape(-1, 64 * 1),
#                 nn.Linear(64 * 1, 13)
#             ) for _ in range(num_channels)
#         ])

#     def forward(self, hyb):
#         fusion_outputs = [fusion_layer(hyb).unsqueeze(dim=1) for fusion_layer in self.fusion_layers]
#         return fusion_outputs

class FusionModel(nn.Module):
    def __init__(self, num_channels):
        super(FusionModel, self).__init__()

        self.emg_pre_layers = nn.ModuleList([ # 先分别用CNN分别提取特征
            nn.Sequential(
                nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
                nn.MaxPool1d(2),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                nn.MaxPool1d(5),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding=1),
                nn.MaxPool1d(2),
                nn.BatchNorm1d(128),
                nn.ReLU()
            ) for _ in range(num_channels)
        ])
        self.fmg_pre_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
                nn.MaxPool1d(2),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                nn.MaxPool1d(5),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding=1),
                nn.MaxPool1d(2),
                nn.BatchNorm1d(128),
                nn.ReLU()
            ) for _ in range(num_channels)
        ])
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
                nn.MaxPool1d(5),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
                # nn.MaxPool1d(2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                Reshape(-1, 32 * 1),
                nn.Linear(32 * 1, 3),
                Reshape(-1, 1, 3)
            ) for _ in range(num_channels)
        ])

    def forward(self, x_emg, x_fmg):
        emg_pres = [emg_pre_layer(x_emg) for emg_pre_layer in self.emg_pre_layers]
        fmg_pres = [fmg_pre_layer(x_fmg) for fmg_pre_layer in self.fmg_pre_layers]
        fusion_outputs = [fusion_layer(torch.cat((emg_pre, fmg_pre), dim=1)) for fusion_layer, emg_pre, fmg_pre in zip(self.fusion_layers, emg_pres, fmg_pres)]
        return fusion_outputs
