import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, local_dim=512, global_dim=128, hidden_dim=256):
        super(CrossAttention, self).__init__()
        # Linear layers for projecting to Q, K, V
        self.query_proj = nn.Linear(local_dim, hidden_dim)
        self.key_proj = nn.Linear(global_dim, hidden_dim)
        self.value_proj = nn.Linear(global_dim, hidden_dim)
        self.attention_scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

        self.output_proj = nn.Linear(hidden_dim, local_dim)  # Project back to original dim

    def forward(self, x, g):
        """
        x: (B, N, 512) - Local features for each point
        g: (B, 1, 128) - Global features of the entire point cloud
        """
        B, N, _ = x.shape

        # Expand global features to match the number of points (N)
        g_expanded = g.repeat(1, N, 1)  # (B, N, 128)

        # Project to Query, Key, Value
        Q = self.query_proj(x)  # (B, N, hidden_dim)
        K = self.key_proj(g_expanded)  # (B, N, hidden_dim)
        V = self.value_proj(g_expanded)  # (B, N, hidden_dim)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.attention_scale  # (B, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N, N)

        # Apply attention weights
        attended_x = torch.matmul(attn_weights, V)  # (B, N, hidden_dim)

        return self.output_proj(attended_x)  # (B, N, 512)


class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.attention_scale = torch.sqrt(torch.tensor(attention_dim, dtype=torch.float32))

    def forward(self, x):
        # x.shape: (bs, N, input_dim)
        batch_size, N, _ = x.size()

        Q = self.query(x)  # (batch_size, N, attention_dim)
        K = self.key(x)  # (batch_size, N, attention_dim)
        V = self.value(x)  # (batch_size, N, attention_dim)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.attention_scale  # (batch_size, N, N)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attended_x = torch.matmul(attn_weights, V)  # (batch_size, N, attention_dim)

        return attended_x


class GlobalEnhancement(nn.Module):
    def __init__(self, input_dim=3, output_dim=128):
        super(GlobalEnhancement, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 16, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, output_dim, kernel_size=1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):  # [bs, 1, 2]
        bs, c = x.shape
        x = x.view(bs, 1, c)
        x = self.leaky_relu(self.conv1(x.transpose(1, 2)))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))

        return x


class StructureEnhancement(nn.Module):
    def __init__(self, input_dim=3, output_dim=256):
        super(StructureEnhancement, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, output_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        return x


class FeatureEnhancement(nn.Module):
    def __init__(self, input_dim=6, output_dim=256):
        super(FeatureEnhancement, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, output_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        return x


class PredictNet(nn.Module):
    def __init__(self, feature_channels=9, global_channels=3, num_classes=50,
                 global_dim=128, structure_dim=256, feature_dim=256,
                 dropout=0.4,
                 attention_dim=256):
        super(PredictNet, self).__init__()

        self.globalEnhancement = GlobalEnhancement(global_channels, global_dim)
        self.structureEnhancement = StructureEnhancement(3, structure_dim)
        self.featureEnhancement = FeatureEnhancement(feature_channels - 3, feature_dim)

        self.cross_attention = CrossAttention(feature_dim + structure_dim, global_dim, attention_dim)

        self.fc = nn.Sequential(
            nn.Linear(feature_dim + structure_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, g, x):

        xyz = x[:, :, :3]
        features = x[:, :, 3:]
        structure = self.structureEnhancement(xyz)      # (bs, 256, N)
        feature = self.featureEnhancement(features)     # (bs, 256, N)
        x = torch.cat([structure, feature], dim=1)  # (bs,256 + 256, N)

        g = self.globalEnhancement(g)  # (bs, 128, 1)

        attn_x = self.cross_attention(x.transpose(1, 2), g.transpose(1, 2)).transpose(1, 2)  # (bs, 512, N)

        # residual connection
        x = x + attn_x  # (bs, 512, N)

        # max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.squeeze(2)  # (bs, 512)

        return self.fc(x)


if __name__ == '__main__':
    net = PredictNet()
    print(net)
    local = torch.randn(16, 1024, 9)
    glob = torch.randn(16, 3)
    out = net(glob, local)
    print(out.shape)

# TODO
#  1 Q=global KV=local-9461
#  2 Q=local KV=global-9490
