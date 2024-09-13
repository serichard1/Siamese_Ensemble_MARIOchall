from .base_models import mobilenet_v3_small
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualVision(nn.Module):
    def __init__(
        self,
        backbone,
        in_size,
        nclasses=4,
        drop_ratio_head=0.4
    ):
        super().__init__()

        self.backbone = backbone
        self.drop_ratio_head = drop_ratio_head

	self.backbone = backbone
        self.dropout_head = dropout_head
        self.merge_bscans = self._create_sequential([in_size*2, 1024, 256, 32])
        self.merge_numeric = nn.Sequential(nn.Linear(32+3, 32), nn.SiLU(inplace=True))
        self.head =nn.Linear(32, nclasses)
        
    def _create_sequential(self, sizes):
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=self.dropout_head))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, bscan_ti, bscan_tj, bscan_num, age_ti, delta_t):
        bscan_ti, bscan_tj = map(lambda f: self.backbone(f), (bscan_ti, bscan_tj))
        bscan_num, age_ti, delta_t = map(lambda f: f.unsqueeze(1), (bscan_num, age_ti, delta_t))

        bscans_embed = self.merge_bscans(torch.cat((bscan_ti, bscan_tj), dim=1))
        final_embed = self.merge_numeric(torch.cat((bscans_embed, bscan_num, age_ti, delta_t), dim=1))
        logits = self.head(final_embed)

        return logits
    
class CrossSightv5(nn.Module):
    """ Ensemble model """
    def __init__(
        self,
	model1,
        model2,
        model3,
        backbone,
        nclasses=3+1,
        dropout_head=0.3
    ):
        super().__init__()

	self.backbone = backbone
        self.model1, self.model2, self.model3 = model1, model2, model3,
        self.dropout_head = dropout_head

        self.merging_embed = self._create_sequential([256*4, 1024, 256, 32])
        self.merging_head = nn.Sequential(nn.Linear(32+3, 16), nn.Tanh(inplace=True), nn.Linear(16, nclasses))
        
    def _create_sequential(self, sizes):
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, bscan_ti, bscan_tj, bscan_num, age_ti, delta_t, optimizer):

        embed1 = self.model1(bscan_ti, bscan_tj)
        embed2 = self.model2(bscan_ti, bscan_tj)
        embed3 = self.model3(bscan_ti, bscan_tj)
        embed_opt = self.backbone(optimizer)

        bscan_num, age_ti, delta_t = map(lambda f: f.unsqueeze(1), (bscan_num, age_ti, delta_t))

        images_embed = self.merging_embed(torch.cat((embed1, embed2, embed3, embed_opt), dim=1))
        logits = self.merging_head(torch.cat((images_embed, bscan_num, age_ti, delta_t), dim=1))

        return logits
    
class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values = self.scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o

