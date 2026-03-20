import torch
import torch.nn as nn
import math


''' ------------------------- PI-GANO -------------------------- '''


class MMLP_Trunk(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.act = nn.Tanh()

        # 记忆锚点投影层 (将包含 Domain_enc 的初始输入投影为 U 和 V)
        self.U_proj = nn.Linear(in_dim, hidden_dim)
        self.V_proj = nn.Linear(in_dim, hidden_dim)

        # 隐藏层：维度链核心
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))  # 第 1 层
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))  # 第 2~4 层

    def forward(self, x):
        # 计算记忆锚点
        U = self.act(self.U_proj(x))
        V = self.act(self.V_proj(x))

        # 逐层前向传播：强行注入记忆 (U, V)
        H = self.act(self.layers[0](x))
        H = H * U + (1 - H) * V  # 注入坐标与几何的高频记忆

        for i in range(1, self.num_layers):
            Z = self.act(self.layers[i](H))
            H = Z * U + (1 - Z) * V  # 每层都强制注入几何记忆！

        return H

class DG(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network
        trunk_layers = [nn.Linear(2, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)
        
    def forward(self, shape_coor, shape_flag):
        '''
        shape_coor: (B, M'', 2)
        shape_flag: (B, M'')

        return u: (B, 1, F)
        '''

        # === 新增：将孔洞坐标归一化到 [-1, 1] ===
        shape_coor_norm = shape_coor / 10.0

        # get the first kernel
        enc = self.branch(shape_coor_norm)    # (B, M, F)
        enc_masked = enc.masked_fill(shape_flag.unsqueeze(-1) == 0, -1e9)
        # 4. 最大池化 (Max Pooling) - 捕捉孔洞的极限锐利特征
        Domain_enc, _ = torch.max(enc_masked, dim=1, keepdim=True)  # (B, 1, F)

        return Domain_enc

class GANO(nn.Module):

    def __init__(self, config):
        super().__init__()

        # define the geometry encoder
        self.DG = DG(config)

        # === 新增: 力学边界 Branch Net (输入维度 101) ===
        branch_layers_f = [nn.Linear(101, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            branch_layers_f.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            branch_layers_f.append(nn.Tanh())
        branch_layers_f.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch_force = nn.Sequential(*branch_layers_f)

        # === 新增: 位移边界 Branch Net (输入维度 101) ===
        branch_layers_d = [nn.Linear(101, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            branch_layers_d.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            branch_layers_d.append(nn.Tanh())
        branch_layers_d.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch_disp = nn.Sequential(*branch_layers_d)

        # parlifting layer
        self.xy_lift1 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift2 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift3 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift4 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift5 = nn.Linear(2, config['model']['fc_dim'])

        # === 替换为 MMLP 核心 ===
        in_dim = config['model']['fc_dim'] * 2  # 512
        hidden_dim = config['model']['fc_dim']  # 256
        num_layers = config['model']['N_layer'] + 1  # 对应 4 层

        self.Trunk_u = MMLP_Trunk(in_dim, hidden_dim, num_layers)
        self.Trunk_v = MMLP_Trunk(in_dim, hidden_dim, num_layers)
        self.Trunk_sxx = MMLP_Trunk(in_dim, hidden_dim, num_layers)
        self.Trunk_syy = MMLP_Trunk(in_dim, hidden_dim, num_layers)
        self.Trunk_sxy = MMLP_Trunk(in_dim, hidden_dim, num_layers)
    
    def predict_geometry_embedding(self, x_coor, y_coor, shape_coor, shape_flag):

        Domain_enc = self.DG(shape_coor, shape_flag)    # (B,1,F)

        return Domain_enc
  
    def forward(self, x_coor, y_coor, input_force, input_disp, shape_coor, shape_flag):
        '''
        input_force: (B, 101)
        input_disp: (B, 101)
        shape_coor: (B, M'', 2)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.DG(shape_coor, shape_flag)    # (B,1,F)

        # === 新增：将主干网络的物理坐标归一化到 [-1, 1] ===
        x_norm = x_coor / 10.0
        y_norm = y_coor / 10.0

        # concat coors
        xy = torch.cat((x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)   # (B,M,F)
        xy_local_v = self.xy_lift2(xy)   # (B,M,F)
        xy_local_sxx = self.xy_lift3(xy)
        xy_local_syy = self.xy_lift4(xy)
        xy_local_sxy = self.xy_lift5(xy)

        # combine with global embedding
        xy_global_u = torch.cat((xy_local_u, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)
        xy_global_v = torch.cat((xy_local_v, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)
        xy_global_sxx = torch.cat((xy_local_sxx, Domain_enc.repeat(1, mD, 1)), -1)
        xy_global_syy = torch.cat((xy_local_syy, Domain_enc.repeat(1, mD, 1)), -1)
        xy_global_sxy = torch.cat((xy_local_sxy, Domain_enc.repeat(1, mD, 1)), -1)

        # === 核心改动: 多分支编码并相加融合 ===
        enc_f = self.branch_force(input_force)  # (B, 2F)
        enc_d = self.branch_disp(input_disp)  # (B, 2F)
        enc = enc_f + enc_d  # (B, 2F)
        enc = enc.unsqueeze(1)  # (B, 1, 2F) 以支持广播

        # === 通过 MMLP 进行高频几何感知预测 ===
        u_trunk = self.Trunk_u(xy_global_u)
        u = torch.mean(u_trunk * enc, -1)  # (B, M) <-- 融合放在这里！原汁原味的 DeepONet 算子点积！

        v_trunk = self.Trunk_v(xy_global_v)
        v = torch.mean(v_trunk * enc, -1)

        sxx_trunk = self.Trunk_sxx(xy_global_sxx)
        sxx = torch.mean(sxx_trunk * enc, -1)

        syy_trunk = self.Trunk_syy(xy_global_syy)
        syy = torch.mean(syy_trunk * enc, -1)

        sxy_trunk = self.Trunk_sxy(xy_global_sxy)
        sxy = torch.mean(sxy_trunk * enc, -1)

        return u, v, sxx, syy, sxy


