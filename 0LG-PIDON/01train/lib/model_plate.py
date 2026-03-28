import torch
import torch.nn as nn
import math

''' ------------------------- PI-GANO -------------------------- '''


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

        # get the first kernel
        enc = self.branch(shape_coor)  # (B, M, F)
        enc_masked = enc * shape_flag.unsqueeze(-1)  # (B, M, F)
        Domain_enc = torch.sum(enc_masked, 1, keepdim=True) / torch.sum(shape_flag.unsqueeze(-1), 1,
                                                                        keepdim=True)  # (B, 1, F)

        return Domain_enc


class GANO(nn.Module):

    def __init__(self, config):
        super().__init__()

        # define the geometry encoder
        self.DG = DG(config)

        # === 核心改动 1：拆分出独立的力学与位移 Branch Net ===
        # 位移分支
        trunk_layers_d = [nn.Linear(101, 2 * config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers_d.append(nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim']))
            trunk_layers_d.append(nn.Tanh())
        trunk_layers_d.append(nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim']))
        self.branch_disp = nn.Sequential(*trunk_layers_d)

        # 力学分支
        trunk_layers_f = [nn.Linear(101, 2 * config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers_f.append(nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim']))
            trunk_layers_f.append(nn.Tanh())
        trunk_layers_f.append(nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim']))
        self.branch_force = nn.Sequential(*trunk_layers_f)

        # parlifting layer
        self.xy_lift1 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift2 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift3 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift4 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift5 = nn.Linear(2, config['model']['fc_dim'])

        # trunk network 1
        self.FC1u = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2u = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3u = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4u = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.act = nn.Tanh()

        # trunk network 2
        self.FC1v = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2v = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3v = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4v = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])

        # === 新增：trunk network 3 (sxx) ===
        self.FC1sxx = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2sxx = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3sxx = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4sxx = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])

        # === 新增：trunk network 4 (syy) ===
        self.FC1syy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2syy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3syy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4syy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])

        # === 新增：trunk network 5 (sxy) ===
        self.FC1sxy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2sxy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3sxy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4sxy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])

    def predict_geometry_embedding(self, x_coor, y_coor, in_disp, in_force, shape_coor, shape_flag):

        Domain_enc = self.DG(shape_coor, shape_flag)  # (B,1,F)

        return Domain_enc

    def forward(self, x_coor, y_coor, in_disp, in_force, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)
        shape_coor: (B, M'', 2)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.DG(shape_coor, shape_flag)  # (B,1,F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)  # (B,M,F)
        xy_local_v = self.xy_lift2(xy)  # (B,M,F)
        xy_local_sxx = self.xy_lift3(xy)
        xy_local_syy = self.xy_lift4(xy)
        xy_local_sxy = self.xy_lift5(xy)

        # combine with global embedding
        xy_global_u = torch.cat((xy_local_u, Domain_enc.repeat(1, mD, 1)), -1)  # (B,M,2F)
        xy_global_v = torch.cat((xy_local_v, Domain_enc.repeat(1, mD, 1)), -1)  # (B,M,2F)
        xy_global_sxx = torch.cat((xy_local_sxx, Domain_enc.repeat(1, mD, 1)), -1)
        xy_global_syy = torch.cat((xy_local_syy, Domain_enc.repeat(1, mD, 1)), -1)
        xy_global_sxy = torch.cat((xy_local_sxy, Domain_enc.repeat(1, mD, 1)), -1)

        # === 核心改动 2：双分支独立提取并加和融合 ===
        enc_d = self.branch_disp(in_disp)  # (B, 2F)
        enc_f = self.branch_force(in_force)  # (B, 2F)
        enc = enc_d * enc_f
        enc = enc.unsqueeze(1)  # (B, 1, 2F)

        # predict u
        u = self.FC1u(xy_global_u)  # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)  # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)  # (B,M,F)
        u = self.act(u)
        # u = u * enc
        u = self.FC4u(u)  # (B,M,F)
        # u = self.act(u)
        u = torch.mean(u * enc, -1)  # (B, M)

        # predict v
        v = self.FC1v(xy_global_v)  # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC2v(v)  # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC3v(v)  # (B,M,F)
        v = self.act(v)
        # v = v * enc
        v = self.FC4v(v)  # (B,M,F)
        # v = self.act(v)
        v = torch.mean(v * enc, -1)  # (B, M)

        # predict sxx
        sxx = self.FC1sxx(xy_global_sxx)
        sxx = self.act(sxx) * enc
        sxx = self.FC2sxx(sxx)
        sxx = self.act(sxx) * enc
        sxx = self.FC3sxx(sxx)
        sxx = self.act(sxx)
        sxx = self.FC4sxx(sxx)
        sxx = torch.mean(sxx * enc, -1)  # (B, M)

        # predict syy
        syy = self.FC1syy(xy_global_syy)
        syy = self.act(syy) * enc
        syy = self.FC2syy(syy)
        syy = self.act(syy) * enc
        syy = self.FC3syy(syy)
        syy = self.act(syy)
        syy = self.FC4syy(syy)
        syy = torch.mean(syy * enc, -1)  # (B, M)

        # predictsxy
        sxy = self.FC1sxy(xy_global_sxy)
        sxy = self.act(sxy) * enc
        sxy = self.FC2sxy(sxy)
        sxy = self.act(sxy) * enc
        sxy = self.FC3sxy(sxy)
        sxy = self.act(sxy)
        sxy = self.FC4sxy(sxy)
        sxy = torch.mean(sxy * enc, -1)  # (B, M)

        return u, v, sxx, syy, sxy


