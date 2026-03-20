import torch.nn as nn
import torch

# PINO loss
# 1. 静力平衡损失 (PDE) - 一阶导数
def plate_stress_loss(sigma_xx, sigma_yy, sigma_xy, x_coor, y_coor):
    # 对预测的应力求一阶偏导计算散度
    sxx_x = \
    torch.autograd.grad(outputs=sigma_xx, inputs=x_coor, grad_outputs=torch.ones_like(sigma_xx), create_graph=True)[0]
    sxy_y = \
    torch.autograd.grad(outputs=sigma_xy, inputs=y_coor, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0]

    sxy_x = \
    torch.autograd.grad(outputs=sigma_xy, inputs=x_coor, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0]
    syy_y = \
    torch.autograd.grad(outputs=sigma_yy, inputs=y_coor, grad_outputs=torch.ones_like(sigma_yy), create_graph=True)[0]

    rx = sxx_x + sxy_y
    ry = sxy_x + syy_y
    return rx, ry


# 2. 本构方程损失 (胡克定律) - 一阶导数
def constitutive_loss(u, v, sigma_xx, sigma_yy, sigma_xy, x_coor, y_coor, params):
    E, mu = params
    G = E / 2 / (1 + mu)

    # 对预测的位移求一阶偏导计算应变
    eps_xx = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    eps_yy = torch.autograd.grad(outputs=v, inputs=y_coor, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(outputs=v, inputs=x_coor, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    eps_xy = (u_y + v_x)

    # 根据应变计算真实的物理应力
    phys_sxx = (E / (1 - mu ** 2)) * (eps_xx + mu * eps_yy)
    phys_syy = (E / (1 - mu ** 2)) * (eps_yy + mu * eps_xx)
    phys_sxy = G * eps_xy

    # 返回预测应力与真实物理应力的误差项
    diff_sxx = sigma_xx - phys_sxx
    diff_syy = sigma_yy - phys_syy
    diff_sxy = sigma_xy - phys_sxy

    return diff_sxx, diff_syy, diff_sxy


# 3. 孔洞无牵引力边界 - 零阶导数 (直接代数运算)
def hole_free_loss(sigma_xx, sigma_yy, sigma_xy, x_coor, y_coor, geo_params):
    B, M = x_coor.shape
    # geo_params shape: (B, 12). Reshape to (B, 4, 3) representing 4 holes' (cx, cy, cr)
    geo_reshaped = geo_params.view(B, 4, 3)
    cx = geo_reshaped[:, :, 0].unsqueeze(1)  # (B, 1, 4)
    cy = geo_reshaped[:, :, 1].unsqueeze(1)  # (B, 1, 4)
    cr = geo_reshaped[:, :, 2].unsqueeze(1)  # (B, 1, 4)

    x_expanded = x_coor.unsqueeze(-1)  # (B, M, 1)
    y_expanded = y_coor.unsqueeze(-1)  # (B, M, 1)

    # 计算当前点到 4 个孔洞圆心的距离的平方: (B, M, 4)
    dist = (x_expanded - cx) ** 2 + (y_expanded - cy) ** 2     # 建立网格与孔洞的对应关系
                                                               # dist形状是(B,M,4)表示每个网格点到4个圆孔圆心的距离平方
    # 找到距离最近的孔洞索引: (B, M, 1)
    min_idx = torch.argmin(dist, dim=-1, keepdim=True)         # 找到每个网格点距离最近的孔洞索引
                                                               # dim=-1:在最后一个维度上找最小值
                                                               # keepdim=True:保持维度不变，argim:返回最小值对应的索引
                                                               

    # 提取最近孔洞的圆心和半径: (B, M)
    cx_closest = torch.gather(cx.expand(B, M, 4), -1, min_idx).squeeze(-1)
    cy_closest = torch.gather(cy.expand(B, M, 4), -1, min_idx).squeeze(-1)
    cr_closest = torch.gather(cr.expand(B, M, 4), -1, min_idx).squeeze(-1)

    # 计算外法线向量 (nx, ny)
    nx = (x_coor - cx_closest) / cr_closest
    ny = (y_coor - cy_closest) / cr_closest

    # 计算边界牵引力 (Traction)
    Tx = sigma_xx * nx + sigma_xy * ny
    Ty = sigma_xy * nx + sigma_yy * ny

    return Tx, Ty


def bc_edgeY_loss(u, v, x_coor, y_coor, params):
    """Free boundary condition for top/bottom edges (perpendicular to Y-axis): sigma_yy=0, sigma_xy=0."""

    # extract parameters
    E, mu = params
    G = E / 2 / (1 + mu)

    # compute strain
    eps_xx = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    eps_yy = torch.autograd.grad(outputs=v, inputs=y_coor, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(outputs=v, inputs=x_coor, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    eps_xy = (u_y + v_x)

    # compute stress
    sigma_yy = (E / (1 - mu ** 2)) * (eps_yy + mu * (eps_xx))
    sigma_xy = G * eps_xy

    return sigma_yy, sigma_xy


def bc_edgeX_loss(u, v, x_coor, y_coor, params):
    """Free boundary condition for left/right edges (perpendicular to X-axis): sigma_xx=0, sigma_xy=0."""

    # extract parameters
    E, mu = params
    G = E / 2 / (1 + mu)

    # compute strain
    eps_xx = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    eps_yy = torch.autograd.grad(outputs=v, inputs=y_coor, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(outputs=v, inputs=x_coor, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    eps_xy = (u_y + v_x)

    # compute stress
    sigma_xx = (E / (1 - mu ** 2)) * (eps_xx + mu * (eps_yy))
    sigma_xy = G * eps_xy

    return sigma_xx, sigma_xy

