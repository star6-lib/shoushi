import torch.nn as nn
import torch

# PINO loss
def plate_stress_loss(sigma_xx, sigma_yy, sigma_xy, x_coor, y_coor):
    sxx_x = torch.autograd.grad(outputs=sigma_xx, inputs=x_coor, grad_outputs=torch.ones_like(sigma_xx), create_graph=True)[0]
    sxy_y = torch.autograd.grad(outputs=sigma_xy, inputs=y_coor, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0]
    sxy_x = torch.autograd.grad(outputs=sigma_xy, inputs=x_coor, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0]
    syy_y = torch.autograd.grad(outputs=sigma_yy, inputs=y_coor, grad_outputs=torch.ones_like(sigma_yy), create_graph=True)[0]

    rx = sxx_x + sxy_y
    ry = sxy_x + syy_y
    return rx, ry

# 2. 新增：本构损失 (胡克定律对齐)
def constitutive_loss(u, v, sigma_xx, sigma_yy, sigma_xy, x_coor, y_coor, params):
    E, mu = params
    G = E / 2 / (1+mu)

    eps_xx = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    eps_yy = torch.autograd.grad(outputs=v, inputs=y_coor, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(outputs=v, inputs=x_coor, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    eps_xy = (u_y + v_x)

    sxx_true = (E / (1-mu**2)) * (eps_xx + mu*(eps_yy))
    syy_true = (E / (1-mu**2)) * (eps_yy + mu*(eps_xx))
    sxy_true = G * eps_xy

    return sigma_xx - sxx_true, sigma_yy - syy_true, sigma_xy - sxy_true

def bc_top_shear_loss(u, v, x_coor, y_coor, params):
    """
    专门针对上边界 (Y 方向受强制位移，X 方向自由滑动) 的边界条件。
    计算并返回剪应力 sigma_xy，以便在外部约束其为 0。
    """
    # extract parameters
    E, mu = params
    G = E / 2 / (1 + mu)

    # compute partial derivatives (只算计算剪应力需要的偏导)
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(outputs=v, inputs=x_coor, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    # compute shear stress
    sigma_xy = G * (u_y + v_x)

    return sigma_xy

def bc_edgeY_loss(sigma_yy, sigma_xy):
    return sigma_yy, sigma_xy

def bc_edgeX_loss(sigma_xx, sigma_xy):
    return sigma_xx, sigma_xy

def hole_free_loss(sigma_xx, sigma_yy, sigma_xy, x_coor, y_coor, geo_params):
    B, M = x_coor.shape
    geo_reshaped = geo_params.view(B, 4, 3)
    cx = geo_reshaped[:, :, 0].unsqueeze(1)
    cy = geo_reshaped[:, :, 1].unsqueeze(1)
    cr = geo_reshaped[:, :, 2].unsqueeze(1)

    x_expanded = x_coor.unsqueeze(-1)
    y_expanded = y_coor.unsqueeze(-1)

    dist = (x_expanded - cx) ** 2 + (y_expanded - cy) ** 2
    min_idx = torch.argmin(dist, dim=-1, keepdim=True)

    cx_closest = torch.gather(cx.expand(B, M, 4), -1, min_idx).squeeze(-1)
    cy_closest = torch.gather(cy.expand(B, M, 4), -1, min_idx).squeeze(-1)
    cr_closest = torch.gather(cr.expand(B, M, 4), -1, min_idx).squeeze(-1)

    nx = (x_coor - cx_closest) / cr_closest
    ny = (y_coor - cy_closest) / cr_closest

    Tx = sigma_xx * nx + sigma_xy * ny
    Ty = sigma_xy * nx + sigma_yy * ny
    return Tx, Ty
