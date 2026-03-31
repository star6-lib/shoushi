import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from .utils_losses import plate_stress_loss, bc_edgeX_loss, hole_free_loss, bc_edgeY_loss, constitutive_loss


# plotting function
def plot(xcoor, ycoor, f):
    # Create a scatter plot with color mapped to the 'f' values
    plt.scatter(xcoor, ycoor, c=f, cmap='viridis', marker='o', s=5)
    # Add a colorbar
    plt.colorbar(label='f')


# validation function
def val(model, loader, args, device, num_nodes_list):
    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes, max_hole_nodes = num_nodes_list

    # === 新增：初始化 3 个独立的相对误差累加器 ===
    mean_err_u = 0.0
    mean_err_v = 0.0
    mean_err_vm = 0.0
    num_eval = 0

    with torch.no_grad():
        for (coors, u, v, sxx, syy, sxy, flag, geo, in_disp, in_force, f_type, vm) in loader:

            # extract domain shape information
            if args.geo_node in ('vary_bound', 'vary_bound_sup'):
                ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes,
                                     max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
            if args.geo_node == 'all_bound':
                ss_index = np.arange(max_pde_nodes,
                                     max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
            if args.geo_node == 'all_domain':
                ss_index = np.arange(0, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
            shape_coor = coors[:, ss_index, :].float().to(device)
            shape_flag = flag[:, ss_index].float().to(device)

            all_coors = coors.float().to(device)
            all_flag = flag.float().to(device)
            in_disp = in_disp.float().to(device)
            in_force = in_force.float().to(device)

            u_gt = u.float().to(device)
            v_gt = v.float().to(device)
            vm_gt = vm.float().to(device)

            # forward pass
            u_pred, v_pred, sxx_pred, syy_pred, sxy_pred = model(all_coors[:, :, 0], all_coors[:, :, 1], in_disp,
                                                                 in_force, shape_coor, shape_flag)

            # 提取掩码内的有效预测值和真实值
            u_pred_valid = u_pred * all_flag
            v_pred_valid = v_pred * all_flag
            u_gt_valid = u_gt * all_flag
            v_gt_valid = v_gt * all_flag

            sxx_pred_valid = sxx_pred * all_flag
            syy_pred_valid = syy_pred * all_flag
            sxy_pred_valid = sxy_pred * all_flag
            vm_gt_valid = vm_gt * all_flag

            # 计算预测的 Von Mises 应力
            vm_pred_valid = torch.sqrt(
                sxx_pred_valid ** 2 + syy_pred_valid ** 2 - sxx_pred_valid * syy_pred_valid + 3 * (sxy_pred_valid ** 2))

            # 分别计算 U, V, VM 的相对 L2 误差
            err_u = torch.norm(u_pred_valid - u_gt_valid) / torch.norm(u_gt_valid)
            err_v = torch.norm(v_pred_valid - v_gt_valid) / torch.norm(v_gt_valid)
            err_vm = torch.norm(vm_pred_valid - vm_gt_valid) / torch.norm(vm_gt_valid)

            mean_err_u += err_u.item()
            mean_err_v += err_v.item()
            mean_err_vm += err_vm.item()
            num_eval += 1

        return mean_err_u / num_eval, mean_err_v / num_eval, mean_err_vm / num_eval


# testing function
def test(model, loader, args, device, num_nodes_list, params, dir):
    # transforme state to be eval
    model.eval()

    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes, max_hole_nodes = num_nodes_list

    mean_relative_L2 = 0
    num_eval = 0
    max_relative_err = -1
    min_relative_err = np.inf
    for (coors, u, v, sxx, syy, sxy, flag, geo, in_disp, in_force, f_type, vm) in loader:

        # extract domain shape information
        if args.geo_node in ('vary_bound', 'vary_bound_sup'):
            ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes,
                                 max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
        if args.geo_node == 'all_bound':
            ss_index = np.arange(max_pde_nodes,
                                 max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
        if args.geo_node == 'all_domain':
            ss_index = np.arange(0, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
        shape_coors = coors[:, ss_index, :].float().to(device)
        shape_flag = flag[:, ss_index]
        shape_flag = shape_flag.float().to(device)

        # prepare the data
        in_disp = in_disp.float().to(device)
        in_force = in_force.float().to(device)
        coors = coors.float().to(device)
        u = u.float().to(device)
        v = v.float().to(device)
        vm = vm.float().to(device)
        flag = flag.float().to(device)

        x_test = coors[:, :, 0]
        y_test = coors[:, :, 1]

        # model forward
        u_pred, v_pred, sxx_pred, syy_pred, sxy_pred = model(x_test, y_test, in_disp, in_force, shape_coors, shape_flag)

        if dir == 'x':
            L2_relative = torch.sqrt(torch.sum((u_pred * flag - u * flag) ** 2, -1)) / (
                        torch.sqrt(torch.sum((u * flag) ** 2, -1)) + 1e-8)
            pred = u_pred
            gt = u

        elif dir == 'y':
            L2_relative = torch.sqrt(torch.sum((v_pred * flag - v * flag) ** 2, -1)) / (
                        torch.sqrt(torch.sum((v * flag) ** 2, -1)) + 1e-8)
            pred = v_pred
            gt = v

        elif dir == 'vm':
            vm_pred = torch.sqrt(sxx_pred ** 2 - sxx_pred * syy_pred + syy_pred ** 2 + 3 * sxy_pred ** 2)
            pred = vm_pred
            gt = vm

            # Mises 专属的相对 L2 误差计算
            L2_relative = torch.sqrt(torch.sum((vm_pred * flag - vm * flag) ** 2, -1)) / (torch.sqrt(
                torch.sum((vm * flag) ** 2, -1)) + 1e-8)

        # find the max and min error sample in this batch
        max_err, max_err_idx = torch.topk(L2_relative, 1)
        if max_err > max_relative_err:
            max_relative_err = max_err
            worst_xcoor = coors[max_err_idx, :, 0].squeeze(0).squeeze(-1).detach().cpu().numpy()
            worst_ycoor = coors[max_err_idx, :, 1].squeeze(0).squeeze(-1).detach().cpu().numpy()
            worst_f = pred[max_err_idx, :].squeeze(0).detach().cpu().numpy()
            worst_gt = gt[max_err_idx, :].squeeze(0).detach().cpu().numpy()
            worst_ff = flag[max_err_idx, :].squeeze(0).detach().cpu().numpy()
            valid_id = np.where(worst_ff > 0.5)[0]
            worst_xcoor = worst_xcoor[valid_id]
            worst_ycoor = worst_ycoor[valid_id]
            worst_f = worst_f[valid_id]
            worst_gt = worst_gt[valid_id]
        min_err, min_err_idx = torch.topk(-L2_relative, 1)
        min_err = -min_err
        if min_err < min_relative_err:
            min_relative_err = min_err
            best_xcoor = coors[min_err_idx, :, 0].squeeze(0).squeeze(-1).detach().cpu().numpy()
            best_ycoor = coors[min_err_idx, :, 1].squeeze(0).squeeze(-1).detach().cpu().numpy()
            best_f = pred[min_err_idx, :].squeeze(0).detach().cpu().numpy()
            best_gt = gt[min_err_idx, :].squeeze(0).detach().cpu().numpy()
            best_ff = flag[min_err_idx, :].squeeze(0).detach().cpu().numpy()
            valid_id = np.where(best_ff >= 0.5)[0]
            best_xcoor = best_xcoor[valid_id]
            best_ycoor = best_ycoor[valid_id]
            best_f = best_f[valid_id]
            best_gt = best_gt[valid_id]

        mean_relative_L2 += torch.sum(L2_relative).detach().cpu().item()
        num_eval += in_disp.shape[0]

    mean_relative_L2 /= num_eval
    mean_relative_L2 = mean_relative_L2

    # 每行独立计算预测值/真实值的颜色范围
    worst_max_color = np.amax(worst_gt)
    worst_min_color = np.amin(worst_gt)
    best_max_color = np.amax(best_gt)
    best_min_color = np.amin(best_gt)
    # 每行独立计算绝对误差的颜色范围
    worst_err_max = np.amax(np.abs(worst_f - worst_gt))
    best_err_max = np.amax(np.abs(best_f - best_gt))

    # make the plot
    cm = plt.cm.get_cmap('RdYlBu')
    plt.figure(figsize=(15, 8))

    # --- 第一行：Worst Case ---
    plt.subplot(2, 3, 1)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_f, cmap=cm, vmin=worst_min_color, vmax=worst_max_color, marker='o',
                s=3)
    plt.colorbar()
    plt.title('prediction')

    plt.subplot(2, 3, 2)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_gt, cmap=cm, vmin=worst_min_color, vmax=worst_max_color, marker='o',
                s=3)
    plt.title('ground truth')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.scatter(worst_xcoor, worst_ycoor, c=np.abs(worst_f - worst_gt), cmap=cm, vmin=0, vmax=worst_err_max, marker='o',
                s=3)
    plt.title('absolute error')
    plt.colorbar()

    # --- 第二行：Best Case ---
    plt.subplot(2, 3, 4)
    plt.scatter(best_xcoor, best_ycoor, c=best_f, cmap=cm, vmin=best_min_color, vmax=best_max_color, marker='o', s=3)
    plt.colorbar()
    plt.title('prediction')

    plt.subplot(2, 3, 5)
    plt.scatter(best_xcoor, best_ycoor, c=best_gt, cmap=cm, vmin=best_min_color, vmax=best_max_color, marker='o', s=3)
    plt.title('ground truth')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.scatter(best_xcoor, best_ycoor, c=np.abs(best_f - best_gt), cmap=cm, vmin=0, vmax=best_err_max, marker='o', s=3)
    plt.title('absolute error')
    plt.colorbar()

    plt.savefig(r'./res/plots/sample_{}_{}_{}_{}.png'.format(args.geo_node, args.model, args.data, dir))

    return mean_relative_L2


# function of extracting the geometry embeddings
def get_geometry_embeddings(model, loader, args, device, num_nodes_list):
    # transforme state to be eval
    model.eval()

    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes, max_hole_nodes = num_nodes_list

    # forward to get the embeddings
    all_geo_embeddings = []
    for (coors, u, v, sxx, syy, sxy, flag, geo, in_disp, in_force, f_type, vm) in loader:

        # extract domain shape information
        if args.geo_node in ('vary_bound', 'vary_bound_sup'):
            ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes,
                                 max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
        if args.geo_node == 'all_bound':
            ss_index = np.arange(max_pde_nodes,
                                 max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
        if args.geo_node == 'all_domain':
            ss_index = np.arange(0, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
        shape_coors = coors[:, ss_index, :].float().to(device)  # (B, max_hole, 2)
        shape_flag = flag[:, ss_index]
        shape_flag = shape_flag.float().to(device)  # (B, max_hole)

        # prepare the data
        coors = coors.float().to(device)
        in_disp = in_disp.float().to(device)
        in_force = in_force.float().to(device)

        # model forward
        Geo_embeddings = model.predict_geometry_embedding(coors[:, :, 0], coors[:, :, 1],
                                                          in_disp, in_force, shape_coors, shape_flag)
        all_geo_embeddings.append(Geo_embeddings)

    all_geo_embeddings = torch.cat(tuple(all_geo_embeddings), 0)

    return all_geo_embeddings

# ==========================================================
# 新增：定义统一的学习率调度器获取函数
# ==========================================================
def get_scheduler(optimizer, config):
    """根据 yaml 配置动态生成学习率衰减机制"""
    decay_type = config['train'].get('lr_decay_type', 'none')

    if decay_type == 'step':
        # 阶梯衰减：每隔 step_size 个 epoch，学习率乘以 gamma
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['train'].get('lr_decay_step', 50),
            gamma=config['train'].get('lr_decay_rate', 0.5)
        )
    elif decay_type == 'exp':
        # 指数衰减：每个 epoch 学习率乘以 gamma
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config['train'].get('lr_decay_gamma', 0.95)
        )
    elif decay_type == 'cosine':
        # 余弦退火：平滑下降到最小学习率
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train']['epochs'],
            eta_min=float(config['train'].get('lr_min', 1e-6))
        )
    elif decay_type == 'plateau':
        # 自适应衰减：当验证集误差停止下降 patience 个次后，自动降低学习率 (极其推荐！)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['train'].get('lr_decay_rate', 0.5),
            patience=config['train'].get('lr_patience', 10),
            min_lr=float(config['train'].get('lr_min', 1e-6))
        )
    else:
        # 设为 'none' 或未定义时，使用固定学习率
        return None

# define the training function
def train(args, config, model, device, loaders, num_nodes_list, params):
    # print training configuration
    print('================================')
    print('training configuration')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])

    # === 新增：将 Loss 权重清晰打印到日志中 ===
    print('--------------------------------')
    print('Loss Weights Configuration:')
    print('weight_load (Disp BC):', config['train']['weight_load'])
    print('weight_fix (Fixed BC):      ', config['train']['weight_fix'])
    print('weight_pde (Equilibrium):   ', config['train']['weight_pde'])
    print('weight_free (Hole/Free BC): ', config['train']['weight_free'])
    print('================================')

    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes, max_hole_nodes = num_nodes_list

    # define model training configuration
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # define optimizer and loss
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['base_lr'])
    # === 新增：初始化学习率调度器 ===
    scheduler = get_scheduler(optimizer, config)

    # visual frequency
    vf = config['train']['visual_freq']

    # err history
    err_hist = []

    # move the model to the defined device
    try:
        model.load_state_dict(
            torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model),
                       weights_only=True))
    except:
        print('No trained models')
    model = model.to(device)

    # define tradeoff weights
    weight_load = config['train']['weight_load']
    weight_pde = config['train']['weight_pde']
    weight_fix = config['train']['weight_fix']
    weight_free = config['train']['weight_free']
    weight_const = config['train']['weight_const']

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        avg_pde_loss = np.inf
        avg_fix_loss = np.inf
        avg_free_loss = np.inf
        avg_load_loss = np.inf
        avg_const_loss = np.inf

        for e in pbar:

            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err = val(model, val_loader, args, device, num_nodes_list)
                err_hist.append(err)
                print('Current epoch error:', err)
                print('current epochs pde loss:', avg_pde_loss)
                print('fix bc loss:', avg_fix_loss)
                print('free bc loss:', avg_free_loss)
                print('load bc loss:', avg_load_loss)
                print('constitutive loss:', avg_const_loss)

                avg_pde_loss = 0
                avg_fix_loss = 0
                avg_free_loss = 0
                avg_load_loss = 0
                avg_const_loss = 0
                if err < min_val_err:
                    torch.save(model.state_dict(),
                               r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data,
                                                                                    args.model))
                    min_val_err = err

            # train one epoch
            model.train()
            for (coors, u, v, sxx, syy, sxy, flag, geo, in_disp, in_force, f_type, vm) in train_loader:

                in_disp = in_disp.float().to(device)
                in_force = in_force.float().to(device)
                coors = coors.float().to(device)
                flag = flag.float().to(device)
                geo = geo.float().to(device)

                for _ in range(config['train']['coor_sampling_freq']):

                    # 修改为 (全流程留在 GPU 内显存极速采样)：
                    ss_index = torch.randint(0, max_pde_nodes, (config['train']['coor_sampling_size'],), device=device)
                    pde_sampled_coors = coors[:, ss_index, :]
                    pde_flag = flag[:, ss_index]

                    ss_index = np.arange(max_pde_nodes + max_par_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes)
                    bcy_coors = coors[:, ss_index, :]
                    bcy_flag = flag[:, ss_index]

                    ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes,
                                         max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
                    bcxy_coors = coors[:, ss_index, :]
                    bcxy_flag = flag[:, ss_index]

                    ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes,
                                         max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                    hole_coors = coors[:, ss_index, :]
                    hole_flag = flag[:, ss_index]

                    if args.geo_node in ('vary_bound', 'vary_bound_sup'):
                        ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes,
                                             max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                    if args.geo_node == 'all_bound':
                        ss_index = np.arange(max_pde_nodes,
                                             max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                    if args.geo_node == 'all_domain':
                        ss_index = np.arange(0,
                                             max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)

                    shape_coor = coors[:, ss_index, :]
                    shape_flag = flag[:, ss_index]

                    # === 核心修改 3：动态物理边界掩码 (Adaptive Masking) ===
                    f_type = f_type.float().to(device)
                    # 假设 Type 1,2 为受力，Type 3,4 为受位移 (维度: B, 1)
                    mask_force = ((f_type == 1) | (f_type == 2)).float().unsqueeze(-1)
                    mask_disp = ((f_type == 3) | (f_type == 4)).float().unsqueeze(-1)

                    # =======================================================
                    # 1. 顶边 (Top Edge) - 直接生成坐标，无须 Variable
                    # =======================================================
                    x_top = torch.linspace(-10, 10, 101).unsqueeze(0).repeat(in_disp.shape[0], 1).to(device)
                    y_top = torch.ones_like(x_top) * 10.0
                    u_top_pred, v_top_pred, _, syy_top, sxy_top = model(x_top, y_top, in_disp, in_force, shape_coor,
                                                                        shape_flag)

                    sigma_yy_top, sigma_xy_top = bc_edgeY_loss(syy_top, sxy_top)

                    # --- 分别计算两种模式的 Loss 并用掩码相加 ---
                    # 模式 A: 受力边界 (逼近应力 in_force，剪应力为 0)
                    loss_load_force = mse(sigma_yy_top * mask_force, in_force * mask_force) + mse(
                        sigma_xy_top * mask_force, torch.zeros_like(sigma_xy_top))
                    # 模式 B: 受位移边界 (逼近位移 in_disp，剪应力为 0)
                    loss_load_disp = mse(v_top_pred * mask_disp, in_disp * mask_disp) + mse(sigma_xy_top * mask_disp,
                                                                                            torch.zeros_like(
                                                                                                sigma_xy_top))

                    load_loss = loss_load_force + loss_load_disp

                    # =======================================================
                    # 2. 侧边 BCxy (位移固定) - 原本就没用 Variable，保持不变
                    # =======================================================
                    u_BCxy_pred, v_BCxy_pred, _, _, _ = model(bcxy_coors[:, :, 0], bcxy_coors[:, :, 1], in_disp,
                                                              in_force, shape_coor, shape_flag)

                    # =======================================================
                    # 3. 内部 PDE (唯一需要保留 requires_grad=True 的地方！！！)
                    # =======================================================
                    x_pde = Variable(pde_sampled_coors[:, :, 0], requires_grad=True)
                    y_pde = Variable(pde_sampled_coors[:, :, 1], requires_grad=True)
                    u_pde_pred, v_pde_pred, sxx_pde, syy_pde, sxy_pde = model(x_pde, y_pde, in_disp, in_force,
                                                                              shape_coor, shape_flag)

                    rx, ry = plate_stress_loss(sxx_pde, syy_pde, sxy_pde, x_pde, y_pde)  # <-- 注意这里传参
                    diff_sxx, diff_syy, diff_sxy = constitutive_loss(u_pde_pred, v_pde_pred, sxx_pde, syy_pde, sxy_pde,
                                                                     x_pde, y_pde, params)

                    # =======================================================
                    # 4. 侧边 BCy (自由边) - 删掉 Variable，直接传切片
                    # =======================================================
                    _, _, sxx_bcy, syy_bcy, sxy_bcy = model(bcy_coors[:, :, 0], bcy_coors[:, :, 1], in_disp, in_force,
                                                            shape_coor, shape_flag)
                    sigma_xx, sigma_xy = bc_edgeX_loss(sxx_bcy, sxy_bcy)

                    # =======================================================
                    # 5. 孔洞 Hole (自由边界) - 删掉 Variable，直接传切片
                    # =======================================================
                    _, _, sxx_hole, syy_hole, sxy_hole = model(hole_coors[:, :, 0], hole_coors[:, :, 1], in_disp,
                                                               in_force, shape_coor, shape_flag)
                    Tx_hole, Ty_hole = hole_free_loss(sxx_hole, syy_hole, sxy_hole, hole_coors[:, :, 0],
                                                      hole_coors[:, :, 1], geo)

                    # compute the losses
                    pde_loss = torch.mean((rx * pde_flag) ** 2) + torch.mean((ry * pde_flag) ** 2)
                    const_loss = torch.mean((diff_sxx * pde_flag) ** 2) + torch.mean(
                        (diff_syy * pde_flag) ** 2) + torch.mean((diff_sxy * pde_flag) ** 2)
                    fix_loss = torch.mean((u_BCxy_pred * bcxy_flag) ** 2) + torch.mean((v_BCxy_pred * bcxy_flag) ** 2)
                    free_bc = torch.mean((sigma_xx * bcy_flag) ** 2) + torch.mean((sigma_xy * bcy_flag) ** 2)
                    free_hole_loss = torch.mean((Tx_hole * hole_flag) ** 2) + torch.mean((Ty_hole * hole_flag) ** 2)
                    free_loss = free_bc + free_hole_loss

                    total_loss = weight_pde * pde_loss + weight_const * const_loss + weight_load * load_loss + weight_fix * fix_loss + weight_free * free_loss

                    avg_pde_loss += pde_loss.detach().cpu().item()
                    avg_fix_loss += fix_loss.detach().cpu().item()
                    avg_free_loss += free_loss.detach().cpu().item()
                    avg_load_loss += load_loss.detach().cpu().item()
                    avg_const_loss += const_loss.detach().cpu().item()

                    # update parameter
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # clear cuda
                    # torch.cuda.empty_cache()
            # === 新增：其它类型调度器在每个 Epoch 结束时自动步进衰减 ===
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

    # final test
    model.load_state_dict(
        torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model),
                   weights_only=True))
    model.eval()
    err = test(model, test_loader, args, device, num_nodes_list, params, dir='x')
    _ = test(model, test_loader, args, device, num_nodes_list, params, dir='y')
    _ = test(model, test_loader, args, device, num_nodes_list, params, dir='vm')
    print('Best L2 relative error on test loader:', err)


# ==========================================================
# define the supervised training function (Pure Data-driven)
# ==========================================================

def sup_train(args, config, model, device, loaders, num_nodes_list, params):
    # print training configuration
    print('================================')
    print('Supervised (Data-Driven) Training Configuration')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])
    print('================================')

    # # === 新增：将 Loss 权重清晰打印到日志中 ===
    # print('--------------------------------')
    # print('Loss Weights Configuration:')
    # print('weight_load (Disp BC):', config['train']['weight_load'])
    # print('weight_fix (Fixed BC):      ', config['train']['weight_fix'])
    # print('weight_pde (Equilibrium):   ', config['train']['weight_pde'])
    # print('weight_free (Hole/Free BC): ', config['train']['weight_free'])
    # print('================================')

    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes, max_hole_nodes = num_nodes_list

    # define model training configuration
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # define optimizer and loss
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['base_lr'])
    # === 新增：初始化学习率调度器 ===
    scheduler = get_scheduler(optimizer, config)

    # visual frequency
    vf = config['train']['visual_freq']

    # err history
    err_hist = []
    train_loss_hist = []

    # move the model to the defined device
    try:
        model.load_state_dict(
            torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model),
                       weights_only=True))
        print('>>> 成功加载已有的最佳模型！')
    except:
        print('No trained models, starting from scratch')
    model = model.to(device)

    # start the training
    if args.phase in ['train', 'sup_train']:
        # 初始化最优误差为无穷大
        min_val_err = np.inf

        # === 新增：打印 Epoch 0 初始模型状态，不保存模型 ===
        model.eval()
        init_u, init_v, init_vm = val(model, val_loader, args, device, num_nodes_list)
        print(f"\n[Epoch 0 / Init] Validation L2 Error | U: {init_u:.6f} | V: {init_v:.6f} | VM: {init_vm:.6f}")
        print("  └─ [保护机制] 初始模型不参与最优模型保存评估 (min_val_err 保持为 inf)\n")

        avg_mse_loss = 0.0
        avg_loss_u = 0.0
        avg_loss_v = 0.0
        avg_loss_sxx = 0.0
        avg_loss_syy = 0.0
        avg_loss_sxy = 0.0

        # === 初始化 NTK 自适应权重和滑动平均系数 ===
        lambda_u, lambda_v, lambda_sxx, lambda_syy, lambda_sxy = 1.0, 1.0, 1.0, 1.0, 1.0
        alpha = 0.9  # 滑动平均衰减率 (EMA)，防止权重震荡
        update_freq = 50  # 每隔 50 个 batch 计算一次梯度权重。避免每次计算严重拖慢训练速度

        for e in pbar:
            # ==============================
            # 第一阶段：训练 (Training)
            # ==============================
            model.train()

            for batch_idx, (coors, u, v, sxx, syy, sxy, flag, geo, in_disp, in_force, f_type, vm) in enumerate(
                    train_loader):

                # 对于纯数据驱动，不需要采样，直接使用全部节点拟合
                all_coors = coors.float().to(device)
                all_flag = flag.float().to(device)  # (B, M)
                u_gt = u.float().to(device)
                v_gt = v.float().to(device)
                sxx_gt = sxx.float().to(device)
                syy_gt = syy.float().to(device)
                sxy_gt = sxy.float().to(device)
                in_disp = in_disp.float().to(device)
                in_force = in_force.float().to(device)

                # extract the boundary of the varying shape
                if args.geo_node in ('vary_bound', 'vary_bound_sup'):
                    ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes,
                                         max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                if args.geo_node == 'all_bound':
                    ss_index = np.arange(max_pde_nodes,
                                         max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                if args.geo_node == 'all_domain':
                    ss_index = np.arange(0,
                                         max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)

                shape_coor = coors[:, ss_index, :].float().to(device)
                shape_flag = flag[:, ss_index].float().to(device)

                # === 核心：极速前向传播 (无 Variable 梯度追踪) ===
                u_pred, v_pred, sxx_pred, syy_pred, sxy_pred = model(all_coors[:, :, 0], all_coors[:, :, 1], in_disp,
                                                                     in_force, shape_coor, shape_flag)

                # === 5 个物理量全方位 MSE 暴力逼近！ ===
                loss_u = mse(u_pred * all_flag, u_gt * all_flag)
                loss_v = mse(v_pred * all_flag, v_gt * all_flag)
                loss_sxx = mse(sxx_pred * all_flag, sxx_gt * all_flag)
                loss_syy = mse(syy_pred * all_flag, syy_gt * all_flag)
                loss_sxy = mse(sxy_pred * all_flag, sxy_gt * all_flag)

                # =============================================================
                # === NTK / 梯度范数 引导的动态权重计算核心逻辑 ===
                # =============================================================
                if batch_idx % update_freq == 0:
                    # 选取网络的底层共享部分（几何特征提取器 DG）作为梯度对齐的基准锚点
                    shared_params = list(model.DG.parameters()) + \
                                    list(model.branch_disp.parameters()) + \
                                    list(model.branch_force.parameters())

                    # 定义辅助函数：执行单项 Loss 反向传播并获取梯度范数
                    def compute_grad_norm(loss_component):
                        optimizer.zero_grad()
                        # retain_graph=True 确保后续还能继续对其他 loss 反向传播
                        loss_component.backward(retain_graph=True)
                        grads = [p.grad.flatten() for p in shared_params if p.grad is not None]
                        if len(grads) == 0:
                            return 1.0
                        return torch.norm(torch.cat(grads)).item()

                    # 分别提取 5 项 Loss 对底层的梯度大小
                    norm_u = compute_grad_norm(loss_u)
                    norm_v = compute_grad_norm(loss_v)
                    norm_sxx = compute_grad_norm(loss_sxx)
                    norm_syy = compute_grad_norm(loss_syy)
                    norm_sxy = compute_grad_norm(loss_sxy)

                    # 以各项梯度范数的平均值作为目标对齐基准
                    mean_norm = (norm_u + norm_v + norm_sxx + norm_syy + norm_sxy) / 5.0

                    # 计算目标权重 (梯度的倒数)，加 1e-8 防止除零错
                    hat_lambda_u = mean_norm / (norm_u + 1e-8)
                    hat_lambda_v = mean_norm / (norm_v + 1e-8)
                    hat_lambda_sxx = mean_norm / (norm_sxx + 1e-8)
                    hat_lambda_syy = mean_norm / (norm_syy + 1e-8)
                    hat_lambda_sxy = mean_norm / (norm_sxy + 1e-8)

                    # 使用指数滑动平均 (EMA) 平滑更新当前权重
                    lambda_u = alpha * lambda_u + (1 - alpha) * hat_lambda_u
                    lambda_v = alpha * lambda_v + (1 - alpha) * hat_lambda_v
                    lambda_sxx = alpha * lambda_sxx + (1 - alpha) * hat_lambda_sxx
                    lambda_syy = alpha * lambda_syy + (1 - alpha) * hat_lambda_syy
                    lambda_sxy = alpha * lambda_sxy + (1 - alpha) * hat_lambda_sxy

                # =============================================================
                # === 应用动态计算出的自适应权重 ===
                # =============================================================
                total_loss = (lambda_u * loss_u +
                              lambda_v * loss_v +
                              lambda_sxx * loss_sxx +
                              lambda_syy * loss_syy +
                              lambda_sxy * loss_sxy)

                # === 单独存储各项损失 ===
                avg_mse_loss += total_loss.detach().cpu().item()
                avg_loss_u += loss_u.detach().cpu().item()
                avg_loss_v += loss_v.detach().cpu().item()
                avg_loss_sxx += loss_sxx.detach().cpu().item()
                avg_loss_syy += loss_syy.detach().cpu().item()
                avg_loss_sxy += loss_sxy.detach().cpu().item()

                # update parameter
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # ==============================
            # 第二阶段：验证与日志打印 (Validation & Logging)
            # ==============================
            if (e + 1) % vf == 0:
                model.eval()
                val_err_u, val_err_v, val_err_vm = val(model, val_loader, args, device, num_nodes_list)

                if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_err_vm)

                # === 新增 2：获取当前最新的学习率 ===
                current_lr = optimizer.param_groups[0]['lr']

                err_hist.append(val_err_vm)

                # 记录训练误差
                div_factor = len(train_loader) * vf
                train_loss_hist.append(avg_mse_loss / div_factor)

                # 打印分离的验证误差
                print(f'\nEpoch {e + 1} - LR: {current_lr:.2e} | Validation L2 Error | U: {val_err_u:.6f} | V: {val_err_v:.6f} | VM: {val_err_vm:.6f}')
                print(f'Total MSE Loss: {(avg_mse_loss / div_factor):.6f}')
                # 打印各损失项的同时，展示当前的 NTK 权重大小
                print(f'  ├─ Loss U:   {(avg_loss_u / div_factor):.6f}  |  Weight(λ_u):   {lambda_u:.4f}')
                print(f'  ├─ Loss V:   {(avg_loss_v / div_factor):.6f}  |  Weight(λ_v):   {lambda_v:.4f}')
                print(f'  ├─ Loss Sxx: {(avg_loss_sxx / div_factor):.6f}  |  Weight(λ_sxx): {lambda_sxx:.4f}')
                print(f'  ├─ Loss Syy: {(avg_loss_syy / div_factor):.6f}  |  Weight(λ_syy): {lambda_syy:.4f}')
                print(f'  └─ Loss Sxy: {(avg_loss_sxy / div_factor):.6f}  |  Weight(λ_sxy): {lambda_sxy:.4f}')

                # 仅使用 VM 应力误差作为判定模型好坏的标准
                if val_err_vm < min_val_err:
                    torch.save(model.state_dict(),
                               r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data,
                                                                                    args.model))
                    min_val_err = val_err_vm
                    print(f"  [Model Saved] 新的最佳模型已保存! 当前 VM 误差: {val_err_vm:.4f}")

                # 清零累加器，准备下一个周期的统计
                avg_mse_loss = 0.0
                avg_loss_u = 0.0
                avg_loss_v = 0.0
                avg_loss_sxx = 0.0
                avg_loss_syy = 0.0
                avg_loss_sxy = 0.0
            # === 新增：其它类型调度器在每个 Epoch 结束时自动步进衰减 ===
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

    # final test (包含 vm 绘图)
    model.load_state_dict(
        torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model),
                   weights_only=True))
    model.eval()
    err_u = test(model, test_loader, args, device, num_nodes_list, params, dir='x')
    err_v = test(model, test_loader, args, device, num_nodes_list, params, dir='y')
    err_vm = test(model, test_loader, args, device, num_nodes_list, params, dir='vm')
    print("\n================ 终极测试结果 (基于最佳 VM 模型) ================")
    print(f" 🌟 Disp U (X方向位移) Relative L2 Error: {err_u:.4f}")
    print(f" 🌟 Disp V (Y方向位移) Relative L2 Error: {err_v:.4f}")
    print(f" 🔥 Von Mises Stress   Relative L2 Error: {err_vm:.4f}")
    print("=================================================================\n")


# ==========================================================
# define the Physics-Informed + Data-Driven training function (双驱动)
# ==========================================================

def plus_train(args, config, model, device, loaders, num_nodes_list, params):
    # print training configuration
    print('================================')
    print('Dual-Driven (Physics + Data) Training Configuration')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])
    print('================================')

    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes, max_hole_nodes = num_nodes_list

    # define model training configuration
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # define optimizer and loss
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['base_lr'])
    # === 新增：初始化学习率调度器 ===
    scheduler = get_scheduler(optimizer, config)

    # visual frequency
    vf = config['train']['visual_freq']

    # err history
    err_hist = []

    # move the model to the defined device
    try:
        model.load_state_dict(
            torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model),
                       weights_only=True))
        print('>>> 成功加载已有的最佳模型！(基于之前的预训练)')
    except:
        print('No trained models, starting from scratch')
    model = model.to(device)

    # start the training
    if args.phase in ['train', 'sup_train', 'plus_train']:
        min_val_err = np.inf

        model.eval()
        init_u, init_v, init_vm = val(model, val_loader, args, device, num_nodes_list)
        print(f"\n[Epoch 0 / Init] Validation L2 Error | U: {init_u:.6f} | V: {init_v:.6f} | VM: {init_vm:.6f}\n")

        # === 初始化 10 个 NTK 自适应权重 (5个数据 + 5个物理) ===
        lam_u, lam_v, lam_sxx, lam_syy, lam_sxy = 1.0, 1.0, 1.0, 1.0, 1.0
        lam_pde, lam_const, lam_load, lam_fix, lam_free = 1.0, 1.0, 1.0, 1.0, 1.0

        alpha = 0.9  # 滑动平均衰减率 (EMA)
        update_freq = 50  # 每 50 个 batch 更新一次权重

        # 累加器初始化
        avg_loss = {
            'data_u': 0.0, 'data_v': 0.0, 'data_sxx': 0.0, 'data_syy': 0.0, 'data_sxy': 0.0,
            'phys_pde': 0.0, 'phys_const': 0.0, 'phys_load': 0.0, 'phys_fix': 0.0, 'phys_free': 0.0,
            'total': 0.0
        }

        for e in pbar:
            model.train()

            for batch_idx, (coors, u, v, sxx, syy, sxy, flag, geo, in_disp, in_force, f_type, vm) in enumerate(
                    train_loader):

                # =======================================================
                # 准备全局数据 (Data-driven 需求)
                # =======================================================
                all_coors = coors.float().to(device)
                all_flag = flag.float().to(device)
                in_disp = in_disp.float().to(device)
                in_force = in_force.float().to(device)
                u_gt, v_gt = u.float().to(device), v.float().to(device)
                sxx_gt, syy_gt, sxy_gt = sxx.float().to(device), syy.float().to(device), sxy.float().to(device)
                f_type = f_type.float().to(device)
                geo = geo.float().to(device)

                # 解析几何编码掩码
                if args.geo_node in ('vary_bound', 'vary_bound_sup'):
                    ss_index_geo = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes,
                                             max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                if args.geo_node == 'all_bound':
                    ss_index_geo = np.arange(max_pde_nodes,
                                             max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                if args.geo_node == 'all_domain':
                    ss_index_geo = np.arange(0,
                                             max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)

                shape_coor = coors[:, ss_index_geo, :].float().to(device)
                shape_flag = flag[:, ss_index_geo].float().to(device)

                # =======================================================
                # 模块 1: 数据驱动损失 (Data-Driven Losses) - 全场评估
                # =======================================================
                u_pred, v_pred, sxx_pred, syy_pred, sxy_pred = model(all_coors[:, :, 0], all_coors[:, :, 1], in_disp,
                                                                     in_force, shape_coor, shape_flag)

                loss_data_u = mse(u_pred * all_flag, u_gt * all_flag)
                loss_data_v = mse(v_pred * all_flag, v_gt * all_flag)
                loss_data_sxx = mse(sxx_pred * all_flag, sxx_gt * all_flag)
                loss_data_syy = mse(syy_pred * all_flag, syy_gt * all_flag)
                loss_data_sxy = mse(sxy_pred * all_flag, sxy_gt * all_flag)

                # =======================================================
                # 模块 2: 物理驱动损失 (Physics-Informed Losses) - 局部采样评估
                # =======================================================

                # PDE 采样与变量追踪
                ss_idx_pde = torch.randint(0, max_pde_nodes, (config['train']['coor_sampling_size'],), device=device)
                pde_sampled_coors = all_coors[:, ss_idx_pde, :]
                pde_flag = all_flag[:, ss_idx_pde]

                x_pde = Variable(pde_sampled_coors[:, :, 0], requires_grad=True)
                y_pde = Variable(pde_sampled_coors[:, :, 1], requires_grad=True)
                u_pde_pred, v_pde_pred, sxx_pde, syy_pde, sxy_pde = model(x_pde, y_pde, in_disp, in_force, shape_coor,
                                                                          shape_flag)

                rx, ry = plate_stress_loss(sxx_pde, syy_pde, sxy_pde, x_pde, y_pde)
                diff_sxx, diff_syy, diff_sxy = constitutive_loss(u_pde_pred, v_pde_pred, sxx_pde, syy_pde, sxy_pde,
                                                                 x_pde, y_pde, params)

                loss_phys_pde = torch.mean((rx * pde_flag) ** 2) + torch.mean((ry * pde_flag) ** 2)
                loss_phys_const = torch.mean((diff_sxx * pde_flag) ** 2) + torch.mean(
                    (diff_syy * pde_flag) ** 2) + torch.mean((diff_sxy * pde_flag) ** 2)

                # Top Edge (载荷边界)
                mask_force = ((f_type == 1) | (f_type == 2)).float().unsqueeze(-1)
                mask_disp = ((f_type == 3) | (f_type == 4)).float().unsqueeze(-1)
                x_top = torch.linspace(-10, 10, 101).unsqueeze(0).repeat(in_disp.shape[0], 1).to(device)
                y_top = torch.ones_like(x_top) * 10.0
                u_top_pred, v_top_pred, _, syy_top, sxy_top = model(x_top, y_top, in_disp, in_force, shape_coor,
                                                                    shape_flag)
                sigma_yy_top, sigma_xy_top = bc_edgeY_loss(syy_top, sxy_top)

                loss_load_force = mse(sigma_yy_top * mask_force, in_force * mask_force) + mse(sigma_xy_top * mask_force,
                                                                                              torch.zeros_like(
                                                                                                  sigma_xy_top))
                loss_load_disp = mse(v_top_pred * mask_disp, in_disp * mask_disp) + mse(sigma_xy_top * mask_disp,
                                                                                        torch.zeros_like(sigma_xy_top))
                loss_phys_load = loss_load_force + loss_load_disp

                # 侧边 BCxy (固定边界)
                idx_bcxy = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes,
                                     max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
                u_BCxy_pred, v_BCxy_pred, _, _, _ = model(all_coors[:, idx_bcxy, 0], all_coors[:, idx_bcxy, 1], in_disp,
                                                          in_force, shape_coor, shape_flag)
                loss_phys_fix = torch.mean((u_BCxy_pred * all_flag[:, idx_bcxy]) ** 2) + torch.mean(
                    (v_BCxy_pred * all_flag[:, idx_bcxy]) ** 2)

                # 侧边 BCy 与孔洞 Hole (自由边界)
                idx_bcy = np.arange(max_pde_nodes + max_par_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes)
                _, _, sxx_bcy, syy_bcy, sxy_bcy = model(all_coors[:, idx_bcy, 0], all_coors[:, idx_bcy, 1], in_disp,
                                                        in_force, shape_coor, shape_flag)
                sigma_xx, sigma_xy = bc_edgeX_loss(sxx_bcy, sxy_bcy)
                free_bc = torch.mean((sigma_xx * all_flag[:, idx_bcy]) ** 2) + torch.mean(
                    (sigma_xy * all_flag[:, idx_bcy]) ** 2)

                idx_hole = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes,
                                     max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                _, _, sxx_hole, syy_hole, sxy_hole = model(all_coors[:, idx_hole, 0], all_coors[:, idx_hole, 1],
                                                           in_disp, in_force, shape_coor, shape_flag)
                Tx_hole, Ty_hole = hole_free_loss(sxx_hole, syy_hole, sxy_hole, all_coors[:, idx_hole, 0],
                                                  all_coors[:, idx_hole, 1], geo)
                free_hole = torch.mean((Tx_hole * all_flag[:, idx_hole]) ** 2) + torch.mean(
                    (Ty_hole * all_flag[:, idx_hole]) ** 2)
                loss_phys_free = free_bc + free_hole

                # =============================================================
                # 模块 3: NTK 自适应权重更新 (10 项 Loss 平衡)
                # =============================================================
                if batch_idx % update_freq == 0:
                    shared_params = list(model.DG.parameters()) + list(model.branch_disp.parameters()) + list(
                        model.branch_force.parameters())

                    def compute_grad_norm(loss_component):
                        optimizer.zero_grad()
                        loss_component.backward(retain_graph=True)
                        grads = [p.grad.flatten() for p in shared_params if p.grad is not None]
                        if len(grads) == 0: return 1.0
                        return torch.norm(torch.cat(grads)).item()

                    # 提取数据梯度范数
                    n_du, n_dv = compute_grad_norm(loss_data_u), compute_grad_norm(loss_data_v)
                    n_dsx, n_dsy, n_dsxy = compute_grad_norm(loss_data_sxx), compute_grad_norm(
                        loss_data_syy), compute_grad_norm(loss_data_sxy)

                    # 提取物理梯度范数
                    n_pde = compute_grad_norm(loss_phys_pde)
                    n_cst = compute_grad_norm(loss_phys_const)
                    n_ld = compute_grad_norm(loss_phys_load)
                    n_fx = compute_grad_norm(loss_phys_fix)
                    n_fr = compute_grad_norm(loss_phys_free)

                    # 计算 10 项平均值
                    mean_norm = (n_du + n_dv + n_dsx + n_dsy + n_dsxy + n_pde + n_cst + n_ld + n_fx + n_fr) / 10.0

                    # EMA 更新 (利用倒数平衡)
                    lam_u = alpha * lam_u + (1 - alpha) * (mean_norm / (n_du + 1e-8))
                    lam_v = alpha * lam_v + (1 - alpha) * (mean_norm / (n_dv + 1e-8))
                    lam_sxx = alpha * lam_sxx + (1 - alpha) * (mean_norm / (n_dsx + 1e-8))
                    lam_syy = alpha * lam_syy + (1 - alpha) * (mean_norm / (n_dsy + 1e-8))
                    lam_sxy = alpha * lam_sxy + (1 - alpha) * (mean_norm / (n_dsxy + 1e-8))

                    lam_pde = alpha * lam_pde + (1 - alpha) * (mean_norm / (n_pde + 1e-8))
                    lam_const = alpha * lam_const + (1 - alpha) * (mean_norm / (n_cst + 1e-8))
                    lam_load = alpha * lam_load + (1 - alpha) * (mean_norm / (n_ld + 1e-8))
                    lam_fix = alpha * lam_fix + (1 - alpha) * (mean_norm / (n_fx + 1e-8))
                    lam_free = alpha * lam_free + (1 - alpha) * (mean_norm / (n_fr + 1e-8))

                # =============================================================
                # 模块 4: Total Loss & Backprop
                # =============================================================
                total_loss = (lam_u * loss_data_u + lam_v * loss_data_v +
                              lam_sxx * loss_data_sxx + lam_syy * loss_data_syy + lam_sxy * loss_data_sxy +
                              lam_pde * loss_phys_pde + lam_const * loss_phys_const +
                              lam_load * loss_phys_load + lam_fix * loss_phys_fix + lam_free * loss_phys_free)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # 记录 Log 数据
                avg_loss['data_u'] += loss_data_u.item();
                avg_loss['data_v'] += loss_data_v.item()
                avg_loss['data_sxx'] += loss_data_sxx.item();
                avg_loss['data_syy'] += loss_data_syy.item();
                avg_loss['data_sxy'] += loss_data_sxy.item()
                avg_loss['phys_pde'] += loss_phys_pde.item();
                avg_loss['phys_const'] += loss_phys_const.item()
                avg_loss['phys_load'] += loss_phys_load.item();
                avg_loss['phys_fix'] += loss_phys_fix.item();
                avg_loss['phys_free'] += loss_phys_free.item()
                avg_loss['total'] += total_loss.item()

            # ==============================
            # 模块 5: 验证与日志打印
            # ==============================
            if (e + 1) % vf == 0:
                model.eval()
                val_err_u, val_err_v, val_err_vm = val(model, val_loader, args, device, num_nodes_list)
                if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_err_vm)

                # === 新增 2：获取当前最新的学习率 ===
                current_lr = optimizer.param_groups[0]['lr']

                err_hist.append(val_err_vm)
                div_factor = len(train_loader) * vf

                print(f'\nEpoch {e + 1} - LR: {current_lr:.2e} | Validation L2 Error | U: {val_err_u:.6f} | V: {val_err_v:.6f} | VM: {val_err_vm:.6f}')
                print(f'Total Dual-Driven Loss: {(avg_loss["total"] / div_factor):.6f}')
                print('--- Data-Driven Losses & NTK Weights ---')
                print(f'  ├─ Data U:   {(avg_loss["data_u"] / div_factor):.6f} | λ: {lam_u:.4f}')
                print(f'  ├─ Data V:   {(avg_loss["data_v"] / div_factor):.6f} | λ: {lam_v:.4f}')
                print(f'  ├─ Data Sxx: {(avg_loss["data_sxx"] / div_factor):.6f} | λ: {lam_sxx:.4f}')
                print(f'  ├─ Data Syy: {(avg_loss["data_syy"] / div_factor):.6f} | λ: {lam_syy:.4f}')
                print(f'  └─ Data Sxy: {(avg_loss["data_sxy"] / div_factor):.6f} | λ: {lam_sxy:.4f}')
                print('--- Physics-Informed Losses & NTK Weights ---')
                print(f'  ├─ Phys PDE: {(avg_loss["phys_pde"] / div_factor):.6f} | λ: {lam_pde:.4f}')
                print(f'  ├─ Phys Cst: {(avg_loss["phys_const"] / div_factor):.6f} | λ: {lam_const:.4f}')
                print(f'  ├─ Phys Lod: {(avg_loss["phys_load"] / div_factor):.6f} | λ: {lam_load:.4f}')
                print(f'  ├─ Phys Fix: {(avg_loss["phys_fix"] / div_factor):.6f} | λ: {lam_fix:.4f}')
                print(f'  └─ Phys Fre: {(avg_loss["phys_free"] / div_factor):.6f} | λ: {lam_free:.4f}')

                if val_err_vm < min_val_err:
                    torch.save(model.state_dict(),
                               r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data,
                                                                                    args.model))
                    min_val_err = val_err_vm
                    print(f"  [Model Saved] 新的最佳模型已保存! 当前 VM 误差: {val_err_vm:.4f}")

                # 清零累加器
                for k in avg_loss: avg_loss[k] = 0.0
            # === 新增：其它类型调度器在每个 Epoch 结束时自动步进衰减 ===
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

    # final test
    model.load_state_dict(
        torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model),
                   weights_only=True))
    model.eval()
    err_u = test(model, test_loader, args, device, num_nodes_list, params, dir='x')
    err_v = test(model, test_loader, args, device, num_nodes_list, params, dir='y')
    err_vm = test(model, test_loader, args, device, num_nodes_list, params, dir='vm')
    print("\n================ 终极双驱动测试结果 ================")
    print(f" 🌟 Disp U Relative L2 Error: {err_u:.4f}")
    print(f" 🌟 Disp V Relative L2 Error: {err_v:.4f}")
    print(f" 🔥 VM Stress Relative L2 Error: {err_vm:.4f}")
    print("====================================================\n")