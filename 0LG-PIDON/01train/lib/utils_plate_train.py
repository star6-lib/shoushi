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

    mean_relative_L2 = 0
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
            shape_coors = coors[:, ss_index, :].float().to(device)  # (B, max_hole, 2)
            shape_flag = flag[:, ss_index]
            shape_flag = shape_flag.float().to(device)  # (B, max_hole)

            # prepare the data
            in_disp = in_disp.float().to(device)
            in_force = in_force.float().to(device)
            coors = coors.float().to(device)
            u = u.float().to(device)
            v = v.float().to(device)
            vm = vm.float().to(device)
            flag = flag.float().to(device)

            # model forward
            u_pred, v_pred, sxx_pred, syy_pred, sxy_pred = model(coors[:, :, 0], coors[:, :, 1], in_disp, in_force,
                                                                 shape_coors, shape_flag)

            # 计算位移的相对误差
            err_uv = torch.sqrt(
                torch.sum((u_pred * flag - u * flag) ** 2 + (v_pred * flag - v * flag) ** 2, -1)) / torch.sqrt(
                torch.sum((u * flag) ** 2 + (v * flag) ** 2, -1) + 1e-8)

            # 计算预测 Mises 和 真实 Mises 的相对误差
            vm_pred = torch.sqrt(sxx_pred ** 2 - sxx_pred * syy_pred + syy_pred ** 2 + 3 * sxy_pred ** 2)
            err_vm = torch.sqrt(torch.sum((vm_pred * flag - vm * flag) ** 2, -1)) / torch.sqrt(
                torch.sum((vm * flag) ** 2, -1) + 1e-8)

            # 综合位移和应力的误差作为最终评估指标
            L2_relative = err_uv + err_vm
            mean_relative_L2 += torch.sum(L2_relative).detach().cpu().item()
            num_eval += in_disp.shape[0]

        mean_relative_L2 /= num_eval
        mean_relative_L2 = mean_relative_L2

    return mean_relative_L2


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

        if dir in ['x', 'y']:
            L2_relative = torch.sqrt(
                torch.sum((u_pred * flag - u * flag) ** 2 + (v_pred * flag - v * flag) ** 2, -1)) / torch.sqrt(
                torch.sum((u * flag) ** 2 + (v * flag) ** 2, -1))
            if dir == 'x':
                pred = u_pred;
                gt = u
            if dir == 'y':
                pred = v_pred;
                gt = v

        elif dir == 'vm':

            vm_pred = torch.sqrt(sxx_pred ** 2 - sxx_pred * syy_pred + syy_pred ** 2 + 3 * sxy_pred ** 2)

            pred = vm_pred
            gt = vm
            # Mises 专属的相对 L2 误差计算
            L2_relative = torch.sqrt(torch.sum((vm_pred * flag - vm * flag) ** 2, -1)) / torch.sqrt(
                torch.sum((vm * flag) ** 2, -1))

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
    except:
        print('No trained models, starting from scratch')
    model = model.to(device)

    # start the training
    if args.phase in ['train', 'sup_train']:
        min_val_err = np.inf
        avg_mse_loss = 0.0
        # === 新增：单独记录 5 个损失项 ===
        avg_loss_u = 0.0
        avg_loss_v = 0.0
        avg_loss_sxx = 0.0
        avg_loss_syy = 0.0
        avg_loss_sxy = 0.0

        for e in pbar:
            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err = val(model, val_loader, args, device, num_nodes_list)
                err_hist.append(err)

                # === 修复：纯数据驱动没有采样循环，直接除以 batch 数量即可 ===
                if e > 0:
                    train_loss_hist.append(avg_mse_loss / (vf * len(train_loader)))
                else:
                    train_loss_hist.append(float('nan'))  # 第 0 轮还没开始训练，用 nan 占位避免曲线跳水

                print(f'\nEpoch {e} - Validation L2 Error: {err:.6f}')
                print(f'Total MSE Loss: {(avg_mse_loss / (vf * len(train_loader))):.6f}')
                print(f'  ├─ Loss U:   {(avg_loss_u / (vf * len(train_loader))):.6f}')
                print(f'  ├─ Loss V:   {(avg_loss_v / (vf * len(train_loader))):.6f}')
                print(f'  ├─ Loss Sxx: {(avg_loss_sxx / (vf * len(train_loader))):.6f}')
                print(f'  ├─ Loss Syy: {(avg_loss_syy / (vf * len(train_loader))):.6f}')
                print(f'  └─ Loss Sxy: {(avg_loss_sxy / (vf * len(train_loader))):.6f}')

                avg_mse_loss = 0.0
                avg_loss_u = 0.0
                avg_loss_v = 0.0
                avg_loss_sxx = 0.0
                avg_loss_syy = 0.0
                avg_loss_sxy = 0.0

                if err < min_val_err:
                    torch.save(model.state_dict(),
                               r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data,
                                                                                    args.model))
                    min_val_err = err

            # train one epoch
            model.train()
            # === 使用最新的 7元素解包 ===
            for (coors, u, v, sxx, syy, sxy, flag, geo, in_disp, in_force, f_type, vm) in train_loader:

                # 对于纯数据驱动，不需要采样，直接使用全部节点拟合
                all_coors = coors.float().to(device)
                all_flag = flag.float().to(device)  # (B, M)
                u_gt = u.float().to(device)
                v_gt = v.float().to(device)
                sxx_gt = sxx.float().to(device)
                syy_gt = syy.float().to(device)
                sxy_gt = sxy.float().to(device)
                vm_gt = vm.float().to(device)
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

                # 可以统一用 1.0 的权重，或者专门给应力加大一点权重（这里先平等对待）
                total_loss = loss_u + loss_v + loss_sxx + loss_syy + loss_sxy

                # === 新增：单独存储各项损失 ===
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

    # final test (包含 vm 绘图)
    model.load_state_dict(
        torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model),
                   weights_only=True))
    model.eval()
    err = test(model, test_loader, args, device, num_nodes_list, params, dir='x')
    _ = test(model, test_loader, args, device, num_nodes_list, params, dir='y')
    _ = test(model, test_loader, args, device, num_nodes_list, params, dir='vm')
    print('================================')
    print('Best L2 relative error on test loader:', err)

