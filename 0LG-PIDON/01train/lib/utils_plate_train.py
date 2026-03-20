import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

# === 1. 新增：系统信号监听库 ===
import signal
import sys

# 定义全局中断标志
interrupted = False

def signal_handler(signum, frame):
    global interrupted
    print(f"\n\n[🚨 警告] 接收到服务器强制中止信号 (Signal: {signum})！")
    print(">>> 正在启动紧急避险程序，准备保存当前进度并绘制云图...")
    interrupted = True

# 注册监听 Ctrl+C (SIGINT) 和 Slurm 中断信号 (SIGTERM)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


from .utils_losses import plate_stress_loss, constitutive_loss, hole_free_loss


# === 新增辅助函数: 计算物理点上的真实表面牵引力 ===
def get_target_traction(x_pts, force_raw_101):
    """将101维的力向量映射到上边界真实的物理坐标节点上"""
    # x_pts 范围在 [-10, 10]，映射到 index [0, 100]
    idx_float = (x_pts + 10.0) / 20.0 * 100.0
    idx_float = torch.clamp(idx_float, 0, 100)
    idx_floor = torch.floor(idx_float).long()
    idx_ceil = torch.ceil(idx_float).long()
    weight_ceil = idx_float - idx_floor
    weight_floor = 1.0 - weight_ceil

    B, M = x_pts.shape
    batch_idx = torch.arange(B).view(-1, 1).expand(B, M).to(x_pts.device)

    val_floor = force_raw_101[batch_idx, idx_floor]
    val_ceil = force_raw_101[batch_idx, idx_ceil]

    t_target = val_floor * weight_floor + val_ceil * weight_ceil
    return t_target


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
    for (force, disp, force_raw, f_type, coors, u, v, flag, geo_params) in loader:

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
        force = force.float().to(device)
        disp = disp.float().to(device)
        coors = coors.float().to(device)
        u = u.float().to(device)
        v = v.float().to(device)
        flag = flag.float().to(device)

        # model forward (使用 force, disp 替代原先的 par, par_flag)
        u_pred, v_pred, _, _, _ = model(coors[:,:,0], coors[:,:,1], force, disp, shape_coors, shape_flag)

        L2_relative = torch.sqrt(
            torch.sum((u_pred * flag - u * flag) ** 2 + (v_pred * flag - v * flag) ** 2, -1)) / torch.sqrt(
            torch.sum((u * flag) ** 2 + (v * flag) ** 2, -1))
        mean_relative_L2 += torch.sum(L2_relative).detach().cpu().item()
        num_eval += force.shape[0]

    mean_relative_L2 /= num_eval
    return mean_relative_L2


# testing function
def test(model, loader, args, device, num_nodes_list, dir):
    # transforme state to be eval
    model.eval()

    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes, max_hole_nodes = num_nodes_list

    mean_relative_L2 = 0
    num_eval = 0
    max_relative_err = -1
    min_relative_err = np.inf
    for (force, disp, force_raw, f_type, coors, u, v, flag, geo_params) in loader:

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
        force = force.float().to(device)
        disp = disp.float().to(device)
        coors = coors.float().to(device)
        u = u.float().to(device)
        v = v.float().to(device)
        flag = flag.float().to(device)

        # model forward
        u_pred, v_pred, _, _, _ = model(coors[:,:,0], coors[:,:,1], force, disp, shape_coors, shape_flag)

        # compute L2 error
        L2_relative = torch.sqrt(
            torch.sum((u_pred * flag - u * flag) ** 2 + (v_pred * flag - v * flag) ** 2, -1)) / torch.sqrt(
            torch.sum((u * flag) ** 2 + (v * flag) ** 2, -1))

        # get the prediction that we want
        if dir == 'x':
            pred = u_pred
            gt = u
        if dir == 'y':
            pred = v_pred
            gt = v

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
            worst_ff = worst_ff[valid_id]
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
            best_ff = best_ff[valid_id]

        # compute average error
        mean_relative_L2 += torch.sum(L2_relative).detach().cpu().item()
        num_eval += force.shape[0]

    mean_relative_L2 /= num_eval

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
    plt.subplot(2, 3, 1)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_f, cmap=cm, vmin=worst_min_color, vmax=worst_max_color, marker='o', s=3)
    plt.colorbar()
    plt.title('prediction')
    plt.subplot(2, 3, 2)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_gt, cmap=cm, vmin=worst_min_color, vmax=worst_max_color, marker='o', s=3)
    plt.title('ground truth')
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.scatter(worst_xcoor, worst_ycoor, c=np.abs(worst_f - worst_gt), cmap=cm, vmin=0, vmax=worst_err_max, marker='o',
                s=3)
    plt.title('absolute error')
    plt.colorbar()
    plt.subplot(2, 3, 4)
    plt.scatter(best_xcoor, best_ycoor, c=best_f, cmap=cm, vmin=best_min_color, vmax=best_max_color, marker='o', s=3)
    plt.colorbar()
    plt.title('prediction')
    plt.subplot(2, 3, 5)
    plt.scatter(best_xcoor, best_ycoor, c=best_gt, cmap=cm, vmin=best_min_color, vmax=best_max_color, marker='o', s=3)
    plt.title('ground truth')
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.scatter(best_xcoor, best_ycoor, c=np.abs(best_f - best_gt), cmap=cm, vmin=0, vmax=best_err_max, marker='o',
                s=3)
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
    for (force, disp, force_raw, f_type, coors, u, v, flag, geo_params) in loader:

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

        coors = coors.float().to(device)

        # model forward
        Geo_embeddings = model.predict_geometry_embedding(coors[:, :, 0], coors[:, :, 1], shape_coors, shape_flag)
        all_geo_embeddings.append(Geo_embeddings)

    all_geo_embeddings = torch.cat(tuple(all_geo_embeddings), 0)
    return all_geo_embeddings


# define the training function
def train(args, config, model, device, loaders, num_nodes_list, params):

    # === 加上这一句，声明整个 train 函数都使用全局的 interrupted ===
    global interrupted

    # print training configuration
    print('================================')
    print('Training Configuration:')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])

    # === 新增：将 Loss 权重清晰打印到日志中 ===
    print('--------------------------------')
    print('Loss Weights Configuration:')
    print('weight_load (Force/Disp BC):', config['train']['weight_load'])
    print('weight_fix (Fixed BC):      ', config['train']['weight_fix'])
    print('weight_pde (Equilibrium):   ', config['train']['weight_pde'])
    print('weight_ce (Constitutive):   ', config['train']['weight_ce'])
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

    # learning rate scheduler
    lr_scheduler = None
    if config['train'].get('lr_decay_type', 'step') == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['train'].get('lr_decay_step', 50),
            gamma=config['train'].get('lr_decay_rate', 0.5)
        )
    elif config['train'].get('lr_decay_type') == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config['train'].get('lr_decay_gamma', 0.95)
        )
    elif config['train'].get('lr_decay_type') == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train']['epochs'],
            eta_min=config['train'].get('lr_min', 1e-6)
        )
    elif config['train'].get('lr_decay_type') == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['train'].get('lr_decay_rate', 0.5),
            patience=config['train'].get('lr_patience', 10),
            min_lr=config['train'].get('lr_min', 1e-6)
        )

    # print learning rate decay info
    if lr_scheduler is not None:
        print('Learning rate decay type:', config['train'].get('lr_decay_type', 'step'))

    # visual frequency
    vf = config['train']['visual_freq']

    # err history
    err_hist = []

    # move the model to the defined device
    try:
        model.load_state_dict(
            torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model)))
    except:
        print('No trained models')
    model = model.to(device)

    # define tradeoff weights
    weight_load = config['train']['weight_load']
    weight_pde = config['train']['weight_pde']
    weight_fix = config['train']['weight_fix']
    weight_free = config['train']['weight_free']
    weight_ce = config['train']['weight_ce']

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        avg_pde_loss = np.inf
        avg_ce_loss = np.inf
        avg_fix_loss = np.inf
        avg_free_loss = np.inf
        avg_load_loss = np.inf

        for e in pbar:
            # === 2. 新增：检查是否被服务器打断 ===
            if interrupted:
                print(f"\n>>> 🛑 训练在第 {e} 轮被紧急中止！正在跳出循环...")
                break  # 触发 break 后，代码会自动跳到最后的 test 画图阶段
            # ====================================
            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err = val(model, val_loader, args, device, num_nodes_list)
                err_hist.append(err)
                print('Current epoch error:', err)
                print('current epochs pde loss:', avg_pde_loss)
                print('CE (Hooke):', avg_ce_loss)
                print('fix bc loss:', avg_fix_loss)
                print('free bc loss:', avg_free_loss)
                print('load bc loss:', avg_load_loss)

                # === 3. 新增：强制把日志刷入硬盘，防止进程被杀时日志截断 ===
                sys.stdout.flush()

                avg_pde_loss = 0
                avg_ce_loss = 0
                avg_fix_loss = 0
                avg_free_loss = 0
                avg_load_loss = 0
                if err < min_val_err:

                    torch.save(model.state_dict(),
                               r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data,
                                                                                    args.model))
                    min_val_err = err

            # train one epoch
            model.train()
            # === 解包新增数据 ===
            for (force, disp, force_raw, f_type, coors, u, v, flag, geo_params) in train_loader:

                # 转移到 device
                force = force.float().to(device)
                disp = disp.float().to(device)
                force_raw = force_raw.float().to(device)
                f_type = f_type.to(device)

                for _ in range(config['train']['coor_sampling_freq']):

                    # random sampling for PDE residual computation
                    ss_index = np.random.choice(np.arange(max_pde_nodes), config['train']['coor_sampling_size'])
                    pde_sampled_coors = coors[:, ss_index, :]
                    pde_sampled_coors = pde_sampled_coors.float().to(device)  # (B, Ms, 2)
                    pde_flag = flag[:, ss_index]
                    pde_flag = pde_flag.float().to(device)  # (B, Ms)

                    # extract bc loading coordinates (上边界)
                    ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_par_nodes)
                    load_coors = coors[:, ss_index, :].float().to(device)  # (B, max_par, 2)
                    load_flag = flag[:, ss_index]
                    load_flag = load_flag.float().to(device)  # (B, max_par)
                    u_load_gt = u[:, ss_index].float().to(device)  # (B, max_par)
                    v_load_gt = v[:, ss_index].float().to(device)  # (B, max_par)

                    # extract bc free condition coordinates (左右边界)
                    ss_index = np.arange(max_pde_nodes + max_par_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes)
                    bcy_coors = coors[:, ss_index, :].float().to(device)  # (B, max_bcy, 2)
                    bcy_flag = flag[:, ss_index]
                    bcy_flag = bcy_flag.float().to(device)  # (B, max_bcy)

                    # extract the bottom edge fixed condition coordinates (下底边)
                    ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes,
                                         max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
                    bcxy_coors = coors[:, ss_index, :].float().to(device)  # (B, max_bcxy, 2)
                    bcxy_flag = flag[:, ss_index]
                    bcxy_flag = bcxy_flag.float().to(device)  # (B, max_bcxy)

                    # extract the hole fixed condition coordinates (孔洞边界)
                    ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes,
                                         max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                    hole_coors = coors[:, ss_index, :].float().to(device)  # (B, max_hole, 2)
                    hole_flag = flag[:, ss_index]
                    hole_flag = hole_flag.float().to(device)  # (B, max_hole)

                    # extract the boundary of the varying shape (used as DG geometry descriptor)
                    if args.geo_node in ('vary_bound', 'vary_bound_sup'):
                        ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes,
                                             max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                    if args.geo_node == 'all_bound':
                        ss_index = np.arange(max_pde_nodes,
                                             max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                    if args.geo_node == 'all_domain':
                        ss_index = np.arange(0,
                                             max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes + max_hole_nodes)
                    shape_coor = coors[:, ss_index, :].float().to(device)  # (B, max_hole, 2)
                    shape_flag = flag[:, ss_index]
                    shape_flag = shape_flag.float().to(device)  # (B, max_hole)

                    # ====== 网络计算与边界约束（0阶导数，不需要requires_grad!）======
                    # forward to get the prediction on bottom edge fixed boundary (下底边)
                    u_BCxy_pred, v_BCxy_pred, _, _, _= model(bcxy_coors[:, :, 0], bcxy_coors[:, :, 1], force, disp, shape_coor,
                                                     shape_flag)
                    fix_loss = torch.mean((u_BCxy_pred * bcxy_flag) ** 2) + torch.mean((v_BCxy_pred * bcxy_flag) ** 2)

                    # forward to get the prediction on hole free boundary (孔洞)
                    geo_params = geo_params.float().to(device)
                    _, _, sxx_hole, syy_hole, sxy_hole = model(hole_coors[:, :, 0], hole_coors[:, :, 1], force, disp,
                                                               shape_coor, shape_flag)
                    Tx_hole, Ty_hole = hole_free_loss(sxx_hole, syy_hole, sxy_hole, hole_coors[:, :, 0],
                                                      hole_coors[:, :, 1], geo_params)

                    _, _, sxx_bcy, _, sxy_bcy = model(bcy_coors[:, :, 0], bcy_coors[:, :, 1], force, disp, shape_coor,
                                                      shape_flag)
                    free_loss = torch.mean((Tx_hole * hole_flag) ** 2) + torch.mean((Ty_hole * hole_flag) ** 2) + \
                                torch.mean((sxx_bcy * bcy_flag) ** 2) + torch.mean((sxy_bcy * bcy_flag) ** 2)

                    # === 改动核心：外部边界的自适应 Masking Loss (上边界) ===
                    u_load, v_load, _, syy_load, sxy_load = model(load_coors[:,:,0], load_coors[:,:,1], force, disp, shape_coor, shape_flag)

                    # 创建动态掩码 (Mask) - 强制转为整型比较，绝对安全
                    f_type_int = torch.round(f_type).long()
                    mask_force = ((f_type_int == 1) | (f_type_int == 2)).float().unsqueeze(1)
                    mask_disp = ((f_type_int == 3) | (f_type_int == 4)).float().unsqueeze(1)

                    # 1. 位移激励损失 (Type 3, 4) - MSE
                    loss_idbc = torch.mean(((u_load - u_load_gt) * load_flag * mask_disp)**2) +\
                                torch.mean(((v_load - v_load_gt) * load_flag * mask_disp)**2)

                    # 2. 力激励损失 (Type 1, 2) - 算牵引力 MSE
                    Ty_target = get_target_traction(load_coors[:, :, 0], force_raw)
                    Tx_target = torch.zeros_like(Ty_target)
                    loss_inbc = torch.mean(((syy_load - Ty_target) * load_flag * mask_force) ** 2) + torch.mean(
                        ((sxy_load - Tx_target) * load_flag * mask_force) ** 2)
                    load_loss = loss_idbc + loss_inbc



                    # forward to get the prediction on pde domian
                    x_pde = Variable(pde_sampled_coors[:, :, 0], requires_grad=True)
                    y_pde = Variable(pde_sampled_coors[:, :, 1], requires_grad=True)
                    u_pde, v_pde, sxx_pde, syy_pde, sxy_pde = model(x_pde, y_pde, force, disp, shape_coor, shape_flag)

                    # 平衡方程损失 (仅需应力一阶导)
                    rx, ry = plate_stress_loss(sxx_pde, syy_pde, sxy_pde, x_pde, y_pde)
                    pde_loss = torch.mean((rx * pde_flag) ** 2) + torch.mean((ry * pde_flag) ** 2)

                    # 本构方程损失 (仅需位移一阶导与应力对齐)
                    diff_sxx, diff_syy, diff_sxy = constitutive_loss(u_pde, v_pde, sxx_pde, syy_pde, sxy_pde, x_pde,
                                                                     y_pde, params)
                    ce_loss = torch.mean((diff_sxx * pde_flag) ** 2) + torch.mean(
                        (diff_syy * pde_flag) ** 2) + torch.mean((diff_sxy * pde_flag) ** 2)

                    # ====== 总损失组合 ======
                    total_loss = weight_pde * pde_loss + weight_ce * ce_loss + weight_load * load_loss + weight_fix * fix_loss + weight_free * free_loss

                    avg_pde_loss += pde_loss.detach().cpu().item()
                    avg_ce_loss += ce_loss.detach().cpu().item()
                    avg_fix_loss += fix_loss.detach().cpu().item()
                    avg_free_loss += free_loss.detach().cpu().item()
                    avg_load_loss += load_loss.detach().cpu().item()

                    # update parameter
                    optimizer.zero_grad()
                    total_loss.backward()

                    # === 新增：防爆核盾牌（梯度裁剪） ===
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # ===================================
                    optimizer.step()

                    # clear cuda
                    # torch.cuda.empty_cache()      #清空显存，主要作用域二阶求导存在时，显存不足情况

            # update learning rate
            if lr_scheduler is not None:
                if config['train'].get('lr_decay_type') == 'plateau':
                    lr_scheduler.step(err)
                else:
                    lr_scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                if e % vf == 0:
                    print('Current learning rate:', current_lr)

    # final test
    print('\n>>> 正在生成最终的物理云图并保存参数...')
    try:
        # 首选：尝试加载历史最好的模型来画图
        model.load_state_dict(
            torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model)))
        print("✅ 已加载 Best Model 用于绘制最终云图。")
    except:
        # 备选：如果因为中止太早没找到 best_model，就用刚刚存的 latest_model
        print("⚠️ 未找到 Best Model，正在使用 Latest Model 绘制云图。")
        model.load_state_dict(
            torch.load(r'./res/saved_models/latest_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model)))

    model.eval()
    err = test(model, test_loader, args, device, num_nodes_list, dir='x')
    _ = test(model, test_loader, args, device, num_nodes_list, dir='y')

    # 退出前最后刷一次日志
    if interrupted:
        print('\n>>> 🛑 避险程序执行完毕，在服务器关闭前已安全抢救数据并退出。')
    else:
        print('\n>>> 🎉 200 轮 Epoch 已全部正常训练完成，完美收官！')

    sys.stdout.flush()