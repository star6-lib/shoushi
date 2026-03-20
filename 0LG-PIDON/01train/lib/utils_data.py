import scipy.io as sio
import numpy as np
import torch

# function to generate data loader for 2D plate stress problem
def generate_plate_stress_data_loader(args, config):

    # load the data
    mat_contents = sio.loadmat(r'./data/{}.mat'.format(args.data))

    # 读取 f_type 并筛选 type3 数据
    f_type_all = mat_contents['f_type'].flatten()
    datasize = len(f_type_all)
    print(f"总样本数: {datasize} 个")

    # 提取 type3 的数据
    u = mat_contents['final_u'][0]
    v = mat_contents['final_v'][0]
    coors = mat_contents['coors_dict'][0]

    flag_BC_load = mat_contents['flag_BC_load_dict'][0]
    flag_BCxy = mat_contents['flag_BCxy_dict'][0]
    flag_BCy = mat_contents['flag_BCy_dict'][0]
    flag_hole = mat_contents['flag_hole_dict'][0]
    geo_val = mat_contents['geo_param_dict'][0]

    # 提取 101 维输入
    in_force = mat_contents['input_force_data'][0]
    in_disp = mat_contents['input_disp_data'][0]

    in_force_arr = np.array([in_force[i].flatten() for i in range(datasize)])
    in_disp_arr = np.array([in_disp[i].flatten() for i in range(datasize)])

    # 标准化 (防止分支映射混乱)
    f_mean, f_std = in_force_arr.mean(), in_force_arr.std() + 1e-8
    d_mean, d_std = in_disp_arr.mean(), in_disp_arr.std() + 1e-8

    force_norm = (in_force_arr - f_mean) / f_std
    disp_norm = (in_disp_arr - d_mean) / d_std

    # young 和 poisson 可能是标量或向量
    young_val = mat_contents['young']
    poisson_val = mat_contents['poisson']

    if young_val.shape == ():
        young = float(young_val)
        nu = float(poisson_val)
    else:
        young = float(young_val[0][0])
        nu = float(poisson_val[0][0])

    # scale the young's module
    scalar_factor = 1
    young = young * scalar_factor

    # 转换为 Tensor，保留 force_raw 用于计算边界真实牵引力 Loss
    forceT = torch.from_numpy(force_norm).float()
    dispT = torch.from_numpy(disp_norm).float()
    force_rawT = torch.from_numpy(in_force_arr).float() * scalar_factor
    f_typeT = torch.from_numpy(f_type_all).float()

    '''
    prepare the data to support batchwise training
    '''
    # find the maximum number of nodes
    datasize = len(u)
    max_pde_nodes = 0
    max_par_nodes = 0
    max_bcy_nodes = 0
    max_bcxy_nodes = 0
    max_hole_nodes = 0
    for i in range(datasize):
        # PDE 节点: 不受任何边界约束的节点
        num_pde = np.sum((1-flag_BC_load[i])*(1-flag_BCxy[i])*(1-flag_BCy[i])*(1-flag_hole[i]))
        if num_pde > max_pde_nodes:
            max_pde_nodes = num_pde
        # Branch 输入: 上边界 (flag_BC_load)
        num_par_ = np.sum((flag_BC_load[i]))
        if num_par_ > max_par_nodes:
            max_par_nodes = num_par_
        # 左右边界自由 (flag_BCy)
        num_bcy = np.sum(flag_BCy[i])
        if num_bcy > max_bcy_nodes:
            max_bcy_nodes = num_bcy
        # 下底边固定 (flag_BCxy)
        num_bcxy = np.sum(flag_BCxy[i])
        if num_bcxy > max_bcxy_nodes:
            max_bcxy_nodes = num_bcxy
        # 孔洞固定 (flag_hole) - 用于 DG
        num_hole = np.sum(flag_hole[i])
        if num_hole > max_hole_nodes:
            max_hole_nodes = num_hole

    max_pde_nodes = int(max_pde_nodes)
    max_bcxy_nodes = int(max_bcxy_nodes)
    max_bcy_nodes = int(max_bcy_nodes)
    max_par_nodes = int(max_par_nodes)
    max_hole_nodes = int(max_hole_nodes)
    print(f"Max nodes - PDE: {max_pde_nodes}, Branch: {max_par_nodes}, BCy: {max_bcy_nodes}, BCxy: {max_bcxy_nodes}, Hole: {max_hole_nodes}")

    # append zeros to the data
    uT = []
    vT = []
    coorT = []
    flagT = []
    for i in range(datasize):
        # extract the index of pde nodes and bc nodes
        # PDE 节点: 不受任何边界约束 (不含 flag_hole)
        pde_idx = np.where((1-flag_BC_load[i])*(1-flag_BCxy[i])*(1-flag_BCy[i])*(1-flag_hole[i])==1)[0]
        # Branch 输入: 上边界 (flag_BC_load)
        bc_load_idx = np.where(flag_BC_load[i]==1)[0]
        # 左右边界自由 (flag_BCy)
        bcy_idx = np.where(flag_BCy[i]==1)[0]
        # 下底边固定 (flag_BCxy)
        bcxy_idx = np.where(flag_BCxy[i]==1)[0]
        # 孔洞边界 (flag_hole) - 用于 DG
        hole_idx = np.where(flag_hole[i]==1)[0]

        # get the number
        num_pde = np.size(pde_idx)
        num_load = np.size(bc_load_idx)
        num_bcy = np.size(bcy_idx)
        num_bcxy = np.size(bcxy_idx)
        num_hole = np.size(hole_idx)

        # re-organize solution u
        up = u[i]
        up = np.concatenate((
                            up[pde_idx,:], np.zeros((max_pde_nodes-num_pde,1)),
                            up[bc_load_idx,:], np.zeros((max_par_nodes-num_load,1)),
                            up[bcy_idx,:], np.zeros((max_bcy_nodes-num_bcy,1)),
                            up[bcxy_idx,:], np.zeros((max_bcxy_nodes-num_bcxy,1)),
                            up[hole_idx,:], np.zeros((max_hole_nodes-num_hole,1))
                            ), 0)    # (max_pde+max_load+max_bcy+max_bcxy+max_hole,1)
        uT.append(up)

        # re-organize solution v
        vp = v[i]
        vp = np.concatenate((
                            vp[pde_idx,:], np.zeros((max_pde_nodes-num_pde,1)),
                            vp[bc_load_idx,:], np.zeros((max_par_nodes-num_load,1)),
                            vp[bcy_idx,:], np.zeros((max_bcy_nodes-num_bcy,1)),
                            vp[bcxy_idx,:], np.zeros((max_bcxy_nodes-num_bcxy,1)),
                            vp[hole_idx,:], np.zeros((max_hole_nodes-num_hole,1))
                            ), 0)    # (max_pde+max_load+max_bcy+max_bcxy+max_hole,1)
        vT.append(vp)

        # re-organize coors
        coorp = coors[i]
        coorp = np.concatenate((
                            coorp[pde_idx,:], np.zeros((max_pde_nodes-num_pde,2)),
                            coorp[bc_load_idx,:], np.zeros((max_par_nodes-num_load,2)),
                            coorp[bcy_idx,:], np.zeros((max_bcy_nodes-num_bcy,2)),
                            coorp[bcxy_idx,:], np.zeros((max_bcxy_nodes-num_bcxy,2)),
                            coorp[hole_idx,:], np.zeros((max_hole_nodes-num_hole,2))
                            ), 0)    # (max_pde+max_load+max_bcy+max_bcxy+max_hole,2)
        coorp = np.expand_dims(coorp, 0)    # (1, max_pde+max_load+max_bcy+max_bcxy+max_hole,2)
        coorT.append(coorp)

        # re-organize node flags
        flagp = np.concatenate((
                            np.ones_like(pde_idx), np.zeros((max_pde_nodes-num_pde)),
                            np.ones_like(bc_load_idx), np.zeros((max_par_nodes-num_load)),
                            np.ones_like(bcy_idx), np.zeros((max_bcy_nodes-num_bcy)),
                            np.ones_like(bcxy_idx), np.zeros((max_bcxy_nodes-num_bcxy)),
                            np.ones_like(hole_idx), np.zeros((max_hole_nodes-num_hole))
                            ), 0)    # (max_pde+max_load+max_bcy+max_bcxy+max_hole)
        flagp = np.expand_dims(flagp, 0)
        flagT.append(flagp)

    
    uT = np.concatenate(tuple(uT), -1).T    # (M, max_node)
    vT = np.concatenate(tuple(vT), -1).T    # (M, max_node)
    coorT = np.concatenate(tuple(coorT), 0)    # (M, max_node, 2)
    flagT = np.concatenate(tuple(flagT), 0)    # (M, max_node)

    geoT = np.zeros((datasize, 12))
    for i in range(datasize):
        geoT[i, :] = geo_val[i].flatten()
    geoT = torch.from_numpy(geoT).float()

    uT = torch.from_numpy(uT)
    vT = torch.from_numpy(vT)
    coorT = torch.from_numpy(coorT)
    flagT = torch.from_numpy(flagT)

    print(f"Data Shapes - uT: {uT.shape}, coorT: {coorT.shape}, geoT: {geoT.shape}")
    print(f"Branch Inputs - Force: {forceT.shape}, Disp: {dispT.shape}")

    # split the data
    bar1 = [0, int(0.7 * datasize)]
    bar2 = [int(0.7 * datasize), int(0.8 * datasize)]
    bar3 = [int(0.8 * datasize), int(datasize)]

    # === 新版 TensorDataset 打包 (严格对应 train 文件中的解包顺序) ===
    train_dataset = torch.utils.data.TensorDataset(
        forceT[bar1[0]:bar1[1], :], dispT[bar1[0]:bar1[1], :],
        force_rawT[bar1[0]:bar1[1], :], f_typeT[bar1[0]:bar1[1]],
        coorT[bar1[0]:bar1[1], :], uT[bar1[0]:bar1[1], :], vT[bar1[0]:bar1[1], :],
        flagT[bar1[0]:bar1[1], :], geoT[bar1[0]:bar1[1], :])

    val_dataset = torch.utils.data.TensorDataset(
        forceT[bar2[0]:bar2[1], :], dispT[bar2[0]:bar2[1], :],
        force_rawT[bar2[0]:bar2[1], :], f_typeT[bar2[0]:bar2[1]],
        coorT[bar2[0]:bar2[1], :], uT[bar2[0]:bar2[1], :], vT[bar2[0]:bar2[1], :],
        flagT[bar2[0]:bar2[1], :], geoT[bar2[0]:bar2[1], :])

    test_dataset = torch.utils.data.TensorDataset(
        forceT[bar3[0]:bar3[1], :], dispT[bar3[0]:bar3[1], :],
        force_rawT[bar3[0]:bar3[1], :], f_typeT[bar3[0]:bar3[1]],
        coorT[bar3[0]:bar3[1], :], uT[bar3[0]:bar3[1], :], vT[bar3[0]:bar3[1], :],
        flagT[bar3[0]:bar3[1], :], geoT[bar3[0]:bar3[1], :])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True,
                                                num_workers = 8,  # 召唤 8 个 CPU 子进程并行拉取数据
                                                pin_memory = True  # 开启锁页内存，让数据无缝直达 H100 显存，极速飙车！
                                                )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers = 4,  # 召唤 4 个 CPU 子进程并行拉取数据
                                                pin_memory = True  # 开启锁页内存，让数据无缝直达 H100 显存，极速飙车！
                                                )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=4,  # 召唤 8 个 CPU 子进程并行拉取数据
                                              pin_memory=True  # 开启锁页内存，让数据无缝直达 H100 显存，极速飙车！
                                              )

    # store the number of nodes of different types
    # 顺序: max_pde, max_bcxy(下底边), max_bcy(左右), max_par(上边界), max_hole(孔洞)
    num_nodes_list = (max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes, max_hole_nodes)
    params = (young, nu)

    return train_loader, val_loader, test_loader, num_nodes_list, params