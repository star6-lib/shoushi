import scipy.io as sio
import numpy as np
import torch
import os


# function to generate data loader for 2D plate stress problem
def generate_plate_stress_data_loader(args, config):
    # load the data
    mat_contents = sio.loadmat(r'./data/{}.mat'.format(args.data))
    u = mat_contents['final_u'][0]  # list of M elements
    v = mat_contents['final_v'][0]  # list of M elements
    sxx = mat_contents['final_sxx'][0]
    syy = mat_contents['final_syy'][0]
    sxy = mat_contents['final_sxy'][0]
    coors = mat_contents['coors_dict'][0]  # list of M elements
    vm = mat_contents['final_vonmises'][0]

    flag_BC_load = mat_contents['flag_BC_load_dict'][0]
    flag_BCxy = mat_contents['flag_BCxy_dict'][0]
    flag_BCy = mat_contents['flag_BCy_dict'][0]
    flag_hole = mat_contents['flag_hole_dict'][0]
    geo_val = mat_contents['geo_param_dict'][0]

    # === 新增：直接从 .mat 提取 101 维函数式数据 ===
    in_disp = mat_contents['input_disp_data'][0]
    in_force = mat_contents['input_force_data'][0]
    f_type = mat_contents['f_type'].flatten()  # 提取样本类型 (1,2为受力; 3,4为受位移)

    nu = mat_contents['poisson'][0][0]  # scalar
    young = mat_contents['young'][0][0]  # scalar
    # element_size = mat_contents['element_size'][0][0]   # scalar

    # scale the young's module
    scalar_factor = 1.0
    young = young * scalar_factor

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
        num_pde = np.sum((1 - flag_BC_load[i]) * (1 - flag_BCxy[i]) * (1 - flag_BCy[i]) * (1 - flag_hole[i]))
        if num_pde > max_pde_nodes:
            max_pde_nodes = num_pde
        num_par_ = np.sum((flag_BC_load[i]))
        if num_par_ > max_par_nodes:
            max_par_nodes = num_par_
        num_bcy = np.sum(flag_BCy[i])
        if num_bcy > max_bcy_nodes:
            max_bcy_nodes = num_bcy
        num_bcxy = np.sum(flag_BCxy[i])
        if num_bcxy > max_bcxy_nodes:
            max_bcxy_nodes = num_bcxy
        num_hole = np.sum(flag_hole[i])
        if num_hole > max_hole_nodes:
            max_hole_nodes = num_hole

    max_pde_nodes = int(max_pde_nodes)
    max_bcxy_nodes = int(max_bcxy_nodes)
    max_bcy_nodes = int(max_bcy_nodes)
    max_par_nodes = int(max_par_nodes)
    max_hole_nodes = int(max_hole_nodes)
    print(
        f"Max nodes - PDE: {max_pde_nodes}, Branch: {max_par_nodes}, BCy: {max_bcy_nodes}, BCxy: {max_bcxy_nodes}, Hole: {max_hole_nodes}")

    # append zeros to the data
    uT, vT, vmT = [], [], []
    sxxT, syyT, sxyT = [], [], []
    coorT, flagT = [], []

    for i in range(datasize):
        # extract the index of pde nodes and bc nodes
        pde_idx = np.where((1 - flag_BC_load[i]) * (1 - flag_BCxy[i]) * (1 - flag_BCy[i]) * (1 - flag_hole[i]) == 1)[0]
        bc_load_idx = np.where(flag_BC_load[i] == 1)[0]
        bcy_idx = np.where(flag_BCy[i] == 1)[0]
        bcxy_idx = np.where(flag_BCxy[i] == 1)[0]
        hole_idx = np.where(flag_hole[i] == 1)[0]

        # get the number
        num_pde = np.size(pde_idx)
        num_load = np.size(bc_load_idx)
        num_bcy = np.size(bcy_idx)
        num_bcxy = np.size(bcxy_idx)
        num_hole = np.size(hole_idx)

        # re-organize solution
        up = u[i]
        up = np.concatenate((
            up[pde_idx, :], np.zeros((max_pde_nodes - num_pde, 1)),
            up[bc_load_idx, :], np.zeros((max_par_nodes - num_load, 1)),
            up[bcy_idx, :], np.zeros((max_bcy_nodes - num_bcy, 1)),
            up[bcxy_idx, :], np.zeros((max_bcxy_nodes - num_bcxy, 1)),
            up[hole_idx, :], np.zeros((max_hole_nodes - num_hole, 1))
        ), 0)  # (max_pde+max_load+max_bcy+max_bcxy,1)
        uT.append(up)

        vp = v[i]
        vp = np.concatenate((
            vp[pde_idx, :], np.zeros((max_pde_nodes - num_pde, 1)),
            vp[bc_load_idx, :], np.zeros((max_par_nodes - num_load, 1)),
            vp[bcy_idx, :], np.zeros((max_bcy_nodes - num_bcy, 1)),
            vp[bcxy_idx, :], np.zeros((max_bcxy_nodes - num_bcxy, 1)),
            vp[hole_idx, :], np.zeros((max_hole_nodes - num_hole, 1))
        ), 0)  # (max_pde+max_load+max_bcy+max_bcxy,1)
        vT.append(vp)

        sxx_p = sxx[i]
        sxx_p = np.concatenate((sxx_p[pde_idx, :], np.zeros((max_pde_nodes - num_pde, 1)), sxx_p[bc_load_idx, :],
                                np.zeros((max_par_nodes - num_load, 1)), sxx_p[bcy_idx, :],
                                np.zeros((max_bcy_nodes - num_bcy, 1)), sxx_p[bcxy_idx, :],
                                np.zeros((max_bcxy_nodes - num_bcxy, 1)), sxx_p[hole_idx, :],
                                np.zeros((max_hole_nodes - num_hole, 1))), 0)
        sxxT.append(sxx_p)

        syy_p = syy[i]
        syy_p = np.concatenate((syy_p[pde_idx, :], np.zeros((max_pde_nodes - num_pde, 1)), syy_p[bc_load_idx, :],
                                np.zeros((max_par_nodes - num_load, 1)), syy_p[bcy_idx, :],
                                np.zeros((max_bcy_nodes - num_bcy, 1)), syy_p[bcxy_idx, :],
                                np.zeros((max_bcxy_nodes - num_bcxy, 1)), syy_p[hole_idx, :],
                                np.zeros((max_hole_nodes - num_hole, 1))), 0)
        syyT.append(syy_p)

        sxy_p = sxy[i]
        sxy_p = np.concatenate((sxy_p[pde_idx, :], np.zeros((max_pde_nodes - num_pde, 1)), sxy_p[bc_load_idx, :],
                                np.zeros((max_par_nodes - num_load, 1)), sxy_p[bcy_idx, :],
                                np.zeros((max_bcy_nodes - num_bcy, 1)), sxy_p[bcxy_idx, :],
                                np.zeros((max_bcxy_nodes - num_bcxy, 1)), sxy_p[hole_idx, :],
                                np.zeros((max_hole_nodes - num_hole, 1))), 0)
        sxyT.append(sxy_p)

        vmp = vm[i]
        vmp = np.concatenate((
            vmp[pde_idx, :], np.zeros((max_pde_nodes - num_pde, 1)),
            vmp[bc_load_idx, :], np.zeros((max_par_nodes - num_load, 1)),
            vmp[bcy_idx, :], np.zeros((max_bcy_nodes - num_bcy, 1)),
            vmp[bcxy_idx, :], np.zeros((max_bcxy_nodes - num_bcxy, 1)),
            vmp[hole_idx, :], np.zeros((max_hole_nodes - num_hole, 1))
        ), 0)
        vmT.append(vmp)

        # re-organize coors
        coorp = coors[i]
        coorp = np.concatenate((
            coorp[pde_idx, :], np.zeros((max_pde_nodes - num_pde, 2)),
            coorp[bc_load_idx, :], np.zeros((max_par_nodes - num_load, 2)),
            coorp[bcy_idx, :], np.zeros((max_bcy_nodes - num_bcy, 2)),
            coorp[bcxy_idx, :], np.zeros((max_bcxy_nodes - num_bcxy, 2)),
            coorp[hole_idx, :], np.zeros((max_hole_nodes - num_hole, 2))
        ), 0)  # (max_pde+max_load+max_bcy+max_bcxy,2)
        coorp = np.expand_dims(coorp, 0)  # (1, max_pde+max_load+max_bcy+max_bcxy,2)
        coorT.append(coorp)

        # re-organize node flags
        flagp = np.concatenate((
            np.ones_like(pde_idx), np.zeros((max_pde_nodes - num_pde)),
            np.ones_like(bc_load_idx), np.zeros((max_par_nodes - num_load)),
            np.ones_like(bcy_idx), np.zeros((max_bcy_nodes - num_bcy)),
            np.ones_like(bcxy_idx), np.zeros((max_bcxy_nodes - num_bcxy)),
            np.ones_like(hole_idx), np.zeros((max_hole_nodes - num_hole))
        ), 0)  # (max_pde+max_load+max_bcy+max_bcxy)
        flagp = np.expand_dims(flagp, 0)
        flagT.append(flagp)

    uT = np.concatenate(tuple(uT), -1).T  # (M, max_node)
    vT = np.concatenate(tuple(vT), -1).T  # (M, max_node)
    vmT = np.concatenate(tuple(vmT), -1).T
    coorT = np.concatenate(tuple(coorT), 0)  # (M, max_node, 2)
    flagT = np.concatenate(tuple(flagT), 0)  # (M, max_node)

    geoT = np.zeros((datasize, 12))
    for i in range(datasize):
        geoT[i, :] = geo_val[i].flatten()

    # 格式转换
    geoT = torch.from_numpy(geoT).float()
    uT = torch.from_numpy(uT)
    vT = torch.from_numpy(vT)
    vmT = torch.from_numpy(vmT)
    coorT = torch.from_numpy(coorT)
    flagT = torch.from_numpy(flagT)
    sxxT = torch.from_numpy(np.concatenate(tuple(sxxT), -1).T)
    syyT = torch.from_numpy(np.concatenate(tuple(syyT), -1).T)
    sxyT = torch.from_numpy(np.concatenate(tuple(sxyT), -1).T)

    # 将 101 维数据平铺并转为 Tensor
    in_disp_arr = np.array([in_disp[i].flatten() for i in range(datasize)])
    in_disp_arr = torch.from_numpy(in_disp_arr).float()

    in_force_arr = np.array([in_force[i].flatten() for i in range(datasize)])
    in_force_arr = torch.from_numpy(in_force_arr).float()

    f_type_arr = torch.from_numpy(f_type).float()

    print(uT.shape, vT.shape, coorT.shape, flagT.shape, geoT.shape)

    # split the data
    bar1 = [0, int(0.7 * datasize)]
    bar2 = [int(0.7 * datasize), int(0.8 * datasize)]
    bar3 = [int(0.8 * datasize), int(datasize)]

    train_dataset = torch.utils.data.TensorDataset(
        coorT[bar1[0]:bar1[1], :],
        uT[bar1[0]:bar1[1], :], vT[bar1[0]:bar1[1], :],
        sxxT[bar1[0]:bar1[1], :], syyT[bar1[0]:bar1[1], :], sxyT[bar1[0]:bar1[1], :],
        flagT[bar1[0]:bar1[1], :], geoT[bar1[0]:bar1[1], :],
        in_disp_arr[bar1[0]:bar1[1], :], in_force_arr[bar1[0]:bar1[1], :],
        f_type_arr[bar1[0]:bar1[1]], vmT[bar1[0]:bar1[1], :])

    val_dataset = torch.utils.data.TensorDataset(
        coorT[bar2[0]:bar2[1], :],
        uT[bar2[0]:bar2[1], :], vT[bar2[0]:bar2[1], :],
        sxxT[bar2[0]:bar2[1], :], syyT[bar2[0]:bar2[1], :], sxyT[bar2[0]:bar2[1], :],
        flagT[bar2[0]:bar2[1], :], geoT[bar2[0]:bar2[1], :],
        in_disp_arr[bar2[0]:bar2[1], :], in_force_arr[bar2[0]:bar2[1], :],
        f_type_arr[bar2[0]:bar2[1]], vmT[bar2[0]:bar2[1], :])

    test_dataset = torch.utils.data.TensorDataset(
        coorT[bar3[0]:bar3[1], :],
        uT[bar3[0]:bar3[1], :], vT[bar3[0]:bar3[1], :],
        sxxT[bar3[0]:bar3[1], :], syyT[bar3[0]:bar3[1], :], sxyT[bar3[0]:bar3[1], :],
        flagT[bar3[0]:bar3[1], :], geoT[bar3[0]:bar3[1], :],
        in_disp_arr[bar3[0]:bar3[1], :], in_force_arr[bar3[0]:bar3[1], :],
        f_type_arr[bar3[0]:bar3[1]], vmT[bar3[0]:bar3[1], :])

    # ================= 终极动态加速版 DataLoader =================
    # 动态获取当前集群分配的核心数 (Rorqual会自动读到 4，Fir会自动读到 8，本地跑默认给 4)
    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    print(f">>> 当前 DataLoader 使用的 worker 数量为: {num_cores}")

    # train_loader 动态分配 worker，彻底榨干 CPU！
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['train']['batchsize'],
        shuffle=True,
        num_workers=num_cores,  # <--- 自动匹配集群核心数！
        pin_memory=True,  # 开启锁页内存，向 GPU 传输直接起飞
        drop_last=False
    )

    # 验证集和测试集 batch_size 为 1，限制在 num_cores 的一半，且最高不超过 4
    val_test_workers = min(4, max(1, num_cores // 2))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=val_test_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=val_test_workers,
        pin_memory=True
    )
    # ====================================================================

    # store the number of nodes of different types
    # store the material properties
    num_nodes_list = (max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes, max_hole_nodes)
    params = (young, nu)

    return train_loader, val_loader, test_loader, num_nodes_list, params