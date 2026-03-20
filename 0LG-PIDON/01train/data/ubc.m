clear; clc;
rng('default');

% ================= Configuration from Paper Appendix D =================
% 数据范围: 物理坐标 [-10, 10] 对应数据索引 [0, 101]
N = 101;
xf = linspace(-10, 10, N);  % 物理坐标范围: -10 到 10 m
len = length(xf);
num_per_category = 1000; % Paper uses 1000 samples per category

% Initialize storage
% We will generate 4000 total samples:
% 1. Distributed Force (GRF)
% 2. Concentrated Force (Fourier Dirac)
% 3. Global Imposed Displacement (GRF)
% 4. Localized Imposed Displacement (Fourier Step)

total_samples = 4 * num_per_category;
data_matrix = zeros(total_samples, len);
type_list = zeros(total_samples, 1); 
% Type ID Key:
% 1 = Force BC (Distributed/GRF)
% 2 = Force BC (Concentrated/Fourier)
% 3 = Disp BC (Global/GRF)
% 4 = Disp BC (Localized/Fourier)

fprintf('Generating dataset based on Table D1...\n');

% ================= CATEGORY 1 & 3: GRF (Smooth) =================
% Table D1: "mean-zero, length-scale l=0.2" 
l_force = 4;  % Type 1 (分布力): 较短 length-scale
l_disp = 4;   % Type 3 (全局位移): 较长 length-scale (更平滑)

% Type 1: Force
cov_force = zeros(len, len);
for i = 1:len
    for j = 1:len
        cov_force(i, j) = exp(-0.5*((xf(i)-xf(j))^2) / l_force^2);
    end
end

% Type 3: Displacement (更平滑的 GRF)
cov_disp = zeros(len, len);
for i = 1:len
    for j = 1:len
        cov_disp(i, j) = exp(-0.5*((xf(i)-xf(j))^2) / l_disp^2);
    end
end

mu = zeros(1, len);

% Generate GRF samples
grf_force = mvnrnd(mu, cov_force, num_per_category);  % Type 1
grf_disp = mvnrnd(mu, cov_disp, num_per_category);   % Type 3

% ================= 缩放系数 (2026-03-08 调整) =================
% 根据弹性变形分析结果调整:
% 目标: 最大位移 < 0.2m (1% 应变)
%
% 分析结果:
% - Type 1 (分布力): 280m → 需要缩放 1/1400
% - Type 2 (集中力): 120575m → 需要缩放 1/600000
% - Type 3 (全局位移): 2m → 需要缩放 1/10
% - Type 4 (局部位移): 299m → 需要缩放 1/1500

SCALE_DISP = 0.20;        % Type 3 (全局位移): 原2.0 → 0.20 (约10倍缩小)
SCALE_FORCE = 3;           % Type 1 (分布力): 原5000 → 3 (约1666倍缩小)

% 集中力振幅 (Type 2)
CONC_FORCE_AMP = 10;        % Type 2 (集中力): 原500000 → 1 (约50万倍缩小!)

% 局部位移振幅 (Type 4)
LOC_DISP_AMP = 0.2;        % Type 4 (局部位移): 原300 → 0.2 (约1500倍缩小)

% --- 2. Type 1: 分布力 (使用前 1000 条数据) ---
% 逻辑：取出数据 -> 归一化到 [-1, 1] -> 乘以力的系数
raw_data_force = grf_force;
% 逐行归一化 (确保每条曲线最大值都是 1，消除 GRF 随机幅度的不可控性)
max_vals_force = max(abs(raw_data_force), [], 2); 
norm_data_force = raw_data_force ./ max_vals_force; 
% 赋值并缩放
data_matrix(1:1000, :) = norm_data_force * SCALE_FORCE; 
type_list(1:1000) = 1;

% --- 3. Type 3: 全局位移 (使用后 1000 条数据) ---
% 逻辑：取出数据 -> 归一化到 [-1, 1] -> 乘以位移系数
raw_data_disp = grf_disp;
% 逐行归一化
max_vals_disp = max(abs(raw_data_disp), [], 2);
norm_data_disp = raw_data_disp ./ max_vals_disp;
% 赋值并缩放 (直接控制最大位移为 8mm)
data_matrix(2001:3000, :) = norm_data_disp * SCALE_DISP; 
type_list(2001:3000) = 3;

% ================= CATEGORY 2: Concentrated Force (Fourier Dirac) =================
% 目标: 与 Type 1 相同的位移响应
fprintf('Generating Concentrated Forces (Fourier Dirac)...\n');
num_terms = 40; % Sufficient terms for sharp spike
for i = 1:num_per_category
    x0 = (2*rand()-1) * 9; 
    amp = (2*rand()-1) * CONC_FORCE_AMP;
    
    signal = zeros(1, len);
    for k = 1:num_terms
        % Windowed Cosine Sum (Dirac approximation)
        w_k = 0.5 * (1 + cos(pi*k/num_terms));
        signal = signal + w_k * cos(k*pi*(xf - x0) / 10);
    end
    % Normalize and scale
    signal = signal / max(abs(signal)) * amp; 
    
    idx = 1000 + i;
    data_matrix(idx, :) = signal;
    type_list(idx) = 2;
end

% ================= CATEGORY 4: Localized Disp (Fourier Step) =================
% 目标: 局部高应变区域，峰值位移约 0.2m
fprintf('Generating Localized Displacements (Fourier Step)...\n');
for i = 1:num_per_category
    x_start = -8 + 10 * rand();
    width = 2 + 4 * rand();
    x_end = x_start + width;
    amp = (2*rand()-1) * LOC_DISP_AMP;
    
    signal = zeros(1, len);
    for k = 1:2:num_terms % Odd terms for box/step shape
        w_k = 0.5 * (1 + cos(pi*k/num_terms));
        term = (sin(k*pi*(xf - x_start)/20) - sin(k*pi*(xf - x_end)/20)) / k;
        signal = signal + w_k * term;
    end
    signal = signal / max(abs(signal)) * amp;
    
    idx = 3000 + i;
    data_matrix(idx, :) = signal;
    type_list(idx) = 4;
end

% ================= SHUFFLE & SAVE =================
rp = randperm(total_samples);
f_bc = data_matrix(rp, :);
f_type = type_list(rp);

% Save 'f_bc' (curves) and 'f_type' (1-4 labels)
save('bc_source.mat', 'f_bc', 'f_type');

fprintf('Saved 4000 samples to bc_source.mat.\n');
fprintf('Type Distribution:\n');
fprintf('1 (Dist Force): %d\n', sum(f_type==1));
fprintf('2 (Conc Force): %d\n', sum(f_type==2));
fprintf('3 (Glob Disp):  %d\n', sum(f_type==3));
fprintf('4 (Loc Disp):   %d\n', sum(f_type==4));