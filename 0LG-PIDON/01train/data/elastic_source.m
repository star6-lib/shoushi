close all
clear
load bc_source.mat 

% ==================== CONFIGURATION ====================
% TYPE_SELECT: 选择生成哪种类型的数据
%   0: 生成所有类型 (默认)
%   1: 只生成 Type 1 (分布力)
%   2: 只生成 Type 2 (集中力)
%   3: 只生成 Type 3 (全局位移)
%   4: 只生成 Type 4 (局部位移)
TYPE_SELECT = 0;  

if TYPE_SELECT == 0
    % 生成所有类型
    f_bc = f_bc;
    f_type = f_type;
    num_samples = length(f_type);
else
    % 只选择特定类型
    idx = find(f_type == TYPE_SELECT);
    f_bc = f_bc(idx, :);
    f_type = f_type(idx);
    num_samples = length(idx);
    fprintf('选择生成 Type %d, 共 %d 个样本\n', TYPE_SELECT, num_samples);
end

% Hole Control: 'free' or 'fixed'
HOLE_BC_TYPE = 'free'; 

% ==================== Initialize Storage ====================
coors_dict = cell(1, num_samples);
final_u = cell(1, num_samples);
final_v = cell(1, num_samples);
geo_param_dict = cell(1, num_samples); 
flag_BC_load_dict = cell(1, num_samples); 
flag_BCxy_dict = cell(1, num_samples);    
flag_BCy_dict = cell(1, num_samples);     
flag_hole_dict = cell(1, num_samples);    

input_force_data = cell(1, num_samples);
input_disp_data = cell(1, num_samples);
young = 200;    % 弹性模量 (GPa) - 钢
poisson = 0.25;  % 泊松比标量

% ==================== 几何参数 (20倍放大，原点为中心) ====================
PLATE_SIZE = 20;          % 板边长 (m)
HALF_SIZE = PLATE_SIZE/2; % 半长 (10m)
HOLE_SCALE = 20;           % 孔洞尺寸缩放因子

fprintf('Plate Geometry: %dx%d m (centered at origin)\n', PLATE_SIZE, PLATE_SIZE);

fprintf('Running FEM with Hole BC: [%s]...\n', HOLE_BC_TYPE);

% [OPTIMIZATION] Define Names and Formula ONCE (Static)
% Since we always have 1 Rect + 4 Circles, we don't need to build this in the loop.
% decsg expects names in columns, so we transpose (') the char array.
fixed_names = char('R1', 'C1', 'C2', 'C3', 'C4')'; 
fixed_formula = 'R1-C1-C2-C3-C4';

for i = 1:num_samples
    if mod(i, 50) == 0, fprintf('Sample %d / %d...\n', i, num_samples); end
    
    % --- 1. Load Data & Material ---
    curve_data = f_bc(i, :);
    type_id = f_type(i);
    
    E_val = young;  % 使用标量值
    v_val = poisson;
    
    if type_id == 1 || type_id == 2
        mode = 'force';
        input_force_data{i} = curve_data;
        input_disp_data{i} = zeros(size(curve_data));
    else
        mode = 'disp';
        input_force_data{i} = zeros(size(curve_data));
        input_disp_data{i} = curve_data;
    end
    
    % --- 2. Geometry Generation ---
    % 改为平面应力 (适用于薄板)
    model = createpde('structural','static-planestress');
    
    % Rect Geometry (以原点为中心的正方形, 20x20)
    % 格式: [3, 4, x1, x2, x3, x4, y1, y2, y3, y4]
    % 右下角 (-10,0), (10,0), (10,20), (-10,20) -> 修正为 (-HALF_SIZE, -HALF_SIZE) 到 (HALF_SIZE, HALF_SIZE)
    rect_geo = [3, 4, -HALF_SIZE, HALF_SIZE, HALF_SIZE, -HALF_SIZE, -HALF_SIZE, -HALF_SIZE, HALF_SIZE, HALF_SIZE]';
    
    % Hole Parameters (20倍放大)
    % 原中心: (0.75, 0.75), (0.25, 0.75), (0.25, 0.25), (0.75, 0.25) -> 缩放到 [-10,10] 范围
    % 原始范围 [0,1] -> 新范围 [-10,10]: x_new = x_old * 20 - 10
    base_centers = [ 5,  5;  -5,  5;  -5, -5;   5, -5];  % 20x放大后的基础中心位置
    perturb = 0.075 * HOLE_SCALE;  % 1.5m
    hole_range = [0.14, 0.075] * HOLE_SCALE;  % [2.8, 1.5]m
    
    circles_geo = []; 
    current_holes = zeros(4,3);
    
    for k = 1:4
        r_rand = sqrt(rand())*perturb; th = rand()*2*pi;
        cx = base_centers(k,1) + r_rand*cos(th);
        cy = base_centers(k,2) + r_rand*sin(th);
        cr = hole_range(1) + diff(hole_range)*rand();
        current_holes(k,:) = [cx, cy, cr];
        C_col = [1; cx; cy; cr];
        circles_geo = [circles_geo, [C_col; zeros(6,1)]];
    end
    
    gm = [rect_geo, circles_geo];
    
    % [CRITICAL FIX]
    % 1. We use the static 'fixed_names' (which is correctly transposed).
    % 2. We use the static 'fixed_formula'.
    % 3. 'rect_geo' is used in gm, so no variable name conflict with 'R1'.
    dl = decsg(gm, fixed_formula, fixed_names); 
    
    geometryFromEdges(model, dl);
    
% --- 3. Boundary Edge Identification (严谨修复版) ---
    edges_top = []; 
    edges_bot = [];
    edges_holes = []; 

    tol_geo = 1e-4;
    for e = 1:size(dl, 2)
        % 判断当前边界是直线段(2)还是圆弧(1)
        if dl(1,e) == 1  % 如果是圆弧 (必然属于孔洞)
            edges_holes = [edges_holes, e];
        else             % 如果是直线 (必然属于外边框)
            xm = (dl(2,e) + dl(3,e))/2;
            ym = (dl(4,e) + dl(5,e))/2;
            
            if abs(ym - HALF_SIZE) < tol_geo
                edges_top = [edges_top, e];
            elseif abs(ym + HALF_SIZE) < tol_geo
                edges_bot = [edges_bot, e];
            end
        end
    end
    
    % --- 4. Physics Application ---
    structuralProperties(model, 'YoungsModulus', E_val, 'PoissonsRatio', v_val);
    structuralBC(model, 'Edge', edges_bot, 'Constraint', 'fixed');
    
    % Hole Constraints
    if strcmp(HOLE_BC_TYPE, 'fixed')
        structuralBC(model, 'Edge', edges_holes, 'Constraint', 'fixed');
    end
    
    global global_ubc_curve
    global_ubc_curve = curve_data; 
    
    if strcmp(mode, 'disp')
        structuralBC(model, 'Edge', edges_top, 'YDisplacement', @myload_wrapper, 'Vectorized', 'on');
    else
        structuralBoundaryLoad(model, 'Edge', edges_top, ...
            'SurfaceTraction', @(l,s) [zeros(1,numel(l.x)); myload_wrapper(l,s)], ...
            'Vectorized', 'on');
    end
    
    % --- 5. Solve & Save ---
    generateMesh(model, 'Hmax', 0.4);  % 网格尺寸随板放大 (原0.02*20=0.4)
    R = solve(model);
    
    nodes = R.Mesh.Nodes;
    final_u{i} = R.Displacement.ux;
    final_v{i} = R.Displacement.uy;
    coors_dict{i} = nodes';
    % 行形式: 左上、左下、右上、右下 (每孔 cx,cy,cr -> 共 1x12)
    order_hl = [2, 3, 1, 4];  % base_centers: k=2 左上, k=3 左下, k=1 右上, k=4 右下
    geo_param_dict{i} = reshape(current_holes(order_hl, :)', 1, 12); 
    
    % Flags (边界识别 - 20倍尺寸)
    xx = nodes(1,:); yy = nodes(2,:);
    tol = 1e-4;

    % 优先定义上下边界（包含角点）
    % Top: y = HALF_SIZE (10)
    is_top = abs(yy - HALF_SIZE) < tol;
    % Bottom: y = -HALF_SIZE (-10)
    is_bottom = abs(yy + HALF_SIZE) < tol;

    % 定义原始左右边界
    is_left_raw = abs(xx + HALF_SIZE) < tol;
    is_right_raw = abs(xx - HALF_SIZE) < tol;

    % 从左右边界中剔除角点
    % 逻辑：是左边 AND 不是上边 AND 不是下边
    is_left = is_left_raw & (~is_top) & (~is_bottom);
    is_right = is_right_raw & (~is_top) & (~is_bottom);

    is_hole_node = false(size(xx));
    for k = 1:4
        cx = current_holes(k, 1); cy = current_holes(k, 2); cr = current_holes(k, 3);
        dist_to_center = sqrt((xx - cx).^2 + (yy - cy).^2);
        is_hole_node = is_hole_node | (abs(dist_to_center - cr) < tol);
    end
    
    flag_BC_load_dict{i} = double(is_top)';                   
    flag_BCxy_dict{i} = double(is_bottom)';                   
    flag_BCy_dict{i} = double((is_left | is_right))';         
    flag_hole_dict{i} = double(is_hole_node)';                
end

save('plate_stree_DG.mat', 'coors_dict', 'final_u', 'final_v', ...
    'input_force_data', 'input_disp_data', 'young', 'poisson', 'geo_param_dict', ...
    'flag_BC_load_dict', 'flag_BCxy_dict', 'flag_BCy_dict', 'flag_hole_dict', 'f_type');
fprintf('Done. Dataset saved to plate_stree_DG.mat\n');
fprintf('共保存 %d 个样本\n', num_samples);