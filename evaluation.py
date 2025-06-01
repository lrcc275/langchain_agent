import numpy as np
import os
import glob
from reference_code import *
def check_matrices(A, B, percent):
    """
    检查两个矩阵的相似性
    
    参数:
    A (numpy.ndarray): 正确解矩阵
    B (numpy.ndarray): 近似解矩阵
    percent (float): 允许的最大相对误差阈值（小数形式）
    
    返回:
    int: 
        - -1 如果形状不同
        - 1 如果存在超过阈值的相对误差
        - 0 如果所有元素的相对误差都小于阈值
    """
    # 步骤1：检查形状是否相同
    if A.shape != B.shape:
        return -1
    
    # 计算绝对差值
    delta = np.abs(B - A)
    
    # 步骤2：检查相对误差
    # 处理A中非零元素
    non_zero_mask = (A != 0)
    if np.any(non_zero_mask):
        relative_error_non_zero = delta[non_zero_mask] / A[non_zero_mask]
        if np.any(relative_error_non_zero >= percent):
            return 1
    
    # 处理A中零元素（此时只需检查绝对差值）
    zero_mask = (A == 0)
    if np.any(zero_mask):
        if np.any(delta[zero_mask] > 0):  # 任何非零差值都会导致无穷大相对误差
            return 1
    
    return 0

def check_onefile(file_path1, file_path2, percent):
    sol = np.load(file_path1)
    actual = np.load(file_path2)
    return check_matrices(sol, actual, percent)


def check_files(dir_path1, dir_path2, prefix_suffix, mode, percent):
    """
    检查两个目录中匹配文件对的矩阵差异
    
    参数:
    dir_path1 (str): 包含正确解矩阵的目录路径
    dir_path2 (str): 包含当前解矩阵的目录路径
    prefix_suffix (str): 文件名前缀或后缀标识
    mode (int): 匹配模式 (0=前缀模式, 1=后缀模式)
    percent (float): 相对误差阈值
    
    返回:
    dict: 包含结果计数的字典 {-1: count, 0: count, 1: count}
    """
    # 初始化结果计数器
    results = {-1: 0, 0: 0, 1: 0}
    
    # 模式0: 前缀匹配
    if mode == 0:
        # 构建目录1中的文件模式 (sol_ + prefix_suffix + 任意字符 + .npy)
        pattern1 = os.path.join(dir_path1, f"sol_{prefix_suffix}*.npy")
        # 构建目录2中的文件模式 (prefix_suffix + 任意字符 + .npy)
        pattern2 = os.path.join(dir_path2, f"{prefix_suffix}*.npy")
        
        # 获取匹配的文件列表
        files1 = sorted(glob.glob(pattern1))
        files2 = sorted(glob.glob(pattern2))
        
        # 提取文件基名用于配对
        base_names1 = [os.path.basename(f)[4:] for f in files1]  # 去掉"sol_"前缀
        base_names2 = [os.path.basename(f) for f in files2]
        print(base_names1)
        print(base_names2)
    # 模式1: 后缀匹配
    elif mode == 1:
        # 构建目录1中的文件模式 (任意字符 + prefix_suffix + _sol.npy)
        pattern1 = os.path.join(dir_path1, f"*{prefix_suffix}_sol.npy")
        # 构建目录2中的文件模式 (任意字符 + prefix_suffix + .npy)
        pattern2 = os.path.join(dir_path2, f"*{prefix_suffix}.npy")
        
        # 获取匹配的文件列表
        files1 = sorted(glob.glob(pattern1))
        files2 = sorted(glob.glob(pattern2))
        
        # 提取文件基名用于配对
        # 去掉"_sol"后缀
        base_names1 = [os.path.basename(f).replace("_sol", "") for f in files1]
        base_names2 = [os.path.basename(f) for f in files2]
        print(base_names1)
        print(base_names2)
    else:
        raise ValueError("无效的模式，必须是0或1")
    
    # 创建基名到文件路径的映射
    file_map1 = dict(zip(base_names1, files1))
    file_map2 = dict(zip(base_names2, files2))
    
    # 查找共同基名（配对文件）
    common_bases = set(base_names1) & set(base_names2)
    
    # 处理每对匹配的文件
    for base in common_bases:
        # 加载两个矩阵
        A = np.load(file_map1[base])
        B = np.load(file_map2[base])
        print("1")
        # 检查矩阵并记录结果
        result = check_matrices(A, B, percent)
        results[result] += 1
    
    return results

def check(file_path1, file_path2, percent, task):
    if os.path.isfile(file_path1) == False:
        return -1
    if os.path.isfile(file_path2) == False:
        return -1
    if task == 1:
        print("1")
        sol_1(file_path1,"solve.npy")
    elif task == 2:
        sol_2(file_path1,"solve.npy")
    elif task == 3:
        sol_3(file_path1,"solve.npy")
    elif task == 4:
        sol_4(file_path1,"solve.npy")
    elif task == 5:
        sol_5(file_path1,"solve.npy")
    elif task == 6:
        sol_6(file_path1,"solve.npy")
    elif task == 7:
        sol_7(file_path1,"solve.npy")
    elif task == 8:
        sol_8(file_path1,"solve.npy")
    # elif task == 9:
    #     sol_9(file_path1,"solve.npy")
    elif task == 10:
        sol_10(file_path1,"solve.npy")
    elif task == 11:
        sol_11(file_path1,"solve.npy")
    return check_onefile("solve.npy", file_path2, percent)
