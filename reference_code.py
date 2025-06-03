import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy import signal
from scipy.signal import coherence
import seaborn as sns
from scipy.fftpack import fft, fftfreq
from sklearn.decomposition import FastICA
import statsmodels.tsa.stattools as smt
from sklearn.cluster import KMeans
import scipy.stats
import antropy

def sol_1(input_filename, output_stats_filename):
    try:
        # 读取数据
        data = np.load(input_filename)
        print(f"成功读取数据。数组形状为: {data.shape}")

        num_channels = data.shape[0]
        num_samples = data.shape[1]

        # 计算每个通道的统计量
        stats_results = np.zeros((num_channels, 3))
        print("计算每个通道的峰峰值、均值和方差...")
        for i in range(num_channels):
            channel_data = data[i, :]
            ptp_val = np.ptp(channel_data)     # 峰峰值 (Peak-to-Peak)
            mean_val = np.mean(channel_data)   # 均值
            var_val = np.var(channel_data)     # 方差 (默认计算总体方差)
            stats_results[i, 0] = ptp_val
            stats_results[i, 1] = mean_val
            stats_results[i, 2] = var_val

        # 保存统计结果
        np.save(output_stats_filename, stats_results)
        print(f"统计结果已保存至 {output_stats_filename}")
        
    except FileNotFoundError:
        print(f"错误: 文件 '{input_filename}' 不存在")
    except Exception as e:
        print(f"错误: 处理数据时发生异常: {e}")

import numpy as np
from scipy import signal

def sol_2(input_file, output_file):
    """
    计算EEG数据在不同频带的功率
    
    参数:
    input_file (str): 输入数据文件路径 (.npy格式)
    output_file (str): 输出结果文件路径 (.npy格式)
    
    返回:
    numpy.ndarray: 形状为(n_windows, n_channels, 4)的数组，包含各窗口各通道的频带功率
    """
    # 加载数据
    try:
        data = np.load(input_file)
    except FileNotFoundError:
        print(f"错误: 文件 {input_file} 不存在")
        return None
    except Exception as e:
        print(f"错误: 加载文件时发生异常: {e}")
        return None
    
    # 设置参数
    fs = 250  # 采样率
    window_size = 30 * fs  # 30秒窗口
    step_size = 10 * fs    # 10秒步长
    
    # 定义频带
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30)
    }
    
    # 初始化结果列表
    results = []
    
    # 处理每个窗口
    for start in range(0, data.shape[1] - window_size + 1, step_size):
        window = data[:, start:start + window_size]
        band_powers = []
        
        # 计算每个通道的频带功率
        for channel in window:
            # 使用Welch方法计算功率谱密度
            freqs, psd = signal.welch(channel, fs=fs, nperseg=window_size)
            
            # 计算每个频带的功率
            channel_bands = []
            for band, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs < high)
                channel_bands.append(np.sum(psd[band_mask]))
            
            band_powers.append(channel_bands)
        
        results.append(band_powers)
    
    # 转换为numpy数组
    results_array = np.array(results)
    
    # 保存结果
    try:
        np.save(output_file, results_array)
        print(f"结果已保存到 {output_file}")
    except Exception as e:
        print(f"错误: 保存文件时发生异常: {e}")


# def sol_2(input_filename, output_stats_filename):
#     fs = 250
#     win_length_sec = 30
#     step_samples = 10 * fs
#     step_sec = 10

#     try:
#         # 读取数据
#         data = np.load(input_filename)
#         print(f"成功读取数据。数组形状为: {data.shape}")
        
#         n_channels, n_samples = data.shape
        
#         # 计算滑窗参数
#         win_length_samples = win_length_sec * fs
#         step_samples = step_sec * fs
#         starts = np.arange(0, n_samples - win_length_samples + 1, step_samples)
#         n_windows = len(starts)
        
#         # 初始化结果数组
#         results = np.zeros((n_windows, n_channels, 4))
#         band_names = ['delta', 'theta', 'alpha', 'beta']
        
#         # 计算每个窗口的频段能量
#         print(f"开始计算频段能量，共{n_windows}个窗口...")
#         for i, start in enumerate(starts):
#             end = start + win_length_samples
#             window_data = data[:, start:end]
#             for ch in range(n_channels):
#                 f, Pxx = signal.welch(window_data[ch], fs, nperseg=win_length_samples)
#                 delta = Pxx[(f >= 0.5) & (f <= 4)].sum()
#                 theta = Pxx[(f >= 4) & (f <= 8)].sum()
#                 alpha = Pxx[(f >= 8) & (f <= 13)].sum()
#                 beta = Pxx[(f >= 13) & (f <= 30)].sum()
#                 results[i, ch] = [delta, theta, alpha, beta]
        
#         # 保存结果
#         print(results)
#         np.save(output_stats_filename, results)
#         print(f"频段能量结果已保存到 {output_stats_filename}")
#         print(f"结果形状: {results.shape} (窗口数, 通道数, 频段数)")
        
#     except Exception as e:
#         print(f"处理过程中发生错误: {e}")

def sol_3(input_filename, output_stats_filename):
    try:
        sampling_rate=250
        window_duration_sec=4
        overlap_percentage=50
        # 计算窗口和重叠对应的样本数
        window_size_samples = int(window_duration_sec * sampling_rate)
        overlap_samples = int(window_size_samples * (overlap_percentage / 100.0))
        
        print(f"采样率: {sampling_rate} Hz")
        print(f"Welch 窗口长度: {window_duration_sec} 秒 ({window_size_samples} 样本)")
        print(f"窗口重叠: {overlap_percentage}% ({overlap_samples} 样本)")
        
        # 读取数据
        data = np.load(input_filename)
        print(f"成功读取数据。数组形状为: {data.shape}")
        num_channels = data.shape[0]
        num_samples_total = data.shape[1]
        print(f"总样本数: {num_samples_total}")
        
        # 计算每个通道的PSD
        print("计算每个通道的 PSD (Welch 方法)...")
        all_channel_psd_list = []
        freqs = None  # 用于存储频率数组，所有通道相同
        
        for i in range(num_channels):
            channel_data = data[i, :]
            # nperseg 设置窗口长度
            # noverlap 设置重叠样本数
            current_freqs, current_psd = signal.welch(
                channel_data,
                fs=sampling_rate,
                nperseg=window_size_samples,
                noverlap=overlap_samples
            )
            all_channel_psd_list.append(current_psd)
            if freqs is None:
                freqs = current_freqs  # 只保存一次频率数组
        
        all_channel_psds = np.array(all_channel_psd_list)
        
        # 保存PSD结果
        np.save(output_stats_filename, all_channel_psds)
        print(f"PSD结果已保存到 {output_stats_filename}")
        print(f"PSD数组形状: {all_channel_psds.shape} (通道数, 频率点数)")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

def sol_4(input_filename, output_stats_filename):
    # --- 1. 读取数据 ---
    data = np.load(input_filename)
    print(f"成功读取数据。数组形状为: {data.shape}")

    fs = 250
    # --- 2. 计算每个通道的 相关性 ---
    n_channels = data.shape[0]
    coh_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            f, Cxy = coherence(data[i], data[j], fs=fs)
            mask = (f >= 8) & (f <= 12)
            avg_coh = np.mean(Cxy[mask]) if mask.any() else 0
            coh_matrix[i, j] = avg_coh
    print(coh_matrix)
    # --- 3. 保存结果 ---   
    np.save(output_stats_filename, coh_matrix)

def calculate_spectral_entropy(data_channel, sf):
    """
    计算单个数据通道的谱熵。
    使用Welch方法估计功率谱密度 (PSD)。

    参数:
        data_channel (np.ndarray): 单个通道的时间序列数据 (1D 数组)。
        sf (float): 采样频率 (Hz)。

    返回:
        float: 谱熵值。
    """
    n_samples = len(data_channel)
    # Welch方法的nperseg参数，默认为256，如果信号长度小于256，则使用信号长度
    nperseg = min(n_samples, 256)

    if n_samples == 0: # 处理空数据通道
        return np.nan
    if nperseg == 0 : # 如果信号长度为0，nperseg也会为0
        return np.nan # 或者根据需求返回0.0

    # 计算功率谱密度 (PSD)
    # freqs频率数组, psd功率谱密度数组
    freqs, psd = scipy.signal.welch(data_channel, fs=sf, nperseg=nperseg)

    # 检查PSD是否接近全零 (例如，对于恒定信号或零信号)
    if np.sum(psd) < 1e-10:
        return 0.0  # 谱熵为0，因为频谱中没有不确定性

    # 将PSD归一化，使其总和为1，形成概率分布
    psd_norm = psd / np.sum(psd)

    # 计算香农熵 (以2为底)
    # scipy.stats.entropy 会自动处理 pk=0 的情况 (0 * log(0) = 0)
    spectral_entropy_val = scipy.stats.entropy(psd_norm, base=2)

    return spectral_entropy_val

def sol_5(input_filename, output_stats_filename):
    # --- 参数设定 ---
    fs = 250  # 采样频率 (Hz)
    segment_duration_s = 10  # 数据段时长 (秒)
    entropy_m = 2  # 样本熵和近似熵的嵌入维度 (antropy中的order参数)
    # antropy 默认使用 r = 0.2 * std(data) 作为容差阈值 (Chebyshev距离)

    data = np.load(input_filename) 

    num_channels, total_samples = data.shape
    samples_per_segment = int(fs * segment_duration_s) # 每段的样本点数

    num_segments = total_samples // samples_per_segment # 计算可以完整分割出的段数

    # --- 初始化结果存储 ---
    # 每个熵值结果将是一个 (num_segments, num_channels) 的2D数组
    results_sample_entropy = np.zeros((num_segments, num_channels))
    results_approximate_entropy = np.zeros((num_segments, num_channels))
    results_spectral_entropy = np.zeros((num_segments, num_channels))

    # --- 逐段处理数据 ---
    for i in range(num_segments):
        start_idx = i * samples_per_segment
        end_idx = start_idx + samples_per_segment
        segment_data = data[:, start_idx:end_idx] # 获取当前数据段，形状为 (7, samples_per_segment)

        for j in range(num_channels): # 遍历每个通道
            channel_data = segment_data[j, :] # 获取当前通道的数据 (1D 数组)

            # 检查通道数据是否为常数，这可能导致熵计算出现问题 (例如std=0)
            if np.std(channel_data) < 1e-9:  # 有效地视为常数
                print(f"  通道 {j+1}: 数据为常数或接近常数。熵值可能为 NaN 或 0。")
                samp_en = np.nan
                ap_en = np.nan
                # 常数信号的谱熵为0 (频谱在0Hz处有一个尖峰)
                spec_en = 0.0 if np.all(channel_data == channel_data[0]) else calculate_spectral_entropy(channel_data, sf=fs)

            else:
                # 1. 计算样本熵 (Sample Entropy)
                    # antropy.sample_entropy 使用 order=m (嵌入维度)
                    # 默认使用 Chebyshev 距离和 r = 0.2 * std(data)
                samp_en = antropy.sample_entropy(channel_data, order=entropy_m)

                # 2. 计算近似熵 (Approximate Entropy)
                    # antropy.app_entropy 使用 order=m (嵌入维度)
                    # 默认使用 Chebyshev 距离和 r = 0.2 * std(data)
                ap_en = antropy.app_entropy(channel_data, order=entropy_m)

                # 3. 计算谱熵 (Spectral Entropy)
                spec_en = calculate_spectral_entropy(channel_data, sf=fs)

            
            results_sample_entropy[i, j] = samp_en
            results_approximate_entropy[i, j] = ap_en
            results_spectral_entropy[i, j] = spec_en
    stacked_array = np.stack([results_sample_entropy.T, results_approximate_entropy.T, results_spectral_entropy.T], axis=1) 
    output_filename = output_stats_filename
    np.save(output_filename, stacked_array) # 使用关键字参数保存，方便加载时按键名读取
    print(f"\n处理完成。结果已保存到 '{output_filename}'。")
    print("已保存熵值数组的形状:")
    print(f"  样本熵: {results_sample_entropy.shape}")
    print(f"  近似熵: {results_approximate_entropy.shape}")
    print(f"  谱熵: {results_spectral_entropy.shape}")
    



def sol_6(input_filename, output_results_filename):
    # --- 配置文件 ---
    # input_filename = "data/6_original.npy"
    # output_results_filename = "6_sol.npy"

    sampling_rate = 250  # Hz

    # Define Frequency Bands (Hz)
    alpha_low = 8
    alpha_high = 12
    beta_low = 13
    beta_high = 30

    # Filter Design Parameters
    # Filter order - affects sharpness of frequency cutoff vs length of filter kernel
    # A common rule of thumb is Fs or 2*Fs, ensuring it's odd for firwin
    filter_order = 2 * int(sampling_rate) # Example: 500 for 250Hz Fs
    if filter_order % 2 == 0: # Ensure filter order is odd for firwin
        filter_order += 1

    print(f"采样率: {sampling_rate} Hz")
    print(f"Alpha频段: {alpha_low}-{alpha_high} Hz")
    print(f"Beta频段: {beta_low}-{beta_high} Hz")
    print(f"滤波器阶数: {filter_order}")

    # --- 1. 读取数据 ---
    print(f"尝试从 '{input_filename}' 读取数据...")

    data = np.load(input_filename)

    num_channels = data.shape[0]
    num_samples_total = data.shape[1]

    # Check if data is long enough for filtering
    if num_samples_total < filter_order:
        print(f"Error: Total samples ({num_samples_total}) is less than filter order ({filter_order}). Cannot apply filter.")
        exit()

    print(f"总样本数: {num_samples_total}")
    print(f"通道数量: {num_channels}")

    # --- 2. 计算跨频段相关性 (Alpha振幅 vs Beta振幅) ---
    print("计算每个通道的 Alpha 振幅与 Beta 振幅之间的相关性...")

    # Initialize array to store results (one correlation value per channel)
    correlation_results = np.zeros((num_channels,))
    correlation_results.fill(np.nan) # Initialize with NaN to indicate channels where calculation failed

    # Design filters once
    nyq = 0.5 * sampling_rate
    b_alpha = signal.firwin(filter_order, [alpha_low/nyq, alpha_high/nyq], pass_zero=False)
    b_beta = signal.firwin(filter_order, [beta_low/nyq, beta_high/nyq], pass_zero=False)
    print("滤波器设计成功。")

    for channel_idx in range(num_channels):
        channel_data = data[channel_idx, :] 
        filtered_alpha = signal.filtfilt(b_alpha, [1], channel_data)
        filtered_beta = signal.filtfilt(b_beta, [1], channel_data)
        alpha_envelope = np.abs(signal.hilbert(filtered_alpha))
        beta_envelope = np.abs(signal.hilbert(filtered_beta))
        correlation_matrix = np.corrcoef(alpha_envelope, beta_envelope)
        correlation_coefficient = correlation_matrix[0, 1]
        correlation_results[channel_idx] = correlation_coefficient
        print(f"通道 {channel_idx}: Alpha-Beta 振幅相关性 = {correlation_coefficient:.4f}")

    # --- 3. 保存结果 ---
    print(f"\n保存计算结果到 '{output_results_filename}'...")
    np.save(output_results_filename, correlation_results)

def sol_7(input_filename, output_results_filename):
    # --- 1.读取数据 ---
    data = np.load(input_filename)

    # --- 2.计算ICA --- 
    samples = data.T
    ica = FastICA(n_components=data.shape[0], random_state=0)
    components = ica.fit_transform(samples).T
    print(components)

    # --- 3.存储结果 ---
    np.save(output_results_filename, components)

import numpy as np
from numpy.fft import fft

def sol_8(input_file, output_file):
    """
    计算EEG数据在指定频率下的SSVEP振幅
    
    参数:
    input_file (str): 输入数据文件路径 (.npy格式)
    output_file (str): 输出结果文件路径 (.npy格式)
    target_freq (float): 目标频率 (Hz), 默认为4Hz
    num_channels (int): 通道数，用于重塑结果，默认为7
    
    返回:
    numpy.ndarray: 形状为(num_channels, -1)的数组，包含各通道的指定频率振幅
    """
    target_freq=4
    num_channels=7
    # 加载数据
    try:
        data = np.load(input_file)
    except FileNotFoundError:
        print(f"错误: 文件 {input_file} 不存在")
        return None
    except Exception as e:
        print(f"错误: 加载文件时发生异常: {e}")
        return None
    
    # 获取采样参数
    fs = 250  # 采样率 (Hz)
    n_samples = data.shape[1]  # 每个通道的样本数
    
    # 计算FFT并提取指定频率的振幅
    try:
        fft_values = np.abs(fft(data, axis=1))
        freqs = np.fft.fftfreq(n_samples, 1/fs)
        target_bin = np.argmin(np.abs(freqs - target_freq))
        
        # 获取所有通道在指定频率的振幅
        ssvep_amplitudes = fft_values[:, target_bin]
        
        # 重塑为指定的通道数
        result = ssvep_amplitudes.reshape(num_channels, -1)
        
        # 保存结果
        np.save(output_file, result)
        print(output_file)
        print(f"SSVEP振幅结果已保存到 {output_file}")
        
        return result
    except Exception as e:
        print(f"错误: 处理数据时发生异常: {e}")
        return None
    
# def sol_8(input_filename, output_results_filename):
#     # --- 1.导入数据 ---
#     data = np.load(input_filename)

#     if len(data.shape) == 2:
#         data = np.mean(data, axis=0)

#     # --- 2.计算fft及其幅值 ---
#     fs = 250
#     n_points = len(data)

#     data = data - np.mean(data)
#     window = np.hanning(n_points)
#     data_windowed = data * window

#     yf = fft(data_windowed)
#     xf = fftfreq(n_points, 1/fs)

#     positive_freq = xf[:n_points//2]
#     amplitude = np.abs(yf[:n_points//2]) * 2 / n_points

#     target_freq = 4.0
#     idx = np.argmin(np.abs(positive_freq - target_freq))
#     amp_value = amplitude[idx]

#     # --- 3.存储结果并打印 ---
#     print(f"4Hz对应幅值: {amp_value:.4f}")

#     np.save(output_results_filename, amp_value)

def sol_10(input_filename, output_results_filename):
    maxlag=10
    significance_level=0.05
    try:
        # 打印配置参数
        print(f"格兰杰因果分析参数:")
        print(f"  最大滞后阶数 (maxlag): {maxlag}")
        print(f"  显著性水平: {significance_level}")
        
        # 读取数据
        print(f"尝试从 '{input_filename}' 读取数据...")
        data = np.load(input_filename)
        print(f"成功读取数据。数组形状为: {data.shape}")
        num_channels = data.shape[0]
        num_samples_total = data.shape[1]
        print(f"总样本数: {num_samples_total}")
        print(f"通道数量: {num_channels}")
        
        # 初始化p值矩阵
        p_values_matrix = np.full((num_channels, num_channels), np.nan)
        
        # 计算两两通道间的格兰杰因果关系
        print(f"\n计算两两通道间的格兰杰因果关系 (测试 X -> Y，即 X 是否引起 Y)，maxlag={maxlag}...")
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    continue  # 跳过自身比较
                    
                # 准备数据对：Y在前，X在后 (格式要求：[:, [Y, X]])
                data_pair = np.vstack([data[j, :], data[i, :]]).T  # Shape (num_samples, 2)
                
                # 执行格兰杰因果测试
                results = smt.grangercausalitytests(data_pair, maxlag=maxlag, verbose=False)
                
                # 获取p值 (使用第一个滞后阶数的F检验结果)
                p_value = results[1][0]["ssr_ftest"][1]  # 注意：索引1对应p值
                p_values_matrix[i, j] = p_value
                
                # 打印进度
                is_significant = p_value < significance_level
                significance_marker = "*" if is_significant else ""
                print(f"测试 通道 {i} -> 通道 {j} (maxlag={maxlag}): p 值 = {p_value:.4f} {significance_marker}")
        
        print("\n格兰杰因果关系计算完成。")
        
        # 打印显著因果关系摘要
        significant_pairs = np.argwhere((p_values_matrix < significance_level) & ~np.isnan(p_values_matrix))
        if len(significant_pairs) > 0:
            print(f"\n在显著性水平 {significance_level} 下，发现 {len(significant_pairs)} 个显著因果关系:")
            for pair in significant_pairs:
                i, j = pair
                print(f"  通道 {i} -> 通道 {j}: p 值 = {p_values_matrix[i, j]:.4f}")
        else:
            print(f"\n在显著性水平 {significance_level} 下，未发现显著因果关系。")
        
        # 保存结果
        np.save(output_results_filename, p_values_matrix)
        print(f"\np值矩阵已保存到 {output_results_filename}")
        print(f"结果矩阵形状: {p_values_matrix.shape}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

def sol_11(input_filename, output_results_filename):
    # --- 1.导入数据 ---
    data = np.load(input_filename)

    # --- 2.聚类计算微状态 ---
    gfp = np.std(data, axis=0)
    peak_indices = np.where(gfp > np.percentile(gfp, 85))[0]
    peaks = data[:, peak_indices].T
    kmeans = KMeans(n_clusters=4, random_state=42).fit(peaks)
    print(kmeans.cluster_centers_)
    # --- 3.保存以及可视化 ---
    np.save(output_results_filename, kmeans.cluster_centers_.T)