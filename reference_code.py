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
    
def sol_2(input_filename, output_stats_filename):
    fs = 250
    win_length_sec = 30
    step_samples = 10 * fs
    step_sec = 10

    try:
        # 读取数据
        data = np.load(input_filename)
        print(f"成功读取数据。数组形状为: {data.shape}")
        
        n_channels, n_samples = data.shape
        
        # 计算滑窗参数
        win_length_samples = win_length_sec * fs
        step_samples = step_sec * fs
        starts = np.arange(0, n_samples - win_length_samples + 1, step_samples)
        n_windows = len(starts)
        
        # 初始化结果数组
        results = np.zeros((n_windows, n_channels, 4))
        band_names = ['delta', 'theta', 'alpha', 'beta']
        
        # 计算每个窗口的频段能量
        print(f"开始计算频段能量，共{n_windows}个窗口...")
        for i, start in enumerate(starts):
            end = start + win_length_samples
            window_data = data[:, start:end]
            for ch in range(n_channels):
                f, Pxx = signal.welch(window_data[ch], fs, nperseg=win_length_samples)
                delta = Pxx[(f >= 0.5) & (f <= 4)].sum()
                theta = Pxx[(f >= 4) & (f <= 8)].sum()
                alpha = Pxx[(f >= 8) & (f <= 13)].sum()
                beta = Pxx[(f >= 13) & (f <= 30)].sum()
                results[i, ch] = [delta, theta, alpha, beta]
        
        # 保存结果
        np.save(output_stats_filename, results)
        print(f"频段能量结果已保存到 {output_stats_filename}")
        print(f"结果形状: {results.shape} (窗口数, 通道数, 频段数)")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

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

    # --- 3. 保存结果 ---   
    np.save(output_stats_filename, coh_matrix)

def sol_5(input_filename, output_stats_filename, fs=250, segment_length_sec=10, m=2):
    try:
        # 加载数据
        data = np.load(input_filename)
        print(f"成功加载数据: {data.shape} (通道数, 样本数)")
        
        # 数据分段处理
        segment_length = segment_length_sec * fs
        n_channels, n_samples = data.shape
        n_segments = n_samples // segment_length
        data_truncated = data[:, :n_segments*segment_length]
        segments = data_truncated.reshape(n_channels, n_segments, segment_length)
        
        print(f"数据分段: 每段{segment_length_sec}秒({segment_length}样本)，共{n_segments}段")
        
        # 初始化结果数组
        entropy_results = np.zeros((n_channels, n_segments, 3))
        
        # 计算三种熵
        print("开始计算熵值特征...")
        for ch in range(n_channels):
            for seg in range(n_segments):
                seg_data = segments[ch, seg]
                
                # 计算样本熵
                samp_ent = sample_entropy(seg_data, m=m)
                
                # 计算近似熵
                app_ent = approx_entropy(seg_data, m=m)
                
                # 计算谱熵
                freqs, psd = signal.periodogram(seg_data, fs=fs)
                psd_norm = psd / psd.sum()
                psd_norm[psd_norm == 0] = 1e-12  # 避免log(0)
                spec_ent = -np.sum(psd_norm * np.log2(psd_norm))
                
                # 保存结果
                entropy_results[ch, seg] = [samp_ent, app_ent, spec_ent]
        
        # 保存结果
        np.save(output_stats_filename, entropy_results)
        print(f"熵值计算结果已保存至: {output_stats_filename}")
        
        # 打印结果摘要
        print("\n熵值结果摘要:")
        for ch in range(min(3, n_channels)):  # 仅打印前3个通道作为示例
            print(f"通道 {ch+1}:")
            for s in range(min(3, n_segments)):  # 仅打印前3个分段作为示例
                res = entropy_results[ch, s]
                print(f"  分段 {s+1}: 样本熵={res[0]:.3f}, 近似熵={res[1]:.3f}, 谱熵={res[2]:.3f}")
        if n_channels > 3 or n_segments > 3:
            print("... 更多结果已保存到文件中")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

def sample_entropy(data, m=2, r=None):
    """计算样本熵"""
    N = len(data)
    if r is None:
        r = 0.2 * np.std(data)
    if N <= m + 1:
        return 0
    x = np.array([data[i:i+m] for i in range(N - m)])
    y = np.array([data[i:i+m+1] for i in range(N - m)])
    C_m = 0
    C_m1 = 0
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                if np.max(np.abs(x[i] - x[j])) <= r:
                    C_m += 1
                if np.max(np.abs(y[i] - y[j])) <= r:
                    C_m1 += 1
    B = C_m / ((N - m) * (N - m - 1)) if (N - m) * (N - m - 1) > 0 else 0
    A = C_m1 / ((N - m) * (N - m - 1)) if (N - m) * (N - m - 1) > 0 else 0
    return -np.log(A / B) if A !=0 and B !=0 else 0

def approx_entropy(data, m=2, r=None):
    """计算近似熵"""
    N = len(data)
    if r is None:
        r = 0.2 * np.std(data)
    
    def _phi(m):
        x = np.array([data[i:i+m] for i in range(N - m + 1)])
        C = np.zeros(len(x))
        for i in range(len(x)):
            dist = np.max(np.abs(x[i] - x), axis=1)
            C[i] = np.sum(dist <= r) - 1  # 减去自身匹配
        C = C / (N - m)
        return np.mean(np.log(C + 1e-12))  # 添加小常数避免log(0)
    
    return _phi(m) - _phi(m+1)



def sol_6(input_filename, output_stats_filename):
    # --- 配置文件 ---
    input_filename = "data/6_original.npy"
    output_results_filename = "6_sol.npy"

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

def sol_8(input_filename, output_results_filename):
    # --- 1.导入数据 ---
    data = np.load(input_filename)

    if len(data.shape) == 2:
        data = np.mean(data, axis=0)

    # --- 2.计算fft及其幅值 ---
    fs = 250
    n_points = len(data)

    data = data - np.mean(data)
    window = np.hanning(n_points)
    data_windowed = data * window

    yf = fft(data_windowed)
    xf = fftfreq(n_points, 1/fs)

    positive_freq = xf[:n_points//2]
    amplitude = np.abs(yf[:n_points//2]) * 2 / n_points

    target_freq = 4.0
    idx = np.argmin(np.abs(positive_freq - target_freq))
    amp_value = amplitude[idx]

    # --- 3.存储结果并打印 ---
    print(f"4Hz对应幅值: {amp_value:.4f}")

    np.save(output_results_filename, amp_value)

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

    # --- 3.保存以及可视化 ---
    np.save(output_results_filename, kmeans.cluster_centers_)