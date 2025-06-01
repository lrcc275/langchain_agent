import numpy as np

# 加载数据
data = np.load('data/95_original.npy')

# 自动处理数据形状：确保通道在第一个维度 (7, N)
if data.ndim == 1:
    if data.size % 7 != 0:
        raise ValueError("数据无法重塑为7个通道")
    data = data.reshape(7, -1)
elif data.ndim == 2:
    if data.shape[0] != 7 and data.shape[1] == 7:
        data = data.T  # 转置使通道在第一维
    elif data.shape[0] != 7:
        # 尝试重塑
        if data.size % 7 == 0:
            data = data.reshape(7, -1)
        else:
            raise ValueError(f"数据形状{data.shape}不兼容7个通道")
else:
    # 高维数据展平
    data = data.ravel()
    if data.size % 7 == 0:
        data = data.reshape(7, -1)
    else:
        raise ValueError(f"数据无法重塑为7个通道")

# 计算每个通道的指标
peak_to_peak = np.ptp(data, axis=1)
means = np.mean(data, axis=1)
variances = np.var(data, axis=1)

# 组合成(7,3)数组
result = np.column_stack((peak_to_peak, means, variances))

# 打印结果
print("通道指标（峰峰值, 均值, 方差）:")
for i, (pp, mean, var) in enumerate(result):
    print(f"通道 {i+1}: {pp:.4f}, {mean:.4f}, {var:.4f}")

# 保存结果
np.save('result/1_95.npy', result)
print("结果已保存到 result/1_95.npy")
