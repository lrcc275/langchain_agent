def generate_input(input_file_name1 : str, input_file_name2 : str, input_file_name3 : str, input_file_name4 : str, task_requirement : str, data_format : str):
    input_q = "(1)使用parse_eeg_data工具，解析同目录下的" + input_file_name1 + "；(2)编写并且运行python代码，对同目录下的" + input_file_name2 + "," + task_requirement + ", 将数值以" + data_format + "的格式存到同目录下的" + input_file_name3 + ", (3)将代码保存到" + input_file_name4
    return input_q

def task_requirements(i : int):
    if i == 1:
        return "计算每个通道的峰峰值、均值和方差，打印结果"
    if i == 2:
        return "将数据分30s为一段，使用滑窗模式，滑窗长度10s，计算每个通道的Delta、Theta、Alpha、Beta频段能量，打印结果"
    if i == 3:
        return "计算每个通道的PSD，使用Welch方法，窗口大小为4s，重叠50%，打印结果"
    if i == 4:
        return "计算所有通道对之间的相干性，频率范围为8-12Hz，打印结果"
    if i == 5:
        return "将数据分10s为一段，计算每个通道的样本熵、近似熵和谱熵，打印结果"
    if i == 6:
        return "计算中Alpha频段(8-12Hz)和Beta频段(13-30Hz)之间的跨频段相关性，打印结果，将数值以(7,1)的格式存到同目录下的result/6_{j+50}.npy，(3)将第二步中代码存到同目录下的文件result/6_{j+50}.py"
    if i == 7:
        return "将数据提取各通道独立成分"
    if i == 8:
        return "将数据提取稳态视觉诱发电位(SSVEP)在FFT上的幅值，频率范围为4Hz，打印结果"
    if i == 9:
        return "将数据使用动态因果建模(DCM)分析7通道间的因果关系，打印结果"
    if i == 10:
        return "对数据使用格兰杰因果分析计算两两通道间的因果关系，打印结果"
    if i == 11:
        return "将所有通道数据分析EEG微状态，打印结果"
    return ""

def data_formats(i):
    if i == 1:
        return "(7,3)"
    if i == 2:
        return "(7,4)"
    if i == 3:
        return "(7,x)[x取决于具体数据]"
    if i == 4:
        return "(7,7)"
    if i == 5:
        return "(7,3,x)[x取决于具体数据]"
    if i == 6:
        return "(7,1)"
    if i == 7:
        return "以(7,x)[x取决于具体数据]"
    if i == 8:
        return "(7,x)[x取决于具体数据]"
    if i == 9:
        return "(1)使用parse_eeg_data工具，解析同目录下的data/{j+80}.txt；(2)编写并且运行python代码，对同目录下的data/{j+80}_original.npy的数据使用动态因果建模(DCM)分析7通道间的因果关系，打印结果并可视化，将数值存到同目录下的result/9_{j+80}.npy，将图片存到同目录下的result/9_{j+80}.jpg，(3)将第二步中代码存到同目录下的文件result/9_{j+80}.py"
    if i == 10:
        return "(7,7)"
    if i == 11:
        return "(7,4)"
    return ""
    
# print(generate_input("1.txt","1.txt","1.txt","1.txt","1.txt","1.txt"))
# print(task_requirements(8))
