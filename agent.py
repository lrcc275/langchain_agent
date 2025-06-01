from langchain.agents import AgentExecutor, create_react_agent
import os
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import Tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool, PythonREPLTool
from langchain.agents import tool
from langchain_core.tools import tool as core_tool
import numpy as np
from contextlib import redirect_stdout
import time
@core_tool
def parse_eeg_data(filepath):
    """
    从 .txt 文件中解析 EEG 数据。

    Args:
        filepath (str): .txt 文件的路径。

    Returns:
        将结果保存到对应的路径
    """
    # 用于存储从所有行收集到的所有数据点（扁平化列表）
    all_data_points_flat = []
    
    # 每行期望的数据值数量（不包括时间戳和包号）
    channel_num=8
    pkg_groups_per_line=5
    expected_data_values_per_line = pkg_groups_per_line * channel_num
    base_name, _ = os.path.splitext(filepath)
    output_path = f"{base_name}_original.npy"
    try:
        with open(filepath, 'r') as f:
            for line_num, line_content in enumerate(f, 1):
                line_content = line_content.strip()
                if not line_content:  # 跳过空行
                    continue

                parts = line_content.split(',')
                if parts[0][0] != '1' and parts[0][0] != '2' and parts[0][0] != '0':
                    continue
                # 至少需要 时间戳(1) + 包号(1) + 数据点(expected_data_values_per_line)
                if len(parts) < 2 + expected_data_values_per_line:
                    print(f"警告: 第 {line_num} 行数据太短。期望至少 "
                          f"{2 + expected_data_values_per_line} 个字段, 实际得到 {len(parts)}。已跳过此行。")
                    continue

                # 提取数据部分：从第3个元素开始（索引为2），取 expected_data_values_per_line 个值
                data_strings = parts[2 : 2 + expected_data_values_per_line]
                
                try:
                    # 将字符串数据转换为浮点数
                    data_floats = [float(val) for val in data_strings]
                    all_data_points_flat.extend(data_floats)
                except ValueError as e:
                    print(f"警告: 第 {line_num} 行在期望数字的位置包含非数字数据: {e}。已跳过此行的数据。")
                    # 如果任何数据点转换失败，则跳过此行对 all_data_points_flat 的贡献
                    continue
        
        if not all_data_points_flat:
            print("未能成功解析任何数据点。")
            return None

        # 将扁平化的数据点列表转换为 NumPy 数组
        raw_array = np.array(all_data_points_flat)

        # 数据是按 CH1_samp1, CH2_samp1, ..., CH8_samp1, CH1_samp2, ... 的顺序排列的
        # 总数据点数必须是 channel_num 的倍数
        num_total_data_points = len(raw_array)
        if num_total_data_points % channel_num != 0:
            print(f"警告: 总数据点数 ({num_total_data_points}) "
                  f"不能被通道数 ({channel_num}) 整除。数据可能已损坏或不完整。")
            # 截断到 channel_num 的最大倍数
            num_total_data_points = (num_total_data_points // channel_num) * channel_num
            raw_array = raw_array[:num_total_data_points]
            if not raw_array.size:
                 print("截断后无可用数据。")
                 return None
        
        # 将数组重塑为 (N, channel_num)，其中 N 是总的采样时刻数
        # 重塑后的数组的每一行是 [CH1_sample, CH2_sample, ..., CH8_sample]
        reshaped_array = raw_array.reshape(-1, channel_num)

        # 转置以获得 (channel_num, N) 的形状
        output_array = reshaped_array.T
        output_array = output_array[:7]
        np.save(output_path, output_array)
        return "你已经完成了第一步，请继续进行下一步！如果你不知道下一步做什么，请多关注我的输入"

    except FileNotFoundError:
        return f"错误: 文件未找到 {filepath}，你可能要考虑输出文件名用双引号"
    except Exception as e:
        print(f"发生意外错误: {e}")
        return


openai_api_key = "sk-7e37bea1ecc249968d28f386eb60b09e"  #deepseek key
# openai_api_key = "sk-84aad3057fbc462399a036c77c213a78"  #ali key
openai_api_base = "https://api.deepseek.com/v1"
# openai_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    model = "deepseek-chat",
    # model = "qvq-max",
    # model = "qwq-32b",
    temperature=0.7,
    stop=['Observation:', 'Observation:\n']
)

prompt = hub.pull("hwchase17/react")

# 创建PythonREPLTool实例
python_repl_tool = PythonREPLTool()

@tool
def code_interpreter(code):
    '''code_interpreter: Use this to execute python commands. Input should be a valid python command with markdown stytle ```python xxx ```. If you want to see the output of a value, you should print it out with `print(...)`,you should print it out with `print(...)'''
    codeStr = code.replace("```python","")
    codeStr = code.replace("```","")
    run_ret = python_repl_tool.run(codeStr)
    return run_ret

tools = [parse_eeg_data, python_repl_tool]

def run(input_q : str, log_name : str) -> float :
    start_time = time.time()
    # print(log_name) 
    with open(log_name, 'w') as f:
            # 构建React智能体
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
            # 解析错误解决方案：添加参数handle_parsing_errors=True
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False,return_intermediate_steps=True,handle_parsing_errors=True,max_iterations=5)
        # print(input_q)
        with redirect_stdout(f): 
            try:
                iteration = 0
                for step in agent_executor.stream({"input": input_q}):
                    print('*' * 100)
                    print("round " + str(iteration))
                    print('*' * 100)
                    iteration += 1
                    thought = step["messages"][0].content.split("\nAction")
                    for k in range(len(thought)):
                        print(thought[k])
            except:
                print("over")
    print("task " + input_q + " is over!")
    end_time = time.time()
    return (end_time - start_time)