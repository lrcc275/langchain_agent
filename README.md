# langchain_agent
## 1.文件总览  
- **data**：文件夹，存放96个初始.txt  
- **log**：存放所有任务运行时产生的log  
- **result**：存放所有任务运行时产生的结果文件  
- **agent.py**：agent定义，提供`run`函数作为接口，函数参数：`input_q`、`log_name`  
- **evaluation.py**：评估模块，提供`check`等多个函数作为接口，`check`函数参数：`file_path1`（原始数据对应的.npy地址）、`file_path2`（本次测试生成的结果文件.npy）、`percent`（允许误差比）、`task`（任务编号）  
- **logg.py**：暂无实现  
- **main.ipynb**：调用`agent.py`、`prompt.py`和`evaluation.py`  
- **prompt.py**：根据任务编号、输入文件名，组合生成input  
- **reference_code.py**：集合所有参考函数  
- **testdata.py**：暂无实现  

## 2.使用流程  
通过`main.ipynb`进行交互。