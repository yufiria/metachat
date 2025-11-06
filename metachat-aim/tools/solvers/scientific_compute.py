"""
科学计算工具模块

该文件实现了基于 NumPy 和 SciPy 的科学计算工具。
智能体可以使用此工具执行数值计算、优化、积分等科学计算任务。
"""

from typing import Dict, Any
import numpy as np
from scipy import constants, optimize, integrate
from core.tools.base import BaseTool, ToolCall
from io import StringIO
import sys

class ScientificCompute(BaseTool):
    """
    科学计算工具类
    
    提供 NumPy 和 SciPy 功能，用于执行科学计算代码。
    支持数值计算、物理常数、优化和积分等功能。
    """
    
    def __init__(self):
        """
        初始化科学计算工具
        
        设置工具名称和描述，定义可用的模块和使用示例。
        """
        super().__init__(
            name="scientific_compute",
            description="""Execute scientific calculations using numpy and scipy. 
            Available modules: numpy (as np), scipy.constants, scipy.optimize, scipy.integrate
            
            Input should be valid Python code as a string.
            Output will be the result of the calculation.
            
            Example:
            Input: "import numpy as np; wavelength = 500e-9; freq = constants.c / wavelength; print(f'{freq:.2e} Hz')"
            """
        )
    
    async def execute(self, code: str) -> Dict[str, Any]:
        """
        执行科学计算 Python 代码
        
        在受控环境中执行 Python 代码，提供 numpy 和 scipy 模块。
        捕获打印输出和最后赋值的变量。
        
        参数:
            code: str - 要执行的 Python 代码字符串
            
        返回:
            Dict[str, Any] - 包含以下键的字典：
                - success: bool - 执行是否成功
                - result: Any - 最后赋值的变量值（成功时）
                - output: str - 捕获的打印输出（成功时）
                - error: str - 错误信息（失败时）
        """
        try:
            # 创建字符串缓冲区捕获打印输出
            output_buffer = StringIO()
            old_stdout = sys.stdout
            sys.stdout = output_buffer
            
            # 创建局部变量字典，提供可用的模块
            locals_dict = {
                'np': np,  # NumPy 数值计算库
                'constants': constants,  # SciPy 物理常数
                'optimize': optimize,  # SciPy 优化模块
                'integrate': integrate  # SciPy 积分模块
            }
            
            try:
                # 执行代码
                exec(code, globals(), locals_dict)
                # 获取捕获的输出
                output = output_buffer.getvalue()
            finally:
                # 恢复标准输出
                sys.stdout = old_stdout
            
            # 返回最后赋值的变量（如果存在）
            result = None
            for var_name, value in locals_dict.items():
                # 跳过内部变量和预定义模块
                if not var_name.startswith('_') and var_name not in ['np', 'constants', 'optimize', 'integrate']:
                    result = value
            
            # 返回成功结果
            return {
                'success': True,
                'result': result,  # 最后的计算结果
                'output': output  # 打印的输出
            }
            
        except Exception as e:
            # 捕获并返回错误
            return {
                'success': False,
                'error': str(e)
            }
