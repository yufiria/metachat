"""
符号计算工具模块

该文件实现了基于 SymPy 的符号数学工具。
智能体可以使用此工具进行符号计算、方程求解、微积分等数学操作。
"""

from typing import Dict, Any
import sympy as sp
from core.tools.base import BaseTool
from io import StringIO
import sys

class SymbolicSolver(BaseTool):
    """
    符号数学求解器工具类
    
    提供 SymPy 符号计算功能，用于执行符号数学操作。
    支持方程求解、符号微积分、代数简化等功能。
    """
    
    def __init__(self):
        """
        初始化符号求解器工具
        
        设置工具名称和描述，定义可用的模块和使用示例。
        """
        super().__init__(
            name="symbolic_solve",
            description="""Perform symbolic mathematics using SymPy.
            Available module: sympy (as sp)
            
            Input should be valid Python code as a string.
            Output will be the result of the symbolic manipulation.
            
            Example:
            Input: "x, y = sp.symbols('x y'); expr = x**2 + y; solved = sp.solve(expr - 10, x); print(solved)"
            """
        )
    
    async def execute(self, code: str) -> Dict[str, Any]:
        """
        执行 SymPy 符号计算代码
        
        在受控环境中执行 SymPy 代码，捕获打印输出和最后的计算结果。
        
        参数:
            code: str - 要执行的 SymPy 代码字符串
            
        返回:
            Dict[str, Any] - 包含以下键的字典：
                - success: bool - 执行是否成功
                - result: Any - 最后赋值的符号表达式或结果（成功时）
                - output: str - 捕获的打印输出（成功时）
                - error: str - 错误信息（失败时）
        """
        try:
            # 创建字符串缓冲区捕获打印输出
            output_buffer = StringIO()
            old_stdout = sys.stdout
            sys.stdout = output_buffer
            
            # 创建局部变量字典，提供 SymPy 模块
            locals_dict = {
                'sp': sp  # SymPy 符号计算库
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
                if not var_name.startswith('_') and var_name not in ['sp']:
                    result = value
            
            # 返回成功结果
            return {
                'success': True,
                'result': result,  # 符号计算结果
                'output': output  # 打印的输出
            }
            
        except Exception as e:
            # 捕获并返回错误
            return {
                'success': False,
                'error': str(e)
            }
