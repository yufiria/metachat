"""
标准智能体模块

该文件实现了单次调用智能体，通过一次模型调用解决问题。
相比迭代智能体更快但功能有限，适合简单的问题求解任务。
"""

from typing import Dict, Any, List, Optional
from .base import Agent
from tools.design.api import NeuralDesignAPI
from datetime import datetime

class StandardAgent(Agent):
    """
    标准单次调用智能体
    
    通过一次大语言模型调用解决问题的简单智能体。
    支持工具调用但不支持迭代推理和内部独白。
    """
    
    def __init__(self, *args, **kwargs):
        """
        初始化标准智能体
        
        参数:
            *args: 传递给基类的位置参数
            **kwargs: 传递给基类的关键字参数
        """
        super().__init__(*args, **kwargs)
        
        # 添加神经网络设计工具
        self.tools['neural_design'] = NeuralDesignAPI()
        
        # 如果未提供系统提示词，使用默认提示词
        if not self.system_prompt:
            self.system_prompt = """You are an expert in optics and photonics with access to neural network-based metalens and superpixel design APIs. 

Think step by step. Break down complex problems into steps and plan your approach before solving.

If you need to design a metalens or superpixel, use these neural network tools:
   - For metalenses, use: <tool>neural_design
   design_metalens(refractive_index, lens_diameter [m], focal_length [m], thickness [m], operating_wavelength [m])
   </tool>
   - For superpixels, use: <tool>neural_design
   design_superpixel(refractive_index, length [m], incident_angle [deg], diffraction_angle [deg], thickness [m], operating_wavelength [m])
   </tool>

Example responses:

For neural network design:
<tool>neural_design
design_metalens(refractive_index=2.7, lens_diameter=100e-6, focal_length=200e-6, thickness=500e-9, operating_wavelength=800e-9)
</tool>

For a direct answer:
Answer: 1.55 μm"""

    def _format_messages(self, problem: str) -> List[Dict[str, str]]:
        """
        格式化消息列表
        
        参数:
            problem: str - 问题描述
            
        返回:
            List[Dict[str, str]] - 格式化的消息列表
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem}
        ]
    
    async def solve(self, problem: str, problem_id: Optional[str] = None, temperature: float = 1.0, disable_cache: bool = False) -> Dict[str, Any]:
        """
        解决问题（单次调用）
        
        通过一次模型调用解决问题，然后处理响应中的工具调用。
        
        参数:
            problem: str - 问题描述
            problem_id: Optional[str] - 问题ID（保留以保持接口一致）
            temperature: float - 生成温度
            disable_cache: bool - 是否禁用缓存（通过添加时间戳）
            
        返回:
            Dict[str, Any] - 包含解决方案和元数据的字典
        """
        # 如果禁用缓存，添加时间戳
        if disable_cache:
            timestamp = datetime.now().isoformat()
            problem = f"Date submitted: {timestamp}\n\n{problem}"
            
        # 格式化消息
        messages = self._format_messages(problem)
        
        # 单次模型调用
        solution = await self._call_model(
            messages,
            temperature=temperature
        )
        
        # 处理解决方案中的工具调用
        processed_solution = solution
        
        # 循环处理所有 neural_design 工具调用
        while "<tool>neural_design" in processed_solution:
            # 提取工具标签之间的代码
            start = processed_solution.find("<tool>neural_design")
            end = processed_solution.find("</tool>", start)
            if end == -1:
                break  # 没有找到结束标签，退出循环
                
            # 获取代码块
            code_block = processed_solution[start + len("<tool>neural_design"):end].strip()
            
            # 执行代码
            result = await self.tools['neural_design'].execute(code_block)
            
            # 将工具调用替换为结果
            if result.get('success', False):
                # 处理API的字符串返回值
                api_call = result.get('result', '')
                output = f"API Call that would be made:\n{api_call}"
                replacement = f"```python\n{code_block}\n```\nOutput:\n```\n{output}\n```"
            else:
                # 处理错误情况
                replacement = f"Error executing code: {result.get('error', 'Unknown error')}"
            
            # 替换原文本
            processed_solution = (
                processed_solution[:start] +
                replacement +
                processed_solution[end + len("</tool>"):]
            )
        
        # 返回结果
        return {
            "solution": processed_solution,  # 处理后的解决方案
            "metadata": {
                "method": "one-shot",  # 使用单次调用方法
                "num_model_calls": 1  # 模型调用次数
            },
            "tool_calls": self.tool_calls  # 工具调用记录
        }