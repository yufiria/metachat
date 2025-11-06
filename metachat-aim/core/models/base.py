"""
基础模型模块

该文件定义了所有大语言模型(LLM)的基础抽象类和响应数据结构。
为 OpenAI、Anthropic、LLaMA 等不同的 LLM 提供统一的接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMResponse:
    """
    大语言模型响应类
    
    封装模型生成的响应及相关元数据，包括生成的内容、原始响应和token使用情况。
    """
    def __init__(self, content: str, raw_response: Any, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None):
        """
        初始化LLM响应对象
        
        参数:
            content: str - 模型生成的文本内容
            raw_response: Any - 原始的API响应对象
            input_tokens: Optional[int] - 输入消耗的token数量
            output_tokens: Optional[int] - 输出生成的token数量
        """
        self.content = content  # 生成的文本内容
        self.raw_response = raw_response  # 原始响应对象，用于调试和详细分析
        self.input_tokens = input_tokens  # 输入token计数，用于成本追踪
        self.output_tokens = output_tokens  # 输出token计数，用于成本追踪

class BaseModel(ABC):
    """
    大语言模型基类
    
    所有LLM实现的抽象基类，定义了生成响应和计算token的标准接口。
    子类需要实现具体的模型调用逻辑。
    """
    def __init__(self, model_name: str, **kwargs):
        """
        初始化基础模型
        
        参数:
            model_name: str - 模型名称标识符（如 "gpt-4", "claude-3"等）
            **kwargs: 其他模型特定的配置参数
        """
        self.model_name = model_name  # 模型标识符
        self.kwargs = kwargs  # 额外的模型配置参数
    
    @abstractmethod
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        """
        生成模型响应（抽象方法）
        
        根据输入的对话消息生成响应。子类必须实现此方法。
        
        参数:
            messages: List[Dict[str, str]] - 对话消息列表，每条消息包含 role 和 content
            temperature: float - 生成温度，控制随机性（0.0-1.0），默认0.0表示确定性输出
            max_tokens: Optional[int] - 最大生成token数，None表示使用模型默认值
            
        返回:
            LLMResponse - 包含生成内容和元数据的响应对象
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量（抽象方法）
        
        用于估算API调用成本和管理上下文长度。子类必须实现此方法。
        
        参数:
            text: str - 需要计算token数量的文本
            
        返回:
            int - 文本对应的token数量
        """
        pass