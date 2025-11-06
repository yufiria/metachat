"""
OpenAI 模型封装模块

该文件实现了 OpenAI GPT 系列模型的接口封装，支持 GPT-4、GPT-3.5 等模型。
通过异步API调用实现高效的对话生成。
"""

from typing import List, Dict, Optional
import openai
from openai import AsyncOpenAI
from .base import BaseModel, LLMResponse

class OpenAIModel(BaseModel):
    """
    OpenAI 模型包装类
    
    封装 OpenAI 的 GPT 系列模型，提供统一的异步生成接口。
    支持 GPT-4, GPT-3.5-turbo 等各种 OpenAI 模型。
    """
    def __init__(self, model_name: str, api_key: str):
        """
        初始化 OpenAI 模型
        
        参数:
            model_name: str - OpenAI 模型名称（如 "gpt-4", "gpt-3.5-turbo"）
            api_key: str - OpenAI API 密钥
        """
        super().__init__(model_name)
        self.client = AsyncOpenAI(api_key=api_key)  # 创建异步OpenAI客户端
    
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        """
        生成模型响应
        
        使用 OpenAI Chat Completions API 生成对话响应。
        
        参数:
            messages: List[Dict[str, str]] - 对话消息列表，包含 role 和 content
            temperature: float - 生成温度，控制输出随机性（0.0-2.0）
            max_tokens: Optional[int] - 最大生成token数
            
        返回:
            LLMResponse - 包含生成文本和原始响应的对象
        """
        # 调用 OpenAI Chat Completions API
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 封装响应为标准格式
        return LLMResponse(
            content=response.choices[0].message.content,  # 提取生成的文本内容
            raw_response=response  # 保存原始响应用于调试
        )
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的 token 数量
        
        参数:
            text: str - 需要计算token的文本
            
        返回:
            int - token 数量
            
        注意:
            当前未实现精确计数，待完善
        """
        # TODO: 实现基于 tiktoken 的精确 token 计数逻辑
        pass