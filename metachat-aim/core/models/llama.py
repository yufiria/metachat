"""
LLaMA 模型封装模块

该文件实现了 Meta LLaMA 系列模型的接口封装。
通过 Together AI 平台提供的 API 访问 LLaMA 3.1 等模型。
"""

from typing import List, Dict, Optional
from together import Together
from .base import BaseModel, LLMResponse

class LlamaModel(BaseModel):
    """
    LLaMA 模型包装类
    
    封装通过 Together AI 托管的 Meta LLaMA 系列模型。
    支持 LLaMA 3.1 的各种规模版本（8B、70B等）。
    """
    
    # 各个模型的最大输出token数限制
    MODEL_MAX_TOKENS = {
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 2048,   # LLaMA 3.1 8B 快速版本
        "meta-llama/Meta-Llama-3.1-70B-Instruct": 2048,        # LLaMA 3.1 70B 指令版本
    }
    
    def __init__(self, model_name: str, api_key: str):
        """
        初始化 LLaMA 模型
        
        参数:
            model_name: str - LLaMA 模型名称（如 "meta-llama/Meta-Llama-3.1-70B-Instruct"）
            api_key: str - Together AI API 密钥
        """
        super().__init__(model_name)
        self.client = Together()  # 创建Together AI客户端
        # self.max_tokens = self.MODEL_MAX_TOKENS.get(model_name, 2048)
    
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        """
        生成模型响应
        
        使用 Together AI Chat Completions API 生成对话响应。
        
        参数:
            messages: List[Dict[str, str]] - 对话消息列表
            temperature: float - 生成温度，控制输出随机性（0.0-1.0）
            max_tokens: Optional[int] - 最大生成token数
            
        返回:
            LLMResponse - 包含生成文本、原始响应和token使用统计的对象
        """
        
        # 创建 Together AI API 请求
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            # max_tokens=min(max_tokens, self.max_tokens) if max_tokens else self.max_tokens,
            stream=False  # 不使用流式响应
        )

        # API请求示例格式：
        # api_request = {
        #     'model': 'llama3.3-70b', 
        #     'messages': [
        #         {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        #         {'role': 'user', 'content': 'What is 2+2?'}
        #     ], 
        #     'temperature': 0.0, 
        #     'stream': False
        # }
        
        # 提取响应内容
        content = response.choices[0].message.content
        
        # 封装响应，包含token使用统计（如果可用）
        return LLMResponse(
            content=content,
            raw_response=response,
            input_tokens=response.usage.prompt_tokens if hasattr(response, 'usage') else None,
            output_tokens=response.usage.completion_tokens if hasattr(response, 'usage') else None
        )
    
    def count_tokens(self, text: str) -> int:
        """
        估算文本的 token 数量
        
        参数:
            text: str - 需要计算token的文本
            
        返回:
            int - 估算的 token 数量
            
        注意:
            使用粗略估算方法
        """
        # 粗略估算：每个单词约1.3个token
        return len(text.split()) * 1.3