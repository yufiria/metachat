"""
Anthropic Claude 模型封装模块

该文件实现了 Anthropic Claude 系列模型的接口封装。
支持 Claude 3 Opus、Sonnet、Haiku 及 3.5 版本等模型。
"""

from typing import List, Dict, Optional
from anthropic import AsyncAnthropic
from .base import BaseModel, LLMResponse

class AnthropicModel(BaseModel):
    """
    Anthropic Claude 模型包装类
    
    封装 Anthropic 的 Claude 系列模型，提供统一的异步生成接口。
    自动处理不同 Claude 版本的最大 token 限制。
    """
    # 各个模型的最大输出token数限制
    MODEL_MAX_TOKENS = {
        "claude-3-opus": 4096,      # Claude 3 Opus 最大输出4096 tokens
        "claude-3-sonnet": 4096,    # Claude 3 Sonnet 最大输出4096 tokens
        "claude-3-haiku": 4096,     # Claude 3 Haiku 最大输出4096 tokens
        "claude-3-5-sonnet": 8192,  # Claude 3.5 Sonnet 最大输出8192 tokens
        "claude-3-5-haiku": 8192    # Claude 3.5 Haiku 最大输出8192 tokens
    }
    
    # 各个模型的上下文窗口大小（用于参考）
    MODEL_CONTEXT_WINDOWS = {
        "claude-3-opus": 200000,      # Claude 3 Opus 支持200k上下文
        "claude-3-sonnet": 200000,    # Claude 3 Sonnet 支持200k上下文
        "claude-3-haiku": 200000,     # Claude 3 Haiku 支持200k上下文
        "claude-3-5-sonnet": 200000,  # Claude 3.5 Sonnet 支持200k上下文
        "claude-3-5-haiku": 200000    # Claude 3.5 Haiku 支持200k上下文
    }
    
    def __init__(self, model_name: str, api_key: str):
        """
        初始化 Anthropic 模型
        
        参数:
            model_name: str - Claude 模型名称（如 "claude-3-opus-20240229"）
            api_key: str - Anthropic API 密钥
        """
        super().__init__(model_name)
        self.client = AsyncAnthropic(api_key=api_key)  # 创建异步Anthropic客户端
        
        # 查找匹配的模型前缀以确定输出token限制
        # 由于模型名称包含日期后缀，需要通过前缀匹配
        matching_model = next(
            (model for model in self.MODEL_MAX_TOKENS.keys() 
             if model_name.startswith(model)), 
            None
        )
        
        # 获取模型的最大输出token数，如果未找到匹配则默认使用4096
        self.max_tokens = self.MODEL_MAX_TOKENS.get(matching_model, 4096)
    
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        """
        生成模型响应
        
        使用 Anthropic Messages API 生成对话响应。
        自动处理系统提示词格式转换。
        
        参数:
            messages: List[Dict[str, str]] - 对话消息列表
            temperature: float - 生成温度，控制输出随机性（0.0-1.0）
            max_tokens: Optional[int] - 最大生成token数
            
        返回:
            LLMResponse - 包含生成文本、原始响应和token使用统计的对象
        """
        # 将消息转换为 Anthropic 格式
        # Anthropic 的系统提示词需要特殊处理
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Anthropic 将系统提示词作为第一条用户消息的一部分
                system_prompt = msg["content"]
            else:
                formatted_messages.append({
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"]
                })
        
        # 如果存在系统提示词，添加到第一条用户消息前
        if formatted_messages and "system_prompt" in locals():
            formatted_messages[0]["content"] = f"{system_prompt}\n\n{formatted_messages[0]['content']}"
        
        # 构建 API 调用参数
        # 如果未指定 max_tokens，使用模型的最大值
        params = {
            "model": self.model_name,
            "messages": formatted_messages,
            "max_tokens": min(max_tokens, self.max_tokens) if max_tokens else self.max_tokens,
            "temperature": temperature,
            "stream": False  # 不使用流式响应
        }
        
        # 调用 Anthropic Messages API
        response = await self.client.messages.create(**params)
        
        # 封装响应，包含 token 使用统计
        return LLMResponse(
            content=response.content[0].text,  # 提取生成的文本内容
            raw_response=response,  # 保存原始响应
            input_tokens=response.usage.input_tokens,  # 输入token数
            output_tokens=response.usage.output_tokens  # 输出token数
        )
    
    def count_tokens(self, text: str) -> int:
        """
        估算文本的 token 数量
        
        参数:
            text: str - 需要计算token的文本
            
        返回:
            int - 估算的 token 数量
            
        注意:
            使用粗略估算，精确值可从API响应中获取
        """
        # 粗略估算：每个单词约1.3个token
        return len(text.split()) * 1.3
