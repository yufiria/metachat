"""
智能体基类模块

该文件定义了所有智能体的抽象基类和核心接口。
为不同类型的智能体（标准智能体、迭代智能体等）提供统一的基础架构。
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from core.models.base import BaseModel
from core.tools.base import BaseTool, ToolCall

class Agent(ABC):
    """
    智能体基类
    
    所有问题求解智能体的抽象基类。定义了与大语言模型交互、
    使用工具和解决问题的核心接口。
    """
    
    def __init__(self,
                 model: BaseModel,
                 tools: Optional[List[BaseTool]] = None,
                 system_prompt: str = ""):
        """
        初始化智能体
        
        参数:
            model: BaseModel - 底层大语言模型实例
            tools: Optional[List[BaseTool]] - 可用的工具列表，None表示无工具
            system_prompt: str - 系统提示词，定义智能体的角色和行为准则
        """
        self.model = model  # 用于生成响应的大语言模型
        self.tools = {tool.name: tool for tool in tools} if tools else {}  # 工具字典，通过名称索引
        self.system_prompt = system_prompt  # 系统级指令
        self.tool_calls: List[ToolCall] = []  # 记录所有工具调用历史

    @abstractmethod
    async def solve(self, problem: str) -> Dict[str, Any]:
        """
        解决问题（抽象方法）
        
        智能体的核心方法，子类必须实现具体的问题求解逻辑。
        
        参数:
            problem: str - 问题描述
            
        返回:
            Dict[str, Any] - 包含以下键值的字典：
                - solution: str - 问题的解答
                - metadata: Dict[str, Any] - 实现相关的元数据（如迭代次数等）
                - tool_calls: List[ToolCall] - 工具调用记录
        """
        pass

    async def _call_model(self, 
                         messages: List[Dict[str, str]], 
                         temperature: float = 1.0) -> str:
        """
        调用底层模型的辅助方法
        
        封装模型调用，统一处理不同返回类型。
        
        参数:
            messages: List[Dict[str, str]] - 对话消息列表
            temperature: float - 生成温度参数
            
        返回:
            str - 模型生成的文本内容
        """
        response = await self.model.generate(messages, temperature=temperature)
        
        # 处理字符串和LLMResponse两种返回类型
        if isinstance(response, str):
            return response
        return response.content

    async def _use_tool(self, 
                       tool_name: str, 
                       **kwargs) -> Any:
        """
        使用工具的辅助方法
        
        调用指定工具并记录使用历史。
        
        参数:
            tool_name: str - 工具名称
            **kwargs: 工具所需的参数
            
        返回:
            Any - 工具执行结果
            
        异常:
            ValueError - 当指定的工具不存在时抛出
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
            
        tool = self.tools[tool_name]
        result = await tool.run(**kwargs)
        self.tool_calls.append(result)  # 记录工具调用
        return result

    def _format_messages(self, 
                        problem: str, 
                        additional_context: Optional[str] = None) -> List[Dict[str, str]]:
        """
        格式化消息的辅助方法
        
        将问题和上下文格式化为模型可接受的消息列表。
        
        参数:
            problem: str - 问题描述
            additional_context: Optional[str] - 额外的上下文信息
            
        返回:
            List[Dict[str, str]] - 格式化的消息列表
        """
        # 构建基础消息列表
        messages = [
            {"role": "system", "content": self.system_prompt},  # 系统提示词
            {"role": "user", "content": problem}  # 用户问题
        ]
        
        # 如果有额外上下文，添加为助手响应
        if additional_context:
            messages.append({"role": "assistant", "content": additional_context})
            
        return messages