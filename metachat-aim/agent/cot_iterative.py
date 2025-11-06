"""
迭代智能体模块（链式思维 - Chain of Thought）

该文件实现了支持迭代推理的智能体，使用内部独白(Internal Monologue)机制。
智能体可以进行多轮自我对话，逐步分解和解决复杂的光学设计问题。
这是 AIM (Agentic Iterative Monologue) 框架的核心实现。
"""

from typing import Dict, Any, List, Optional
from .base import Agent
from tools.design.api import NeuralDesignAPI
from pathlib import Path
import json
from datetime import datetime
import uuid

class IterativeAgent(Agent):
    """
    迭代智能体类
    
    实现 AIM (Agentic Iterative Monologue) 框架的核心智能体。
    支持内部独白、工具调用和多轮迭代推理来解决复杂的光学设计问题。
    
    特点:
    - 支持内部思考（未标记的文本）
    - 支持工具调用（<tool>标签）
    - 支持最终响应（<response>标签）
    - 自动记录完整的对话日志
    """
    
    def __init__(self, *args, **kwargs):
        """
        初始化迭代智能体
        
        参数:
            *args: 传递给基类的位置参数
            **kwargs: 关键字参数，包括：
                - log_dir: str - 日志目录路径
                - debug: bool - 是否开启调试模式
        """
        super().__init__(*args, **kwargs)
        
        # 添加神经网络设计工具
        # 用于调用 FiLM WaveY-Net 进行超表面和金属透镜设计
        self.tools['neural_design'] = NeuralDesignAPI()
        
        # 配置日志系统
        # 每个模型的日志保存在独立目录中
        self.log_dir = Path(kwargs.get('log_dir', "experiments/logs/eval_v1_corrected_noneq/cot_iterative/self_chat")) / self.model.model_name
        self.log_dir.mkdir(parents=True, exist_ok=True)  # 创建日志目录
        self.debug = kwargs.get('debug', False)  # 调试模式标志
        
        # 如果未提供系统提示词，使用默认的光学专家提示词
        if not self.system_prompt:
            self.system_prompt = """You are an expert in optics and photonics engaging in a continuous conversation to help users with their optics and photonics questions. 
You have access to neural network-based design APIs. You can iteratively talk to yourself and have an internal monologue.

Guidelines:
0. Think step by step. Break down complex problems into steps and plan your approach before solving.
1. If you need to design a metalens or superpixel, use these neural network tools:
   - For metalenses, use: <tool>neural_design
   design_metalens(refractive_index, lens_diameter [m], focal_length [m], thickness [m], operating_wavelength [m])
   </tool>
   - For superpixels, use: <tool>neural_design
   design_superpixel(refractive_index, length [m], incident_angle [deg], diffraction_angle [deg], thickness [m], operating_wavelength [m])
   </tool>
2. Return the final answer wrapped in <response> tags. Make sure your code prints the final answer in the correct units
3. If no calculations are needed, simply state the answer directly
4. You can only use ONE type of tag per message
5. Make sure to convert intermediate results to the correct units before using them to prevent multiplication or function unit mismatch errors

IMPORTANT: Any text not wrapped in tags will be treated as your internal thoughts and planning. Only text within <response> tags will be shown to the user.

Examples:

1. Respond to the user (Without wrapping in <response> tags the user will not be able to see your response!):
   <response>
   Your final answer or response to the user
   </response>

2. For neural network design (return the text you receive so the user can run the API call):
<tool>neural_design
design_metalens(refractive_index=2.7, lens_diameter=100e-6, focal_length=200e-6, thickness=500e-9, operating_wavelength=800e-9)
</tool>

Example workflow:
1. Think about approach:
    This problem requires calculating X, then checking material properties, and finally calling the neural network design tool.

2. Perform calculations:
   To find the angle of incidence, we need to use Snell's law:
   n1 * sin(theta1) = n2 * sin(theta2)
   where n1 is the refractive index of the first medium, n2 is the refractive index of the second medium, theta1 is the angle of incidence, and theta2 is the angle of refraction.

   Thus theta2 = 1.8 radians

3. Call the neural network design tool:
<tool>neural_design
design_superpixel(refractive_index=2.7, length=100e-6, incident_angle=10, diffraction_angle=20, thickness=500e-9, operating_wavelength=800e-9)
</tool>

4. Provide final answer:
<response>
The superpixel design is completed by the API call to design_superpixel(refractive_index=2.7, length=100e-6, incident_angle=10, diffraction_angle=20, thickness=500e-9, operating_wavelength=800e-9)
</response>"""

    def _format_messages(self, problem: str) -> List[Dict[str, str]]:
        """
        格式化消息列表
        
        将问题格式化为包含系统提示词的消息列表。
        
        参数:
            problem: str - 用户提出的问题
            
        返回:
            List[Dict[str, str]] - 格式化的消息列表
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem}
        ]
    
    async def _log_conversation(self, problem_id: str, messages: List[Dict[str, str]]):
        """
        记录对话到JSON文件
        
        将对话消息追加到日志文件中，用于调试和分析智能体行为。
        
        参数:
            problem_id: str - 问题的唯一标识符
            messages: List[Dict[str, str]] - 要记录的消息列表
        """
        log_file = self.log_dir / f"{problem_id}.json"
        
        # 如果日志文件已存在，先加载现有内容
        existing_log = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_log = json.load(f)
            except json.JSONDecodeError:
                existing_log = []  # 如果文件损坏，从空列表开始
                
        # 追加新消息到现有日志
        existing_log.extend(messages)
        
        # 写入更新后的日志
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_log, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to log file: {e}")

    async def solve(self, problem: str, problem_id: Optional[str] = None, temperature: float = 1.0, disable_cache: bool = False) -> Dict[str, Any]:
        """
        解决问题的主方法
        
        使用迭代推理方法解决光学设计问题。智能体会进行多轮内部独白，
        调用工具，最终生成答案。
        
        参数:
            problem: str - 问题描述
            problem_id: Optional[str] - 问题ID，用于日志记录，None则自动生成
            temperature: float - 模型生成温度，控制创造性
            disable_cache: bool - 是否禁用缓存（通过添加时间戳）
            
        返回:
            Dict[str, Any] - 包含以下键的字典：
                - solution: str - 最终答案
                - metadata: Dict - 元数据（迭代次数、对话历史等）
                - tool_calls: List - 工具调用记录
                或在错误情况下：
                - status: "error"
                - error: str - 错误信息
        """
        # 如果未提供问题ID，生成唯一ID
        if not problem_id:
            problem_id = str(uuid.uuid4())
            
        # 记录初始问题到日志
        await self._log_conversation(problem_id, [{
            "role": "user",
            "content": problem,
            "timestamp": datetime.now().isoformat()
        }])

        # 如果禁用缓存，添加时间戳到问题中
        # 这确保每次调用都是唯一的，避免LLM缓存
        if disable_cache:
            timestamp = datetime.now().isoformat()
            problem = f"Date submitted: {timestamp}\n\n{problem}"

        # 格式化初始消息
        messages = self._format_messages(problem)
        iteration_count = 0  # 迭代计数器
        max_iterations = 20  # 最大迭代次数限制，防止无限循环
        conversation = []  # 对话历史记录

        # 主迭代循环 - 持续运行直到自然完成或达到最大迭代次数
        while True:
            iteration_count += 1
            
            # 检查是否超过最大迭代次数
            if iteration_count > max_iterations:
                # 记录错误日志
                await self._log_conversation(problem_id, [{
                    "role": "error",
                    "content": "Maximum iterations reached",
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])

                # 返回错误状态
                return {
                    "status": "error",
                    "error": "Maximum iterations reached",
                    "conversation": conversation,
                    "problem_id": problem_id
                }
            
            # 调试模式下打印当前迭代信息
            if self.debug:
                print(f"\n=== Iteration {iteration_count} ===")
            
            # 调用大语言模型生成响应
            solution = await self._call_model(messages, temperature=temperature)
            current_response = solution

            # 记录模型的响应到日志
            await self._log_conversation(problem_id, [{
                "role": "assistant",
                "content": current_response,
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration_count
            }])

            # 处理最终响应 - 检查是否包含<response>标签
            if "<response>" in current_response:
                # 提取<response>标签中的内容作为最终答案
                response = current_response.split("<response>")[1].split("</response>")[0].strip()
                
                # 记录最终响应到日志
                await self._log_conversation(problem_id, [{
                    "role": "response",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])
                
                # 返回成功结果
                return {
                    "solution": response,  # 最终答案
                    "metadata": {
                        "method": "iterative",  # 使用的方法
                        "num_iterations": iteration_count,  # 迭代次数
                        "conversation": conversation,  # 对话历史
                        "problem_id": problem_id  # 问题ID
                    },
                    "tool_calls": self.tool_calls  # 工具调用记录
                }

            # 处理工具调用和思考过程
            processed_solution = current_response
            
            # 如果没有任何标签，将其视为内部思考/规划
            # 这是AIM的关键特性 - 允许智能体进行内部独白
            if not any(tag in processed_solution for tag in ["<tool>", "<response>"]):
                # 记录思考过程到日志
                await self._log_conversation(problem_id, [{
                    "role": "thinking",
                    "content": processed_solution,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])
                
                # 将思考添加到对话历史
                messages.append({"role": "assistant", "content": processed_solution})
                # 提示智能体继续或给出最终答案
                messages.append({"role": "user", "content": "Continue with your approach. If you need to reply to the user with an answer or need clarification, respond with <response> tags."})

                continue  # 继续下一轮迭代

            # 处理工具调用 - 目前支持 neural_design 工具
            for tool_name in ['neural_design']:
                if f"<tool>{tool_name}" in processed_solution:
                    # 查找工具调用的起始和结束位置
                    start = processed_solution.find(f"<tool>{tool_name}")
                    end = processed_solution.find("</tool>", start)
                    if end != -1:
                        # 提取工具调用代码块
                        code_block = processed_solution[start + len(f"<tool>{tool_name}"):end].strip()
                        
                        # 记录工具调用到日志
                        await self._log_conversation(problem_id, [{
                            "role": "tool_call",
                            "tool": tool_name,
                            "code": code_block,
                            "timestamp": datetime.now().isoformat(),
                            "iteration": iteration_count
                        }])
                        
                        # 执行工具调用
                        result = await self.tools[tool_name].execute(code_block)
                        
                        # 记录工具响应到日志
                        await self._log_conversation(problem_id, [{
                            "role": "tool_response",
                            "tool": tool_name,
                            "result": result,
                            "timestamp": datetime.now().isoformat(),
                            "iteration": iteration_count
                        }])
                        
                        # 将工具调用和结果添加到对话历史
                        messages.append({"role": "assistant", "content": current_response})
                        messages.append({"role": "user", "content": f"Tool output: {result}"})

            # 更新对话历史记录
            # 保存每一轮迭代的输入输出和时间戳
            conversation.append({
                "iteration": iteration_count,
                "input": current_response,
                "output": processed_solution,
                "timestamp": datetime.now().isoformat()
            })