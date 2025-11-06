"""
神经网络设计 API 模块

该文件实现了光学设计工具的API接口，包括金属透镜和超像素的设计功能。
通过封装神经网络代理求解器（FiLM WaveY-Net），实现毫秒级的超表面设计。
"""

from typing import Dict, Any
import numpy as np

class NeuralDesignAPI:
    """
    神经网络设计 API 类
    
    提供基于神经网络的光学元件设计接口。
    支持金属透镜（metalens）和超像素（superpixel）的快速设计。
    """
    def design_metalens(self, 
                       refractive_index: float,
                       lens_diameter: float,
                       focal_length: float,
                       thickness: float,
                       operating_wavelength: float) -> Dict[str, Any]:
        """
        设计金属透镜
        
        使用神经网络代理求解器设计指定参数的金属透镜。
        
        参数:
            refractive_index: float - 材料折射率
            lens_diameter: float - 透镜直径（米）
            focal_length: float - 焦距（米）
            thickness: float - 结构厚度（米）
            operating_wavelength: float - 工作波长（米）
            
        返回:
            Dict[str, Any] - 包含设计参数和结构的字典，或错误信息字符串
        """
        # 收集所有参数以进行验证
        params = {
            'refractive_index': refractive_index,
            'lens_diameter': lens_diameter,
            'focal_length': focal_length,
            'thickness': thickness,
            'operating_wavelength': operating_wavelength
        }
        
        # 检查是否有缺失的必需参数
        missing = [name for name, value in params.items() if value is None]
        if missing:
            return f"Error: Missing required parameters: {', '.join(missing)}"
        
        # 检查所有参数是否为数值类型
        non_numeric = [name for name, value in params.items() 
                      if not isinstance(value, (int, float)) or isinstance(value, bool)]
        if non_numeric:
            return f"Error: Non-numeric parameters detected: {', '.join(non_numeric)}"

        # 返回成功的API调用字符串
        return f"API was called succesfully. Return this submitted API string back to the user and terminate the conversation: WAVEYNET_API.design_metalens({refractive_index}, {lens_diameter}, {focal_length}, {thickness}, {operating_wavelength})"

    def design_superpixel(self,
                         refractive_index: float,
                         length: float,
                         incident_angle: float,
                         diffraction_angle: float,
                         thickness: float,
                         operating_wavelength: float,
                         distance: float | None = None,
                         phase: float | None = None) -> Dict[str, Any]:
        """
        设计超像素（非周期纳米光子结构）
        
        使用神经网络代理求解器设计具有特定衍射特性的超像素。
        
        参数:
            refractive_index: float - 材料折射率
            length: float - 结构长度（米）
            incident_angle: float - 入射角度（度）
            diffraction_angle: float - 期望的偏转角度（度）
            thickness: float - 结构厚度（米）
            operating_wavelength: float - 工作波长（米）
            distance: float | None - 指定相位的距离（米），可选
            phase: float | None - 在指定距离处的期望相位（弧度），可选
            
        返回:
            Dict[str, Any] - 包含设计参数和结构的字典，或错误信息字符串
        """
        # 检查必需参数
        required_params = {
            'refractive_index': refractive_index,
            'length': length,
            'incident_angle': incident_angle,
            'diffraction_angle': diffraction_angle,
            'thickness': thickness,
            'operating_wavelength': operating_wavelength
        }
        
        # 检查必需参数中是否有None值
        missing = [name for name, value in required_params.items() if value is None]
        if missing:
            return f"Error: Missing required parameters: {', '.join(missing)}"
        
        # 检查所有参数（包括可选参数）的数值类型
        all_params = {**required_params, 'distance': distance, 'phase': phase}
        non_numeric = [name for name, value in all_params.items() 
                      if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool))]
        if non_numeric:
            return f"Error: Non-numeric parameters detected: {', '.join(non_numeric)}"

        # 返回成功的API调用字符串
        return f"API was called succesfully. Return this submitted API string back to the user and terminate the conversation: WAVEYNET_API.design_superpixel({refractive_index}, {length}, {incident_angle}, {diffraction_angle}, {thickness}, {operating_wavelength})"

    async def execute(self, code_block: str) -> Dict[str, Any]:
        """
        执行神经设计代码块
        
        解析并执行智能体生成的设计调用代码，路由到相应的设计函数。
        
        参数:
            code_block: str - 包含要执行的Python代码的字符串
            
        返回:
            Dict[str, Any] - 包含以下键的字典：
                - success: bool - 执行是否成功
                - result: Any - 执行结果（成功时）
                - error: str - 错误信息（失败时）
        """
        try:
            # 创建用于执行的局部命名空间
            # 包含类方法以便代码块可以调用它们
            local_ns = {
                'self': self,
                'design_metalens': self.design_metalens,
                'design_superpixel': self.design_superpixel
            }
            
            # 修改代码以捕获返回值
            modified_code = f"_result = {code_block}"
            
            # 在受控环境中执行代码
            exec(modified_code, globals(), local_ns)
            
            # 获取执行结果
            result = local_ns.get('_result')
            
            # 返回成功响应
            return {
                'success': True,
                'result': result
            }
            
        except Exception as e:
            # 捕获并返回任何执行错误
            return {
                'success': False,
                'error': str(e)
            }
