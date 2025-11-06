"""
物理场计算模块

该文件实现了电磁场的有限差分计算方法。
包含 Hz 场到 Ex、Ey 场的转换，以及实现亥姆霍兹方程的完整场计算。
这些函数用于神经网络的物理一致性约束和场重建。
"""

import numpy as np
import torch
import consts

def Hz_to_Ex(Hz_R: torch.Tensor, Hz_I: torch.Tensor, Mz_R: torch.Tensor, Mz_I: torch.Tensor, dL: float, omega: torch.Tensor, eps_grid: torch.Tensor, \
             eps_0: float = consts.eps_0) -> torch.Tensor:
    """
    通过有限差分从 Hz 场计算 Ex 场
    
    使用有限差分方法从磁场分量 Hz 计算电场分量 Ex。
    注意：ceviche 实现的有限差分与传统设置略有不同。
    
    参数:
        Hz_R: torch.Tensor - Hz 场的实部 (-Hy)
        Hz_I: torch.Tensor - Hz 场的虚部
        Mz_R: torch.Tensor - 磁流源的实部（未使用，但保留接口一致性）
        Mz_I: torch.Tensor - 磁流源的虚部（未使用，但保留接口一致性）
        dL: float - 网格步长
        omega: torch.Tensor - 角频率
        eps_grid: torch.Tensor - 材料介电常数网格
        eps_0: float - 真空介电常数，默认使用 consts.eps_0
    
    返回:
        torch.Tensor - 计算得到的 Ex 场（实部和虚部堆叠）
        
    注意:
        - Hz[:, 1] 和 Hz[:, 0] 的差分产生 Ex[:, 0] 而不是 Ex[:, 1]
        - 材料网格需要比场网格在底部多一维以进行平均
    """
    # 材料参数平均 - 在网格点之间插值
    x = 1 / 2 * (eps_grid[:, :, 1:, :] + eps_grid[:, :, 0:-1, :])
    
    # 计算 Ex 场的实部 - 使用 Hz 虚部的差分
    Ex_R = (Hz_I[:, 1:, :] - Hz_I[:, 0:-1, :])/dL/omega/eps_0/x[:, 0, 0:-1, :]
    
    # 计算 Ex 场的虚部 - 使用 Hz 实部的差分（带负号）
    Ex_I = -(Hz_R[:, 1:, :] - Hz_R[:, 0:-1, :])/dL/omega/eps_0/x[:, 0, 0:-1, :]

    # 将实部和虚部堆叠返回
    return torch.stack((Ex_R, Ex_I), axis = 1)

def Hz_to_Ey(Hz_R: torch.Tensor, Hz_I: torch.Tensor, Mz_R: torch.Tensor, Mz_I: torch.Tensor, dL: float, omega: torch.Tensor, eps_grid: torch.Tensor, \
             eps_0: float = consts.eps_0) -> torch.Tensor:
    """
    通过有限差分从 Hz 场计算 Ey 场
    
    使用有限差分方法从磁场分量 Hz 计算电场分量 Ey (Ez)。
    
    参数:
        Hz_R: torch.Tensor - Hz 场的实部 (-Hy)
        Hz_I: torch.Tensor - Hz 场的虚部
        Mz_R: torch.Tensor - 磁流源的实部（未使用，但保留接口一致性）
        Mz_I: torch.Tensor - 磁流源的虚部（未使用，但保留接口一致性）
        dL: float - 网格步长
        omega: torch.Tensor - 角频率
        eps_grid: torch.Tensor - 材料介电常数网格
        eps_0: float - 真空介电常数，默认使用 consts.eps_0
    
    返回:
        torch.Tensor - 计算得到的 Ey (Ez) 场（实部和虚部堆叠）
        
    注意:
        - Hz[1, :] 和 Hz[0, :] 的差分产生 Ey[0, :] 而不是 Ey[1, :]
        - 由于周期性结构，材料网格维度与场网格相同
    """
    # 材料参数平均 - 考虑周期性边界条件
    y = 1 / 2 * (eps_grid[:, :, 1:, :] + torch.roll(eps_grid[:, :, 1:, :], 1, dims = 3))

    # 计算 Ey 场的实部 - 使用 Hz 虚部的差分（带负号）
    Ey_R = -(torch.roll(Hz_I, -1, dims = 2) - Hz_I)/dL/omega/eps_0/y[:, 0, :, :]

    # 计算 Ey 场的虚部 - 使用 Hz 实部的差分
    Ey_I = (torch.roll(Hz_R, -1, dims = 2) - Hz_R)/dL/omega/eps_0/y[:, 0, :, :]

    # 将实部和虚部堆叠返回
    return torch.stack((Ey_R, Ey_I), axis = 1)

def E_to_Hz(Ey_R: torch.Tensor, Ey_I: torch.Tensor, Ex_R: torch.Tensor, Ex_I: torch.Tensor, Mz_R: torch.Tensor, Mz_I: torch.Tensor, dL: float, \
            omega: torch.Tensor, mu_0: float = consts.mu_0) -> torch.Tensor:
    """
    通过有限差分从 Ey 和 Ex 场计算 Hz 场
    
    使用电场分量的有限差分重建磁场分量。
    
    参数:
        Ey_R: torch.Tensor - Ey 场的实部 (Ez)
        Ey_I: torch.Tensor - Ey 场的虚部
        Ex_R: torch.Tensor - Ex 场的实部
        Ex_I: torch.Tensor - Ex 场的虚部
        Mz_R: torch.Tensor - 磁流源的实部（未使用，但保留接口一致性）
        Mz_I: torch.Tensor - 磁流源的虚部（未使用，但保留接口一致性）
        dL: float - 网格步长
        omega: torch.Tensor - 角频率
        mu_0: float - 真空磁导率，默认使用 consts.mu_0
    
    返回:
        torch.Tensor - 计算得到的 Hz (-Hy) 场（实部和虚部堆叠）
        
    注意:
        - 分母中的 -1j 已被吸收用于 Hz -> -Hy 的转换
    """
    # 计算 Hz 场的实部 - 使用 Ey 和 Ex 虚部的差分
    Hz_R = ((Ey_I[:, 1:] - torch.roll(Ey_I[:, 1:], 1, dims = 2)) - (Ex_I[:, 1:] - \
             Ex_I[:, 0:-1]))/dL/omega/mu_0
    
    # 计算 Hz 场的虚部 - 使用 Ey 和 Ex 实部的差分（带负号）
    Hz_I = -((Ey_R[:, 1:] - torch.roll(Ey_R[:, 1:], 1, dims = 2)) - (Ex_R[:, 1:] - \
              Ex_R[:, 0:-1]))/dL/omega/mu_0
    
    # 将实部和虚部堆叠返回
    return torch.stack((Hz_R, Hz_I), axis = 1)

def H_to_H(Hz_R: torch.Tensor, Hz_I: torch.Tensor, Mz_R: torch.Tensor, Mz_I: torch.Tensor, dL: float, omega: torch.Tensor, eps_grid: torch.Tensor, \
           eps_0: float = consts.eps_0, mu_0: float = consts.mu_0) -> torch.Tensor:
    """
    实现磁场的亥姆霍兹方程
    
    通过调用 Hz_to_Ex、Hz_to_Ey 和 E_to_Hz 实现磁场的完整亥姆霍兹方程。
    这是物理一致性约束的关键，确保场满足麦克斯韦方程。
    
    参数:
        Hz_R: torch.Tensor - Hz 场的实部
        Hz_I: torch.Tensor - Hz 场的虚部
        Mz_R: torch.Tensor - 磁流源的实部
        Mz_I: torch.Tensor - 磁流源的虚部
        dL: float - 网格步长
        omega: torch.Tensor - 角频率
        eps_grid: torch.Tensor - 材料介电常数网格
        eps_0: float - 真空介电常数，默认使用 consts.eps_0
        mu_0: float - 真空磁导率，默认使用 consts.mu_0
    
    返回:
        torch.Tensor - 满足亥姆霍兹方程的磁场
        
    注意:
        - 包含源项的贡献
        - 用于训练时的物理约束损失计算
    """
    # 步骤1: 从 Hz 计算 Ex 场
    FD_Ex = Hz_to_Ex(Hz_R, Hz_I, Mz_R, Mz_I, dL, omega, eps_grid, eps_0)
    
    # 步骤2: 从 Hz 计算 Ey 场
    FD_Ey = Hz_to_Ey(Hz_R, Hz_I, Mz_R, Mz_I, dL, omega, eps_grid, eps_0)
    
    # 步骤3: 从 Ex 和 Ey 重建 Hz 场（应用亥姆霍兹算子）
    FD_H = -E_to_Hz(-FD_Ey[:, 0, :-1], -FD_Ey[:, 1, :-1], -FD_Ex[:, 0], -FD_Ex[:, 1], Mz_R, Mz_I, dL, omega, mu_0)

    # 计算源项贡献
    source_vector = (1/(mu_0)) * torch.stack((Mz_I, -Mz_R), axis = 1)[:,:,1:-1,:]

    # 减去源项得到最终结果
    FD_H -= source_vector
    
    return FD_H