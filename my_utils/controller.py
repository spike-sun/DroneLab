import torch
from .math import normalize, quat_to_rotation_matrix, inner_product, quat_rotate, quat_rotate_inv, vee_map

class PositionController:
    def __init__(self, num_envs, device, scale):
        # 多个无人机共用一个实例，不要在内部保存状态
        self.num_envs = num_envs
        self.device = device
        self.scale = scale
        self.zeros = torch.zeros(num_envs, device=device)
        self.ones = torch.ones(num_envs, device=device)
        self.e3 = torch.stack([self.zeros, self.zeros, self.ones], dim=-1)
        
        self.m = 0.0282
        self.g = 9.81
        self.gravity_vector = - self.g * self.e3
        J_body = torch.tensor([1.6572e-5, 1.6656e-5, 2.9262e-5], device=device)
        J_prop = 4.0 * torch.tensor([2.0000e-9, 1.6700e-7, 1.6800e-7], device=device)
        # 旋翼的质量产生的转动惯量ml^2, 改变scale后质量不变, 但臂长变了, 所以转动惯量改变
        J_axis = scale * scale * torch.tensor([3.0752e-6, 3.0752e-6, 1.4112e-6], device=device)
        self.J = J_body + J_prop + J_axis
        self.z_w = torch.tensor([0.0, 0.0, 1.0], device=device).expand(num_envs, 3)
        self.kp = torch.ones(3, device=device) * 20.0
        self.kv = torch.ones(3, device=device) * 15.0
        self.kr = torch.ones(3, device=device) * 200.0
        self.kw = torch.ones(3, device=device) * 40.0
        self.thrust_max = 5.0 * self.m * self.g
        self.torque_min = -0.04 * scale * scale
        self.torque_max = 0.04 * scale * scale
    
    def pos_cmd(self, pos, quat, vel_w, angvel_b, pos_des, pos_des_dot, pos_des_ddot):
        pos_error = pos - pos_des
        vel_w_error = vel_w - pos_des_dot
        acc_w_des = - self.kp * pos_error - self.kv * vel_w_error + pos_des_ddot
        return self.acc_yawrate_cmd(quat, angvel_b, acc_w_des, self.zeros)
    
    def acc_yaw_cmd(self, quat, angvel_b, acc_w_des, yaw_des):
        x_c_des = torch.stack([
            torch.cos(yaw_des),
            torch.sin(yaw_des),
            self.zeros
        ], dim=-1)
        z_b = quat_rotate(quat, self.z_w)
        f_w_des = self.m * (acc_w_des + self.g * self.z_w)
        thrust = inner_product(f_w_des, z_b)
        z_b_des = normalize(f_w_des)
        y_b_des = normalize(torch.cross(z_b_des, x_c_des, dim=-1))
        x_b_des = torch.cross(y_b_des, z_b_des, dim=-1)
        rotmat_des = torch.stack([x_b_des, y_b_des, z_b_des], dim=-1)
        return thrust.squeeze(-1).clamp(0.0, self.thrust_max), self.rotmat_cmd(quat, angvel_b, rotmat_des)

    def acc_yawrate_cmd(self, quat, angvel_b, acc_w_des, yawrate_des):
        z_b = quat_rotate(quat, self.z_w)
        f_w_des = self.m * (acc_w_des + self.g * self.z_w)
        thrust = inner_product(f_w_des, z_b)
        thrust = thrust.squeeze(-1).clamp(0.0, self.thrust_max)  # 正数，z轴正方向上的推力大小

        z_b_des = normalize(f_w_des)
        cp = torch.cross(z_b, z_b_des, dim=-1)
        nonzero = torch.norm(cp, p=2, dim=-1) > 1e-3
        n_w = torch.zeros((self.num_envs, 3), device=self.device)
        n_w[nonzero, :] = normalize(cp[nonzero, :])
        n_b = quat_rotate_inv(quat, n_w)
        alpha = torch.acos(inner_product(z_b, z_b_des).clamp(max=1.0))
        angvel_b_des = torch.concat([
            20.0 * torch.sign(torch.cos(alpha / 2.0)) * torch.sin(alpha / 2.0) * n_b[:, 0:2],
            yawrate_des.unsqueeze(-1)
        ], dim=-1)
        
        return thrust, self.angvel_cmd(angvel_b, angvel_b_des)

    def acc_yawrate_cmd_hack(self, quat, angvel_b, acc_b_des, yawrate_des):
        # 控制线加速度和偏航角速度，但保持z轴竖直
        force = self.m * (acc_b_des - quat_rotate_inv(quat, self.gravity_vector))  # 体坐标系下的力矢量

        z_b = quat_rotate(quat, self.z_w)
        cp = torch.cross(z_b, self.z_w, dim=-1)
        nonzero = torch.norm(cp, p=2, dim=-1) > 1e-3
        n_w = torch.zeros((self.num_envs, 3), device=self.device)
        n_w[nonzero, :] = normalize(cp[nonzero, :])
        n_b = quat_rotate_inv(quat, n_w)
        alpha = torch.acos(inner_product(z_b, self.z_w).clamp(max=1.0))
        angvel_b_des = torch.concat([
            20.0 * torch.sign(torch.cos(alpha / 2.0)) * torch.sin(alpha / 2.0) * n_b[:, 0:2],
            yawrate_des.unsqueeze(-1)
        ], dim=-1)

        return force, self.angvel_cmd(angvel_b, angvel_b_des)
    
    def rotmat_cmd(self, quat: torch.Tensor, angvel_b: torch.Tensor, rotmat_des: torch.Tensor) -> torch.Tensor:
        rotmat = quat_to_rotation_matrix(quat)
        rotmat_error = 0.5 * vee_map(rotmat_des.transpose(1,2) @ rotmat - rotmat.transpose(1,2) @ rotmat_des)
        torque = self.J * (- self.kr * rotmat_error - self.kw * angvel_b + torch.cross(angvel_b, angvel_b, dim=-1))
        return torque.clamp(self.torque_min, self.torque_max)
    
    def angvel_cmd(self, angvel_b: torch.Tensor, angvel_b_des: torch.Tensor) -> torch.Tensor:
        angvel_error = angvel_b - angvel_b_des
        torque = self.J * (- self.kw * angvel_error + torch.cross(angvel_b, angvel_b, dim=-1))
        return torque.clamp(self.torque_min, self.torque_max)