import torch

@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

@torch.jit.script
def quat_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    # (N, 4) -> (N, 3, 3)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, xy, xz, xw = x * x, x * y, x * z, x * w
    yy, yz, yw = y * y, y * z, y * w
    zz, zw = z * z, z * w
    m = torch.stack(
        [
            1.0 - 2.0 * (yy + zz),    2.0 * (xy - zw),          2.0 * (xz + yw),
            2.0 * (xy + zw),          1.0 - 2.0 * (xx + zz),    2.0 * (yz - xw),
            2.0 * (xz - yw),          2.0 * (yz + xw),          1.0 - 2.0 * (xx + yy),
        ],
        dim=-1,
    ).view(-1, 3, 3)
    return m

@torch.jit.script
def quat_to_z_axis(q: torch.Tensor) -> torch.Tensor:
    # (N, 4) -> (N, 3)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, xz, xw = x * x, x * z, x * w
    yy, yz, yw = y * y, y * z, y * w
    m3 = torch.stack([
        2.0 * (xz + yw),
        2.0 * (yz - xw),
        1.0 - 2.0 * (xx + yy)
    ],dim=-1)
    return m3

@torch.jit.script
def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    # (N, 4) -> (N)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z - x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw

@torch.jit.script
def inner_product(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    # (*N, D), (*N, D) -> (*N, 1)
    return torch.sum(v1 * v2, dim=-1, keepdim=True)

@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # rotate vectors by quaternions (w, x, y, z)
    # (*N, 4), (*N, 3) -> (*N, 3)
    q_w = q[..., 0:1]  # (*N, 1)
    q_vec = q[..., 1:4]  # (*N, 3)
    a = v * (2.0 * q_w * q_w - 1.0)  # (*N, 3)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0  # (*N, 3)
    c = q_vec * inner_product(q_vec, v) * 2.0
    return a + b + c

@torch.jit.script
def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # rotate vectors by quaternions (w, -x, -y, -z)
    # (*N, 4), (*N, 3) -> (*N, 3)
    q_w = q[..., 0:1]
    q_vec = -q[..., 1:4]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * inner_product(q_vec, v) * 2.0
    return a + b + c

@torch.jit.script
def vee_map(skew_matrix: torch.Tensor) -> torch.Tensor:
    # return vee map vectors of skew matrixs
    # (*N, 3, 3) -> (*N, 3)
    v = torch.stack(
        [
        skew_matrix[..., 2, 1] - skew_matrix[..., 1, 2],
        skew_matrix[..., 0, 2] - skew_matrix[..., 2, 0],
        skew_matrix[..., 1, 0] - skew_matrix[..., 0, 1]
        ],
        dim=-1
    )
    return v / 2.0

@torch.jit.script
def quat_inv(q: torch.Tensor):
    return torch.cat((q[..., 0:1], -q[..., 1:4]), dim=-1)