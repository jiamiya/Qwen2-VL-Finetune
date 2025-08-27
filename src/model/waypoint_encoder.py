from typing import Tuple, Union

try:
    import torch
    _has_torch = True
except Exception:
    _has_torch = False

import numpy as np

ArrayLike = Union[np.ndarray, "torch.Tensor"]


class WaypointDiffNormCoder:
    """
    对形状 (..., T, 2) 的 waypoints 执行：
      编码:  (x, y) --(时间差分)--> (dx, dy) --(归一化)--> (dx_n, dy_n)
      解码:  (dx_n, dy_n) --(反归一化)--> (dx, dy) --(累积和)--> (x, y)

    归一化策略（对齐你的统计范围）：
      - dx ∈ [dx_min, dx_max]（默认 [0, 10]），映射到 [-1, 1]
            dx_n = (dx - mid) / half_rng, 其中 mid=(max+min)/2, half_rng=(max-min)/2
      - dy ∈ [-dy_absmax, +dy_absmax]（默认 3），映射到 [-1, 1]
            dy_n = dy / dy_absmax

    参数
    ----
    dx_range : Tuple[float, float]
        dx 的经验范围 (min, max)，默认 (0.0, 10.0)
    dy_absmax : float
        |dy| 的经验上界，默认 3.0
    origin_xy : Tuple[float, float]
        差分时第一个点与原点做差分所用的原点，默认 (0.0, 0.0)
    """

    def __init__(
        self,
        dx_range: Tuple[float, float] = (0.0, 10.0),
        dy_absmax: float = 3.0,
        origin_xy: Tuple[float, float] = (0.0, 0.0),
    ):
        self.dx_min, self.dx_max = float(dx_range[0]), float(dx_range[1])
        assert self.dx_max > self.dx_min, "dx_range must satisfy max > min"
        self.dx_mid = 0.5 * (self.dx_min + self.dx_max)
        self.dx_half_rng = 0.5 * (self.dx_max - self.dx_min)

        self.dy_absmax = float(dy_absmax)
        assert self.dy_absmax > 0, "dy_absmax must be positive"

        self.origin_x = float(origin_xy[0])
        self.origin_y = float(origin_xy[1])

    # ---------- 工具：后端与类型适配 ----------
    @staticmethod
    def _is_torch(x: ArrayLike) -> bool:
        return _has_torch and isinstance(x, torch.Tensor)

    @staticmethod
    def _cat(arrs, dim, torch_backend: bool):
        return torch.cat(arrs, dim=dim) if torch_backend else np.concatenate(arrs, axis=dim)

    @staticmethod
    def _stack(arrs, dim, torch_backend: bool):
        return torch.stack(arrs, dim=dim) if torch_backend else np.stack(arrs, axis=dim)

    @staticmethod
    def _cumsum(x: ArrayLike, dim: int, torch_backend: bool):
        return torch.cumsum(x, dim=dim) if torch_backend else np.cumsum(x, axis=dim)

    @staticmethod
    def _zeros_like(x: ArrayLike, shape, torch_backend: bool):
        if torch_backend:
            return torch.zeros(*shape, dtype=x.dtype, device=x.device)
        else:
            return np.zeros(shape, dtype=x.dtype)

    @staticmethod
    def _to_tensor(x: ArrayLike, torch_backend: bool):
        if torch_backend:
            return x
        else:
            return x

    # ---------- 编码 ----------
    def encode(self, waypoints: ArrayLike) -> ArrayLike:
        """
        输入:
            waypoints: 形状 (..., T, 2)，最后一个维度为 (x, y)
        输出:
            diffs_norm: 形状 (..., T, 2)，最后一个维度为 (dx_n, dy_n)，范围约 [-1, 1]
        """
        assert waypoints.shape[-1] == 2, "The last dimension must be (x, y)"
        torch_backend = self._is_torch(waypoints)

        # 拆分 x, y
        x = waypoints[..., 0]  # (..., T)
        y = waypoints[..., 1]  # (..., T)
        # print('x', x)
        # print('y', y)

        # 计算时间差分，首帧与 origin 做差分
        # x_diff[t] = x[t] - x[t-1]; x_diff[0] = x[0] - origin_x
        if torch_backend:
            x_prev = torch.roll(x, shifts=1, dims=-1)
            y_prev = torch.roll(y, shifts=1, dims=-1)
            # 首帧替换为 origin
            x_prev[..., 0] = self.origin_x
            y_prev[..., 0] = self.origin_y
            dx = x - x_prev
            dy = y - y_prev
            # print('dx: ', dx)
            # print('dy: ', dy)
        else:
            x_prev = np.roll(x, shift=1, axis=-1)
            y_prev = np.roll(y, shift=1, axis=-1)
            x_prev[..., 0] = self.origin_x
            y_prev[..., 0] = self.origin_y
            dx = x - x_prev
            dy = y - y_prev

        # 归一化到 [-1, 1]
        dx_n = (dx - self.dx_mid) / self.dx_half_rng
        dy_n = dy / self.dy_absmax
        # print('dxn: ', dx_n)
        # print('dyn: ', dy_n)

        # 拼回 (..., T, 2)
        if torch_backend:
            diffs_norm = torch.stack([dx_n, dy_n], dim=-1)
        else:
            diffs_norm = np.stack([dx_n, dy_n], axis=-1)

        return diffs_norm

    # ---------- 解码 ----------
    def decode(self, diffs_norm: ArrayLike) -> ArrayLike:
        """
        输入:
            diffs_norm: 形状 (..., T, 2)，最后一维为 (dx_n, dy_n)，范围约 [-1, 1]
        输出:
            waypoints: 形状 (..., T, 2)，最后一维为 (x, y)
        """
        assert diffs_norm.shape[-1] == 2, "The last dimension must be (dx_n, dy_n)"
        torch_backend = self._is_torch(diffs_norm)

        dx_n = diffs_norm[..., 0]
        dy_n = diffs_norm[..., 1]

        # 反归一化
        dx = dx_n * self.dx_half_rng + self.dx_mid
        dy = dy_n * self.dy_absmax

        # 从差分恢复坐标：x[0] = origin_x + dx[0], x[t] = x[t-1] + dx[t]
        x = self._cumsum(dx, dim=-1, torch_backend=torch_backend)
        y = self._cumsum(dy, dim=-1, torch_backend=torch_backend)

        if torch_backend:
            x[..., 0] = self.origin_x + dx[..., 0]
            y[..., 0] = self.origin_y + dy[..., 0]
            waypoints = torch.stack([x, y], dim=-1)
        else:
            x[..., 0] = self.origin_x + dx[..., 0]
            y[..., 0] = self.origin_y + dy[..., 0]
            waypoints = np.stack([x, y], axis=-1)

        return waypoints


class WaypointRangeNormCoder:
    """
    将 (x, y) 轨迹点线性映射到 [-1, 1]，以及从 [-1, 1] 反映射回原值。

    初始化时只需要给定 x 和 y 的范围:
        - x_range = (x_min, x_max)
        - y_range = (y_min, y_max)

    encode:  (x - mid) / half_range
    decode:  x_n * half_range + mid

    输入输出形状: (..., T, 2)，最后一维是 (x, y)。
    """

    def __init__(
            self, x_range: Tuple[float, float] = (0., 100.),
            y_range: Tuple[float, float] = (-30., 30.),
            clip: bool = True):
        x_min, x_max = x_range
        y_min, y_max = y_range
        assert x_max > x_min and y_max > y_min, "范围必须满足 max > min"

        self.x_mid = 0.5 * (x_min + x_max)
        self.x_half = 0.5 * (x_max - x_min)
        self.y_mid = 0.5 * (y_min + y_max)
        self.y_half = 0.5 * (y_max - y_min)

        self.clip = clip

    @staticmethod
    def _is_torch(x: ArrayLike) -> bool:
        return _has_torch and isinstance(x, torch.Tensor)

    @staticmethod
    def _stack(arrs, dim, torch_backend: bool):
        return torch.stack(arrs, dim=dim) if torch_backend else np.stack(arrs, axis=dim)

    def encode(self, waypoints: ArrayLike) -> ArrayLike:
        assert waypoints.shape[-1] == 2, "最后一维必须是 (x, y)"
        torch_backend = self._is_torch(waypoints)

        x, y = waypoints[..., 0], waypoints[..., 1]

        x_n = (x - self.x_mid) / self.x_half
        y_n = (y - self.y_mid) / self.y_half

        if self.clip:
            if torch_backend:
                x_n = torch.clamp(x_n, -1.0, 1.0)
                y_n = torch.clamp(y_n, -1.0, 1.0)
            else:
                x_n = np.clip(x_n, -1.0, 1.0)
                y_n = np.clip(y_n, -1.0, 1.0)

        return self._stack([x_n, y_n], -1, torch_backend)

    def decode(self, normed: ArrayLike) -> ArrayLike:
        assert normed.shape[-1] == 2, "最后一维必须是 (x_n, y_n)"
        torch_backend = self._is_torch(normed)

        x_n, y_n = normed[..., 0], normed[..., 1]
        print(x_n, y_n)

        x = x_n * self.x_half + self.x_mid
        y = y_n * self.y_half + self.y_mid
        print(x,y)

        return self._stack([x, y], -1, torch_backend)


# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    coder = WaypointDiffNormCoder(dx_range=(0, 10), dy_absmax=3, origin_xy=(0.0, 0.0))

    # NumPy：单条轨迹 (T=10, 2)
    wp_np = np.array([[i, (-1)**i * 0.5] for i in range(10)], dtype=np.float32)
    enc_np = coder.encode(wp_np)          # (10, 2)
    dec_np = coder.decode(enc_np)         # (10, 2)，应≈还原到 wp_np

    # NumPy：批量 (B=3, T=10, 2)
    wp_np_b = np.stack([wp_np, wp_np + 1.0, wp_np + 2.0], axis=0)
    enc_np_b = coder.encode(wp_np_b)      # (3, 10, 2)
    dec_np_b = coder.decode(enc_np_b)     # (3, 10, 2)

    # 如果使用 PyTorch（可选）
    if _has_torch:
        wp_t = torch.tensor(wp_np_b, device="cpu")
        enc_t = coder.encode(wp_t)        # torch.Tensor
        dec_t = coder.decode(enc_t)       # torch.Tensor
