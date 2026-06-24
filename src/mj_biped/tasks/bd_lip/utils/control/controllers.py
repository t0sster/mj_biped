from __future__ import annotations

import math

import torch


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


class PositionPIDController:
    _DT = 0.02

    def __init__(
        self,
        kp: float = 4.0,
        ki: float = 1.5,
        kd: float = 0.5,
        kang: float = 1.5,
        vmax: float = 0.15,
        vmax_neg: float = 0.0,
        vy_max: float = 0.0,
        wzmax: float = 0.5,
        integral_clip: float = 0.5,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kang = kang
        self.vmax = vmax
        self.vmax_neg = vmax_neg
        self.vy_max = vy_max
        self.wzmax = wzmax
        self.integral_clip = integral_clip
        self._integral_w = torch.zeros(2)
        self._prev_err_b = torch.zeros(2)

    def reset(self) -> None:
        self._integral_w.zero_()
        self._prev_err_b.zero_()

    def compute(
        self,
        robot_pos_xy: torch.Tensor,
        robot_heading: torch.Tensor,
        target_xy: torch.Tensor,
    ) -> torch.Tensor:
        err_w = target_xy - robot_pos_xy

        c = torch.cos(robot_heading)
        s = torch.sin(robot_heading)

        err_bx = c * err_w[0] + s * err_w[1]
        err_by = -s * err_w[0] + c * err_w[1]

        self._integral_w += err_w * self._DT
        self._integral_w.clamp_(-self.integral_clip, self.integral_clip)
        int_bx = c * self._integral_w[0] + s * self._integral_w[1]
        int_by = -s * self._integral_w[0] + c * self._integral_w[1]

        err_b = torch.stack([err_bx, err_by])
        d_b = (err_b - self._prev_err_b) / self._DT
        self._prev_err_b = err_b.clone()

        vx = torch.clamp(
            self.kp * err_bx + self.ki * int_bx + self.kd * d_b[0],
            -self.vmax_neg, self.vmax,
        )
        vy = torch.clamp(
            self.kp * err_by + self.ki * int_by + self.kd * d_b[1],
            -self.vy_max, self.vy_max,
        )

        desired_heading = torch.atan2(err_w[1], err_w[0])
        heading_err = _wrap_to_pi(desired_heading - robot_heading)
        wz = torch.clamp(self.kang * heading_err, -self.wzmax, self.wzmax)

        return torch.stack([vx, vy, wz])


PositionPController = PositionPIDController


class ContinuousController:

    state_name = "WALK"

    def __init__(
        self,
        kang: float = 4.0,
        vmax: float = 0.15,
        wzmax: float = 0.5,
    ) -> None:
        self.kang = kang
        self.vmax = vmax
        self.wzmax = wzmax

    def reset(self) -> None:
        pass

    def compute(
        self,
        robot_pos_xy: torch.Tensor,
        robot_heading: torch.Tensor,
        target_xy: torch.Tensor,
        wp_idx: int = 0,
    ) -> torch.Tensor:
        err_w = target_xy - robot_pos_xy
        desired_heading = torch.atan2(err_w[1], err_w[0])
        heading_err = _wrap_to_pi(desired_heading - robot_heading)
        wz = torch.clamp(self.kang * heading_err, -self.wzmax, self.wzmax)
        return torch.tensor([self.vmax, 0.0, wz.item()])

