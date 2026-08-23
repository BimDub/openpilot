"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from openpilot.cereal import custom
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import MAX_T, STOP_DISTANCE
from openpilot.sunnypilot import get_sanitize_int_param

AccelProfile = custom.LongitudinalPlanSP.AccelController.Profile

MAX_ACCEL_BREAKPOINTS = [0., 3., 5., 8., 10., 25., 40.]
MAX_ACCEL_PROFILES = {
  AccelProfile.eco:    [1.20, 0.95, 0.70, 0.48, 0.40, 0.28, 0.18],
  AccelProfile.normal: [1.60, 1.35, 1.05, 0.65, 0.52, 0.40, 0.28],
  AccelProfile.sport:  [2.00, 1.90, 1.55, 1.00, 0.80, 0.60, 0.40],
}
TARGET_SPEED_DEADBAND = 0.2  # m/s
TARGET_SPEED_APPROACH_WINDOW = 2.0  # m/s
TARGET_SPEED_APPROACH_GAIN = 0.5
TARGET_SPEED_APPROACH_MIN_SPEED = 3.0  # m/s
TARGET_SPEED_APPROACH_FULL_SPEED = 5.0  # m/s
CATCHUP_ERROR_BREAKPOINTS = [TARGET_SPEED_DEADBAND, 0.5, 1.0, 2.0, 3.0, 4.0]
CATCHUP_ACCEL_SCALE = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
LEAD_COMFORT_ACCEL = -0.4  # m/s^2
LEAD_COMFORT_JERK = 0.6  # m/s^3


class AccelController:
  def __init__(self):
    self.params = Params()
    self.frame = 0
    self._profile = get_sanitize_int_param("AccelPersonality", AccelProfile.eco, AccelProfile.sport, self.params)
    self._enabled = self.params.get_bool("AccelPersonalityEnabled")

  def update(self) -> None:
    self.frame += 1
    if self.frame % int(1.0 / DT_MDL) == 0:
      self._profile = get_sanitize_int_param("AccelPersonality", AccelProfile.eco, AccelProfile.sport, self.params)
      self._enabled = self.params.get_bool("AccelPersonalityEnabled")

  @property
  def profile(self) -> int:
    return self._profile

  def is_enabled(self) -> bool:
    return self._enabled

  def get_max_accel(self, v_ego: float, v_target: float | None = None) -> float:
    v_ego = max(0.0, v_ego)
    max_accel = float(np.interp(v_ego, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES[self._profile]))
    if v_target is None or v_ego <= TARGET_SPEED_APPROACH_MIN_SPEED or v_target <= v_ego:
      return max_accel

    speed_error = v_target - v_ego
    speed_blend = float(np.interp(v_ego, [TARGET_SPEED_APPROACH_MIN_SPEED, TARGET_SPEED_APPROACH_FULL_SPEED], [0.0, 1.0]))
    catchup_scale = float(np.interp(speed_error, CATCHUP_ERROR_BREAKPOINTS, CATCHUP_ACCEL_SCALE))
    raw_accel = min(speed_error, max_accel)
    catchup_accel = min(speed_error, max_accel * catchup_scale)
    return float(raw_accel + speed_blend * (catchup_accel - raw_accel))

  def get_cruise_target(self, v_ego: float, v_target: float) -> float:
    if v_ego <= TARGET_SPEED_APPROACH_MIN_SPEED:
      return v_target

    speed_error = v_target - v_ego
    if speed_error >= 0.0:
      return v_target

    speed_blend = float(np.interp(v_ego, [TARGET_SPEED_APPROACH_MIN_SPEED, TARGET_SPEED_APPROACH_FULL_SPEED], [0.0, 1.0]))
    target_blend = float(np.interp(abs(speed_error), [TARGET_SPEED_DEADBAND, TARGET_SPEED_APPROACH_WINDOW], [1.0, 0.0]))
    deadband = TARGET_SPEED_DEADBAND * speed_blend * target_blend
    adjusted_error = np.sign(speed_error) * max(0.0, abs(speed_error) - deadband)
    gain = 1.0 - (1.0 - TARGET_SPEED_APPROACH_GAIN) * speed_blend * target_blend
    return float(v_ego + gain * adjusted_error)

  def get_lead_departure_accel(self, v_ego: float, v_lead: float, a_lead: float, mpc_accel: float) -> float:
    values = (v_ego, v_lead, a_lead, mpc_accel)
    if not self._enabled or not all(np.isfinite(value) for value in values) or a_lead < 0.0 or mpc_accel < 0.0 or v_lead <= v_ego:
      return mpc_accel

    return max(mpc_accel, min(v_lead - v_ego, self.get_max_accel(v_ego, v_lead)))

  def get_lead_accel(self, v_ego: float, d_rel: float, v_rel: float, a_lead: float,
                     cruise_accel: float, previous_accel: float, t_follow: float) -> float:
    values = (v_ego, d_rel, v_rel, a_lead, cruise_accel, previous_accel, t_follow)
    if not self._enabled or not all(np.isfinite(value) for value in values) or v_ego < TARGET_SPEED_APPROACH_FULL_SPEED or d_rel <= 0.0:
      return cruise_accel

    v_lead = max(v_ego + v_rel, 0.0)
    clearance = d_rel - (STOP_DISTANCE + t_follow * v_lead)
    lead_lookahead = 2.0 * t_follow
    projected_closing = max(-(v_rel + a_lead * lead_lookahead), 0.0)
    if projected_closing <= 0.0 or clearance / projected_closing >= MAX_T:
      return cruise_accel

    required_accel = a_lead - projected_closing ** 2 / (2.0 * max(clearance, STOP_DISTANCE))
    target_accel = float(np.clip(required_accel, LEAD_COMFORT_ACCEL, 0.0))

    max_step = LEAD_COMFORT_JERK * DT_MDL
    previous_comfort_accel = max(previous_accel, LEAD_COMFORT_ACCEL)
    lead_accel = float(np.clip(target_accel, previous_comfort_accel - max_step, previous_comfort_accel + max_step))
    return min(cruise_accel, lead_accel)
