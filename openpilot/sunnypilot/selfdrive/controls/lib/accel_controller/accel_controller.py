"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from openpilot.cereal import custom
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import get_sanitize_int_param

AccelProfile = custom.LongitudinalPlanSP.AccelController.Profile

MAX_ACCEL_BREAKPOINTS = [0., 3., 5., 10., 20., 25., 40.]
# Road-speed values are held to a fraction of stock (A_CRUISE_MAX_VALS interpolates to 0.80 at 25 m/s and
# 0.60 at 40 m/s): eco ~65-70%, normal ~85%. Below that a profile cannot hold speed on grade - 1% of grade
# costs 0.098 m/s^2 of gravity - and merges and passes stop working. eco never exceeds stock anywhere.
MAX_ACCEL_PROFILES = {
  AccelProfile.eco:    [1.60, 1.48, 1.22, 0.86, 0.66, 0.52, 0.40],
  AccelProfile.normal: [1.90, 1.70, 1.42, 0.99, 0.80, 0.66, 0.52],
  AccelProfile.sport:  [2.00, 2.00, 1.86, 1.30, 1.02, 0.86, 0.72],
}
# Comfort budget for a speed change. sqrt(J * dv) is the peak acceleration of a constant-jerk maneuver that
# changes speed by dv, and closing the error with that law releases at a constant J / 2 - so the jerk you
# actually feel is HALF of J, not J. Deceleration converges on stock's A_CRUISE_MIN once
# |dv| > A_CRUISE_MIN**2 / J + TARGET_SPEED_DEADBAND: eco 9.8 m/s (22 mph), normal 6.0 (13), sport 4.0 (9).
# Beyond that every profile uses stock authority. This is the only knob that separates the profiles for
# small and medium speed changes, and the only one that makes them differ on the brake side at all.
COMFORT_JERK = {
  AccelProfile.eco:    0.15,
  AccelProfile.normal: 0.25,
  AccelProfile.sport:  0.38,
}
TARGET_SPEED_DEADBAND = 0.2  # m/s, errors below this are not worth a pedal input
# Authority needed to actually break away from a stop. Without this, sqrt(J * dv) toward a small target
# (creeping in traffic) commands less than the powertrain's breakaway acceleration and the car never moves.
LAUNCH_FLOOR_BREAKPOINTS = [1.0, 3.0]  # m/s
LAUNCH_FLOOR_VALUES = [1.2, 0.0]  # m/s^2


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

  def get_max_accel(self, v_ego: float) -> float:
    """Speed-scheduled acceleration authority: the hardest this profile will ever pull at this speed."""
    return float(np.interp(max(0.0, v_ego), MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES[self._profile]))

  def get_comfort_accel(self, v_ego: float, v_target: float) -> float:
    """Acceleration that closes v_target - v_ego over a human-length maneuver.

    Deliberately unbounded below: stock's own clip to A_CRUISE_MIN is what guarantees a large speed drop is
    never braked more gently than stock, so adding a decel cap here would only ever brake less than stock.
    """
    v_ego = max(0.0, v_ego)
    speed_error = v_target - v_ego
    error = max(0.0, abs(speed_error) - TARGET_SPEED_DEADBAND)
    if not error > 0.0:  # also short-circuits a non-finite target
      return 0.0

    # The linear term keeps the slope finite at the deadband edge, where sqrt() is vertical: without it a
    # 0.201 m/s error asks for 0.019 m/s^2 and a 0.21 m/s error asks for 0.06, a wall right where the loop
    # sits when settled. It binds below error == COMFORT_JERK and the sqrt binds above it.
    accel = min(error, float(np.sqrt(COMFORT_JERK[self._profile] * error)))
    if speed_error < 0.0:
      return -accel

    launch = float(np.interp(v_ego, LAUNCH_FLOOR_BREAKPOINTS, LAUNCH_FLOOR_VALUES))
    return min(max(accel, launch), error, self.get_max_accel(v_ego))

  def get_cruise_target(self, v_ego: float, v_target: float) -> float:
    """Virtual set speed whose stock 1 s proportional response equals the comfort acceleration.

    v_target <= 0 is how force_decel reaches this hook (the stock planner zeroes v_cruise), so it must pass
    through untouched.
    """
    if not np.isfinite(v_target) or v_target <= 0.0:
      return v_target

    return float(v_ego + self.get_comfort_accel(v_ego, v_target))

  def get_lead_departure_accel(self, v_ego: float, v_lead: float, a_lead: float, mpc_accel: float) -> float:
    values = (v_ego, v_lead, a_lead, mpc_accel)
    if not self._enabled or not all(np.isfinite(value) for value in values) or a_lead < 0.0 or mpc_accel < 0.0 or v_lead <= v_ego:
      return mpc_accel

    return max(mpc_accel, self.get_comfort_accel(v_ego, v_lead))
