"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  A_CRUISE_MAX_BP, A_CRUISE_MIN, J_CRUISE_VALS, get_cruise_accel, get_max_accel,
)
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.accel_controller import (
  AccelController, AccelProfile, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES,
)


class TestAccelController(OpenpilotTestCase):
  def setUp(self):
    self.params = Params()
    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    self.params.put("AccelPersonality", AccelProfile.normal, block=True)

  def set_profile(self, profile: int) -> AccelController:
    self.params.put("AccelPersonality", profile, block=True)
    return AccelController()

  def test_table_breakpoints(self):
    for profile, values in MAX_ACCEL_PROFILES.items():
      controller = self.set_profile(profile)
      for speed, expected in zip(MAX_ACCEL_BREAKPOINTS, values, strict=True):
        assert controller.get_max_accel(speed) == expected

  def test_profile_ordering_and_bounds(self):
    controllers = {
      AccelProfile.eco: self.set_profile(AccelProfile.eco),
      AccelProfile.normal: self.set_profile(AccelProfile.normal),
      AccelProfile.sport: self.set_profile(AccelProfile.sport),
    }
    previous = {profile: float("inf") for profile in controllers}

    for speed in np.linspace(0.0, 55.0, 551):
      values = {profile: controller.get_max_accel(speed) for profile, controller in controllers.items()}
      assert 0.0 <= values[AccelProfile.eco] <= values[AccelProfile.normal] <= values[AccelProfile.sport] <= 2.0
      for profile, value in values.items():
        assert value <= previous[profile]
        previous[profile] = value

  def test_normal_matches_stock(self):
    controller = self.set_profile(AccelProfile.normal)
    for speed in np.linspace(0.0, 55.0, 551):
      assert np.isclose(controller.get_max_accel(speed), get_max_accel(speed), rtol=0.0, atol=1e-12)

  def test_eco_keeps_useful_road_speed_acceleration(self):
    controller = self.set_profile(AccelProfile.eco)
    for speed in np.linspace(8.0, 40.0, 321):
      assert controller.get_max_accel(speed) >= 0.75 * get_max_accel(speed) - 1e-12

  def test_negative_speed_uses_standstill_value(self):
    controller = self.set_profile(AccelProfile.sport)
    assert controller.get_max_accel(-1.0) == MAX_ACCEL_PROFILES[AccelProfile.sport][0]

  def test_profile_change_has_no_controller_filter(self):
    controller = self.set_profile(AccelProfile.normal)
    self.params.put("AccelPersonality", AccelProfile.sport, block=True)
    controller.frame = int(1.0 / DT_MDL) - 1
    controller.update()
    assert controller.get_max_accel(8.0) == MAX_ACCEL_PROFILES[AccelProfile.sport][3]

  def test_params_refresh_once_per_second(self):
    controller = self.set_profile(AccelProfile.normal)
    self.params.put("AccelPersonality", AccelProfile.sport, block=True)
    controller.update()
    assert controller.profile == AccelProfile.normal
    controller.frame = int(1.0 / DT_MDL) - 1
    controller.update()
    assert controller.profile == AccelProfile.sport

  def test_enabled_param_refresh(self):
    controller = self.set_profile(AccelProfile.normal)
    self.params.put_bool("AccelPersonalityEnabled", False, block=True)
    controller.frame = int(1.0 / DT_MDL) - 1
    controller.update()
    assert not controller.is_enabled()


class TestPlannerIntegration(OpenpilotTestCase):
  def setUp(self):
    self.params = Params()
    self.params.put_bool("AccelPersonalityEnabled", False, block=True)

  def test_none_override_matches_stock(self):
    for e2e in (False, True):
      for allow_throttle in (False, True):
        args = (e2e, 30.0, 12.0, 0.2, 4.0, _fake_cp(), DT_MDL, -0.3, allow_throttle)
        assert get_cruise_accel(*args) == get_cruise_accel(*args, max_accel_override=None)

  def test_profiles_do_not_change_braking(self):
    args = (False, 0.0, 20.0, 0.0, 0.0, _fake_cp(), 10.0, -0.3, True)
    stock = get_cruise_accel(*args)
    assert stock == A_CRUISE_MIN
    for profile_values in MAX_ACCEL_PROFILES.values():
      assert get_cruise_accel(*args, max_accel_override=profile_values[0]) == stock

  def test_stock_jerk_limit_still_owns_smoothing(self):
    speed = 8.0
    sport_limit = np.interp(speed, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES[AccelProfile.sport])
    target = get_cruise_accel(False, 30.0, speed, 0.0, 0.0, _fake_cp(), DT_MDL, 0.0, True, sport_limit)
    jerk_limit = np.interp(speed, A_CRUISE_MAX_BP, J_CRUISE_VALS) * DT_MDL
    assert np.isclose(target, jerk_limit)

  def test_disabled_and_e2e_leave_stock_limit_active(self):
    planner = _bare_planner()
    assert planner.get_max_accel_override(5.0, e2e=False) is None
    assert planner.accel_controller_active is False

    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    planner = _bare_planner()
    assert planner.get_max_accel_override(5.0, e2e=True) is None
    assert planner.accel_controller_active is False

  def test_enabled_acc_uses_python_native_telemetry_types(self):
    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    self.params.put("AccelPersonality", AccelProfile.sport, block=True)
    planner = _bare_planner()
    assert planner.get_max_accel_override(5.0, e2e=False) == MAX_ACCEL_PROFILES[AccelProfile.sport][2]
    assert type(planner.accel_controller_active) is bool
    assert type(planner.accel_controller.is_enabled()) is bool
    assert type(planner.accel_controller.profile) is int

  def test_normal_profile_uses_exact_stock_path(self):
    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    self.params.put("AccelPersonality", AccelProfile.normal, block=True)
    planner = _bare_planner()
    assert planner.get_max_accel_override(5.0, e2e=False) is None
    assert planner.accel_controller_active is False


def _fake_cp():
  class CP:
    steerRatio = 15.0
    wheelbase = 2.7

  return CP()


def _bare_planner():
  from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.accel_controller = AccelController()
  planner.accel_controller_active = False
  return planner
