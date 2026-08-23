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
  AccelController, AccelProfile, CATCHUP_ACCEL_SCALE, CATCHUP_ERROR_BREAKPOINTS, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES,
  TARGET_SPEED_APPROACH_FULL_SPEED, TARGET_SPEED_APPROACH_GAIN, TARGET_SPEED_APPROACH_MIN_SPEED, TARGET_SPEED_APPROACH_WINDOW,
  TARGET_SPEED_DEADBAND,
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

  def test_normal_stays_below_stock(self):
    controller = self.set_profile(AccelProfile.normal)
    for speed in np.linspace(0.0, 55.0, 551):
      assert controller.get_max_accel(speed) <= get_max_accel(speed) + 1e-12

  def test_profiles_taper_below_stock_at_road_speed(self):
    for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport):
      controller = self.set_profile(profile)
      for speed in np.linspace(8.0, 40.0, 321):
        assert controller.get_max_accel(speed) < get_max_accel(speed)

  def test_launch_caps_stay_close_to_stock(self):
    for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport):
      controller = self.set_profile(profile)
      for speed in np.linspace(0.0, 3.0, 61):
        assert controller.get_max_accel(speed) >= 0.9 * get_max_accel(speed)

  def test_eco_keeps_useful_road_speed_acceleration(self):
    controller = self.set_profile(AccelProfile.eco)
    for speed in np.linspace(8.0, 40.0, 321):
      assert controller.get_max_accel(speed) >= 0.35 * get_max_accel(speed) - 1e-12

  def test_profile_caps_drop_quickly_after_launch(self):
    for values in MAX_ACCEL_PROFILES.values():
      assert values[2] <= 0.7 * values[1]
      assert values[3] <= 0.5 * values[1]

  def test_sport_stays_below_reported_route_acceleration(self):
    controller = self.set_profile(AccelProfile.sport)
    route_samples = ((8.1, 0.877), (12.0, 0.858), (15.5, 0.802), (18.8, 0.750), (21.7, 0.654))
    for speed, recorded_accel in route_samples:
      assert controller.get_max_accel(speed) <= 0.85 * recorded_accel

  def test_positive_catchup_limit_is_continuous_and_monotonic(self):
    controller = self.set_profile(AccelProfile.normal)
    v_ego = 20.0
    max_accel = controller.get_max_accel(v_ego)
    errors = np.linspace(1e-4, 8.0, 321)
    accel_limits = np.asarray([controller.get_max_accel(v_ego, v_ego + error) for error in errors])

    assert np.all(np.isfinite(accel_limits))
    assert np.all(np.diff(accel_limits) >= -1e-12)
    assert np.all(accel_limits <= errors + 1e-12)
    assert np.all(accel_limits <= max_accel + 1e-12)
    assert controller.get_max_accel(v_ego, v_ego + TARGET_SPEED_DEADBAND) == 0.0
    assert np.isclose(controller.get_max_accel(v_ego, v_ego + CATCHUP_ERROR_BREAKPOINTS[-1]), max_accel)

    for error, scale in zip(CATCHUP_ERROR_BREAKPOINTS, CATCHUP_ACCEL_SCALE, strict=True):
      expected_accel = min(error, max_accel * scale)
      assert np.isclose(controller.get_max_accel(v_ego, v_ego + error), expected_accel)
      below = controller.get_max_accel(v_ego, v_ego + error - 1e-6)
      above = controller.get_max_accel(v_ego, v_ego + error + 1e-6)
      assert above >= below
      assert above - below < 1e-4

  def test_cruise_decel_settling_is_continuous(self):
    controller = self.set_profile(AccelProfile.normal)
    v_ego = 20.0
    near_error = 0.5
    target_blend = (TARGET_SPEED_APPROACH_WINDOW - near_error) / (TARGET_SPEED_APPROACH_WINDOW - TARGET_SPEED_DEADBAND)
    adjusted_error = near_error - TARGET_SPEED_DEADBAND * target_blend
    gain = 1.0 - (1.0 - TARGET_SPEED_APPROACH_GAIN) * target_blend
    expected_error = adjusted_error * gain

    assert np.isclose(controller.get_cruise_target(v_ego, v_ego - near_error), v_ego - expected_error)
    assert controller.get_cruise_target(v_ego, v_ego - TARGET_SPEED_DEADBAND) == v_ego
    assert controller.get_cruise_target(v_ego, v_ego - TARGET_SPEED_APPROACH_WINDOW) == v_ego - TARGET_SPEED_APPROACH_WINDOW

    errors = np.linspace(-5.0, 0.0, 201)
    shaped_errors = np.asarray([controller.get_cruise_target(v_ego, v_ego + error) - v_ego for error in errors])
    assert np.all(np.isfinite(shaped_errors))
    assert np.all(np.diff(shaped_errors) >= 0.0)
    assert np.all(np.abs(shaped_errors) <= np.abs(errors) + 1e-12)

  def test_catchup_limit_does_not_touch_launch(self):
    controller = self.set_profile(AccelProfile.sport)

    def command(speed: float, target: float, dynamic: bool = True) -> float:
      max_accel = controller.get_max_accel(speed, target) if dynamic else controller.get_max_accel(speed)
      return get_cruise_accel(False, target, speed, 0.0, 0.0, _fake_cp(), 10.0, 0.0, True, max_accel)

    for speed in np.linspace(0.0, TARGET_SPEED_APPROACH_MIN_SPEED, 101):
      for target in (0.0, 8.0, 30.0):
        assert command(speed, target) == command(speed, target, dynamic=False)

    below = command(TARGET_SPEED_APPROACH_MIN_SPEED - 1e-3, TARGET_SPEED_APPROACH_MIN_SPEED + 0.5 - 1e-3)
    at = command(TARGET_SPEED_APPROACH_MIN_SPEED, TARGET_SPEED_APPROACH_MIN_SPEED + 0.5)
    above = command(TARGET_SPEED_APPROACH_MIN_SPEED + 1e-3, TARGET_SPEED_APPROACH_MIN_SPEED + 0.5 + 1e-3)
    assert np.isclose(below, 0.5)
    assert np.isclose(at, 0.5)
    assert abs(above - at) < 1e-3

    below_full = command(TARGET_SPEED_APPROACH_FULL_SPEED - 1e-3, TARGET_SPEED_APPROACH_FULL_SPEED + 0.5 - 1e-3)
    at_full = command(TARGET_SPEED_APPROACH_FULL_SPEED, TARGET_SPEED_APPROACH_FULL_SPEED + 0.5)
    above_full = command(TARGET_SPEED_APPROACH_FULL_SPEED + 1e-3, TARGET_SPEED_APPROACH_FULL_SPEED + 0.5 + 1e-3)
    assert abs(below_full - at_full) < 1e-3
    assert abs(above_full - at_full) < 1e-3

  def test_cruise_target_deadband_removes_small_sign_flips(self):
    controller = self.set_profile(AccelProfile.normal)
    v_ego = 20.0
    errors = np.asarray([0.10, -0.10, 0.15, -0.15, 0.30, -0.30])
    commands = []
    for error in errors:
      raw_target = v_ego + error
      target = controller.get_cruise_target(v_ego, raw_target)
      max_accel = controller.get_max_accel(v_ego, raw_target)
      commands.append(get_cruise_accel(False, target, v_ego, 0.0, 0.0, _fake_cp(), 10.0, 0.0, True, max_accel))
    shaped = np.asarray(commands)

    assert np.count_nonzero(shaped[:4]) == 0
    assert shaped[4] > 0.0
    assert shaped[5] < 0.0
    assert np.all(np.abs(shaped) <= np.abs(errors) + 1e-12)

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

  def test_profiles_do_not_change_far_braking(self):
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

  def test_cruise_accel_tapers_before_target(self):
    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    self.params.put("AccelPersonality", AccelProfile.normal, block=True)
    controller = AccelController()
    speed = 20.0
    speed_errors = (4.0, 3.0, 2.0, 1.0, 0.5, TARGET_SPEED_DEADBAND)
    targets = [controller.get_cruise_target(speed, speed + error) for error in speed_errors]
    max_accels = [controller.get_max_accel(speed, speed + error) for error in speed_errors]
    accels = [get_cruise_accel(False, target, speed, 0.0, 0.0, _fake_cp(), 10.0, 0.0, True, max_accel)
              for target, max_accel in zip(targets, max_accels, strict=True)]

    assert np.isclose(accels[0], controller.get_max_accel(speed))
    assert accels[1] < controller.get_max_accel(speed)
    assert all(current > following for current, following in zip(accels, accels[1:], strict=False))
    assert accels[-1] == 0.0

  def test_disabled_leaves_stock_limit_active(self):
    planner = _bare_planner()
    for e2e in (False, True):
      assert planner.get_max_accel_override(5.0, 30.0, e2e=e2e) is None
      assert planner.get_cruise_target_override(20.0, 20.5, e2e=e2e) == 20.5
      assert planner.accel_controller_active is False

  def test_e2e_uses_enabled_profile(self):
    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    planner = _bare_planner()
    assert planner.get_max_accel_override(5.0, 30.0, e2e=True) == MAX_ACCEL_PROFILES[AccelProfile.normal][2]
    assert planner.accel_controller_active is True

  def test_enabled_acc_uses_python_native_telemetry_types(self):
    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    self.params.put("AccelPersonality", AccelProfile.sport, block=True)
    planner = _bare_planner()
    assert planner.get_max_accel_override(5.0, 30.0, e2e=False) == MAX_ACCEL_PROFILES[AccelProfile.sport][2]
    assert type(planner.accel_controller_active) is bool
    assert type(planner.accel_controller.is_enabled()) is bool
    assert type(planner.accel_controller.profile) is int

  def test_normal_profile_uses_tuned_limit(self):
    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    self.params.put("AccelPersonality", AccelProfile.normal, block=True)
    planner = _bare_planner()
    assert planner.get_max_accel_override(5.0, 30.0, e2e=False) == MAX_ACCEL_PROFILES[AccelProfile.normal][2]
    assert planner.accel_controller_active is True

  def test_planner_applies_cruise_settling_only_when_safe(self):
    from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource

    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    planner = _bare_planner()
    expected_limit = planner.accel_controller.get_max_accel(20.0, 20.5)
    expected_decel_target = planner.accel_controller.get_cruise_target(20.5, 20.0)
    assert planner.get_cruise_target_override(20.0, 20.5, e2e=False) == 20.5
    assert np.isclose(planner.get_cruise_target_override(20.5, 20.0, e2e=False), expected_decel_target)
    assert np.isclose(planner.get_max_accel_override(20.0, 20.5, e2e=False), expected_limit)

    planner.allow_throttle = False
    assert planner.get_cruise_target_override(20.0, 20.5, e2e=False) == 20.5
    assert planner.get_max_accel_override(20.0, 20.5, e2e=False) is None
    assert planner.accel_controller_active is False
    assert planner.get_cruise_target_override(20.0, 20.5, e2e=True) == 20.5
    assert np.isclose(planner.get_max_accel_override(20.0, 20.5, e2e=True), expected_limit)
    assert planner.accel_controller_active is True

    planner.source = LongitudinalPlanSource.sccVision
    assert planner.get_cruise_target_override(20.0, 20.5, e2e=True) == 20.5
    assert planner.get_max_accel_override(20.0, 20.5, e2e=True) == planner.accel_controller.get_max_accel(20.0)


def _fake_cp():
  class CP:
    steerRatio = 15.0
    wheelbase = 2.7

  return CP()


def _bare_planner():
  from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, LongitudinalPlanSource

  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.accel_controller = AccelController()
  planner.accel_controller_active = False
  planner.allow_throttle = True
  planner.source = LongitudinalPlanSource.cruise
  return planner
