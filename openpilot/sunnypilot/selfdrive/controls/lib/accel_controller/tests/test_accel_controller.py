"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from opendbc.car.interfaces import ACCEL_MAX
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS, A_CRUISE_MIN, J_CRUISE_VALS, get_cruise_accel,
)
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.accel_controller import (
  AccelController, AccelProfile, COMFORT_JERK, LAUNCH_FLOOR_VALUES, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES,
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

  def test_profiles_stay_within_openpilot_accel_max(self):
    for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport):
      controller = self.set_profile(profile)
      for speed in np.linspace(0.0, 55.0, 551):
        assert controller.get_max_accel(speed) <= ACCEL_MAX

  def test_decel_reaches_stock_parity_above_profile_crossover(self):
    # The comfort law is deliberately unbounded below - stock's own clip to A_CRUISE_MIN owns the decel floor,
    # which means every profile converges on stock authority once the law's peak reaches it, at
    # |dv| = A_CRUISE_MIN**2 / J + deadband. Below that the law is gentler on purpose; at or above it the
    # command must be bit-identical to stock. A decel cap of any kind breaks the second half, and lowering J
    # pushes the crossover out, so this test is what keeps "never brake less than stock" honest.
    for profile, jerk in COMFORT_JERK.items():
      controller = self.set_profile(profile)
      crossover = A_CRUISE_MIN ** 2 / jerk + TARGET_SPEED_DEADBAND
      for v_ego in (20.0, 29.0, 40.0):
        cap = controller.get_max_accel(v_ego)
        args = (v_ego, A_CRUISE_MIN, 0.0, _fake_cp(), 10.0, 0.0, True, cap)

        for drop in (crossover * 1.001, crossover + 1.0, crossover + 10.0):
          v_target = v_ego - drop
          if v_target <= 0.1:
            continue
          shaped = controller.get_cruise_target(v_ego, v_target)
          assert get_cruise_accel(False, shaped, *args) == get_cruise_accel(False, v_target, *args), (profile, v_ego, drop)

        # Below the crossover the law softens, but only ever down to the constant-jerk value.
        for drop in (crossover * 0.5, crossover * 0.25):
          v_target = v_ego - drop
          expected = -np.sqrt(jerk * (drop - TARGET_SPEED_DEADBAND))
          assert np.isclose(controller.get_comfort_accel(v_ego, v_target), expected), (profile, drop)
          assert expected > A_CRUISE_MIN, (profile, drop)

  def test_profiles_have_material_separation(self):
    controllers = [self.set_profile(profile) for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport)]
    for speed in MAX_ACCEL_BREAKPOINTS:
      eco, normal, sport = (controller.get_max_accel(speed) for controller in controllers)
      assert normal - eco >= 0.1 - 1e-12
      assert sport - normal >= 0.1 - 1e-12
    for speed in MAX_ACCEL_BREAKPOINTS[1:-1]:
      assert controllers[2].get_max_accel(speed) - controllers[0].get_max_accel(speed) >= 0.3 - 1e-12

  def test_profiles_keep_usable_road_speed_acceleration(self):
    # A previous revision had eco at 0.20 m/s^2 at 40 m/s. 1% of road grade costs 0.098 m/s^2 of gravity, so
    # that profile cannot hold speed on anything steeper than ~2% and can never recover once it bleeds off.
    # This is a LOWER bound on purpose: the tapered upper bounds it replaces let highway accel go to zero.
    controllers = {profile: self.set_profile(profile) for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport)}
    for speed in np.linspace(8.0, 40.0, 321):
      stock = float(np.interp(speed, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS))
      values = {profile: controller.get_max_accel(speed) for profile, controller in controllers.items()}
      # 0.35 m/s^2 holds a 3% grade; the fractions keep merges and passes usable.
      assert values[AccelProfile.eco] >= max(0.35, 0.60 * stock), speed
      assert values[AccelProfile.normal] >= 0.80 * stock, speed
      assert values[AccelProfile.sport] >= stock, speed

  def test_eco_never_exceeds_stock(self):
    controller = self.set_profile(AccelProfile.eco)
    for speed in np.linspace(0.0, 55.0, 551):
      assert controller.get_max_accel(speed) <= float(np.interp(speed, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)) + 1e-12, speed

  def test_comfort_profile_caps_taper_after_launch(self):
    for profile in (AccelProfile.eco, AccelProfile.normal):
      values = MAX_ACCEL_PROFILES[profile]
      assert values[3] <= 0.55 * values[0]

  def test_sport_uses_openpilot_accel_max_at_launch(self):
    controller = self.set_profile(AccelProfile.sport)
    assert controller.get_max_accel(0.0) == ACCEL_MAX
    assert all(controller.get_max_accel(speed) <= ACCEL_MAX for speed in np.linspace(0.0, 55.0, 551))

  def test_comfort_accel_is_continuous_and_monotonic(self):
    for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport):
      controller = self.set_profile(profile)
      for v_ego in (0.0, 5.0, 20.0, 40.0):
        envelope = controller.get_max_accel(v_ego)
        errors = np.linspace(-12.0, 12.0, 961)
        accels = np.asarray([controller.get_comfort_accel(v_ego, v_ego + error) for error in errors])

        assert np.all(np.isfinite(accels))
        assert np.all(np.diff(accels) >= -1e-9), (profile, v_ego)  # monotone in the signed error
        assert np.all(np.abs(accels) <= np.abs(errors) + 1e-12)  # never asks for more than the error itself
        assert np.all(accels <= envelope + 1e-12)  # the accel side is envelope-bounded
        assert np.all(accels[errors < 0.0] <= 0.0) and np.all(accels[errors > 0.0] >= 0.0)  # sign is correct

  def test_comfort_accel_matches_the_constant_jerk_closed_form(self):
    # The law is min(error, sqrt(J * error)) on the deadbanded error. The linear term wins below error == J
    # and keeps the slope finite at the deadband edge, where sqrt is vertical; the sqrt term wins above it.
    # Asserted in closed form so nobody "optimises" the min() away.
    for profile, jerk in COMFORT_JERK.items():
      controller = self.set_profile(profile)
      v_ego = 25.0
      for error in (jerk / 2.0, jerk, jerk * 2.0, 1.0, 3.0, 9.0):
        expected = min(error, np.sqrt(jerk * error))
        assert np.isclose(controller.get_comfort_accel(v_ego, v_ego - error - TARGET_SPEED_DEADBAND), -expected), (profile, error)

      # Continuity across the crossover.
      crossover = jerk + TARGET_SPEED_DEADBAND
      below = controller.get_comfort_accel(v_ego, v_ego - crossover + 1e-6)
      above = controller.get_comfort_accel(v_ego, v_ego - crossover - 1e-6)
      assert abs(above - below) < 1e-5, profile

  def test_comfort_accel_deadband_is_exact(self):
    for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport):
      controller = self.set_profile(profile)
      for error in (0.0, 0.1, TARGET_SPEED_DEADBAND, -0.1, -TARGET_SPEED_DEADBAND):
        assert controller.get_comfort_accel(20.0, 20.0 + error) == 0.0, (profile, error)
      assert controller.get_comfort_accel(20.0, float("nan")) == 0.0

  def test_launch_floor_guarantees_breakaway_authority(self):
    # sqrt(J * dv) toward a small target is below the powertrain's breakaway acceleration, so without a floor
    # the car would sit still at a green light behind a creeping lead. The floor must not exceed the error.
    for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport):
      controller = self.set_profile(profile)
      for target in np.linspace(1.0, 30.0, 59):
        accel = controller.get_comfort_accel(0.0, target)
        assert accel >= min(LAUNCH_FLOOR_VALUES[0], target - TARGET_SPEED_DEADBAND) - 1e-12, (profile, target)
        assert accel <= target - TARGET_SPEED_DEADBAND + 1e-12, (profile, target)

  def test_cruise_target_passes_through_degenerate_targets(self):
    # v_target <= 0 is how force_decel reaches this hook (the stock planner zeroes v_cruise), so it must
    # never be reshaped. Non-finite targets must not propagate NaN into the cruise accel either.
    controller = self.set_profile(AccelProfile.normal)
    v_ego = 20.0

    assert controller.get_cruise_target(v_ego, 0.0) == 0.0
    assert controller.get_cruise_target(v_ego, -1.0) == -1.0
    assert np.isnan(controller.get_cruise_target(v_ego, float("nan")))
    assert controller.get_cruise_target(v_ego, float("inf")) == float("inf")

  def test_lead_departure_accel_is_bounded_by_lead_speed_and_profile(self):
    # Chasing a lead crawling at 2 m/s used to command the full standstill envelope. The comfort law bounds
    # it by the closing rate and the launch floor instead, which is enough to break away and no more.
    controller = self.set_profile(AccelProfile.eco)
    assert np.isclose(controller.get_lead_departure_accel(0.0, 0.7, 0.1, 0.1), 0.5)
    assert np.isclose(controller.get_lead_departure_accel(0.0, 2.0, 0.1, 0.1), LAUNCH_FLOOR_VALUES[0])

    for v_lead in np.linspace(0.3, 6.0, 58):
      accel = controller.get_lead_departure_accel(0.0, v_lead, 0.1, -1.0)
      assert accel <= v_lead + 1e-12, v_lead  # never faster than the lead is pulling away

  def test_lead_departure_accel_preserves_non_acceleration_cases(self):
    controller = self.set_profile(AccelProfile.normal)
    assert controller.get_lead_departure_accel(0.0, 1.0, 0.1, -0.1) == -0.1
    assert controller.get_lead_departure_accel(0.0, 1.0, -0.1, 0.1) == 0.1
    assert controller.get_lead_departure_accel(1.0, 1.0, 0.1, 0.1) == 0.1
    assert controller.get_lead_departure_accel(0.0, float("nan"), 0.1, 0.1) == 0.1

    self.params.put_bool("AccelPersonalityEnabled", False, block=True)
    assert AccelController().get_lead_departure_accel(0.0, 1.0, 0.1, 0.1) == 0.1

  def test_composed_cruise_command_is_lipschitz_in_speed(self):
    # The shaped target and the ceiling are both functions of v_ego, so a discontinuity in either shows up as
    # a step in the commanded acceleration. dt=10 makes the stock jerk limiter a no-op so nothing can hide a
    # step. The law's steepest branch is the linear one, whose slope in v_ego is exactly 1, so the bound is
    # the sample spacing - anything above that is a genuine jump.
    for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport):
      controller = self.set_profile(profile)

      def command(speed: float, target: float, ctrl: AccelController = controller) -> float:
        shaped = ctrl.get_cruise_target(speed, target)
        return get_cruise_accel(False, shaped, speed, 0.0, 0.0, _fake_cp(), 10.0, 0.0, True, ctrl.get_max_accel(speed))

      for target in (2.0, 8.0, 30.0):
        speeds = np.linspace(0.0, min(target + 4.0, 45.0), 451)
        spacing = float(speeds[1] - speeds[0])
        commands = np.asarray([command(speed, target) for speed in speeds])
        assert np.all(np.isfinite(commands)), (profile, target)
        assert np.all(np.abs(np.diff(commands)) <= spacing * 1.05 + 1e-9), (profile, target)

  def test_cruise_target_deadband_removes_small_sign_flips(self):
    controller = self.set_profile(AccelProfile.normal)
    v_ego = 20.0
    errors = np.asarray([0.10, -0.10, 0.15, -0.15, 0.30, -0.30])
    commands = []
    for error in errors:
      raw_target = v_ego + error
      target = controller.get_cruise_target(v_ego, raw_target)
      max_accel = controller.get_max_accel(v_ego)
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
    index = MAX_ACCEL_BREAKPOINTS.index(10.0)
    assert controller.get_max_accel(10.0) == MAX_ACCEL_PROFILES[AccelProfile.sport][index]

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

  def test_stock_cruise_law_is_unit_time_constant(self):
    # get_cruise_target reshapes a SPEED, but the value it returns is v_ego + a_desired. That only commands
    # a_desired because stock's cruise law reads a speed error in m/s directly as an acceleration in m/s^2:
    # target_accel = clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel). It is a proportional law with a 1.0 s
    # time constant, and that constant is what the fork's target shaping is inverting.
    #
    # dt=100 with a_cruise_prev=a makes the stock jerk limiter a no-op; angle_steers=0 and allow_throttle=True
    # make the turn and coast limits no-ops; |a| <= 1.2 stays off the A_CRUISE_MIN clip. So this isolates the
    # one line the inversion depends on.
    #
    # If this fails, openpilot changed the cruise law. Re-derive the time constant before shipping anything
    # that relies on the inversion - the failure is otherwise silent, just wrong feel.
    for v_ego in (5.0, 20.0, 35.0):
      for a in (-1.2, -0.5, -0.1, 0.0, 0.1, 0.5, 1.2):
        got = get_cruise_accel(False, v_ego + a, v_ego, a, 0.0, _fake_cp(), 100.0, 0.0, True, 2.0)
        assert np.isclose(got, a), (v_ego, a, got)

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
    max_accels = [controller.get_max_accel(speed) for _ in speed_errors]
    accels = [get_cruise_accel(False, target, speed, 0.0, 0.0, _fake_cp(), 10.0, 0.0, True, max_accel)
              for target, max_accel in zip(targets, max_accels, strict=True)]

    # The taper now comes from the shaped target rather than from the ceiling. Large errors saturate on the
    # envelope (so they tie), and the law takes over below it - monotone throughout, landing exactly on zero.
    envelope = controller.get_max_accel(speed)
    assert accels[0] <= envelope + 1e-12
    assert all(current >= following for current, following in zip(accels, accels[1:], strict=False))
    off_envelope = [accel for accel in accels if accel < envelope - 1e-12]
    assert len(off_envelope) >= 3
    assert all(current > following for current, following in zip(off_envelope, off_envelope[1:], strict=False))
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
    expected = np.interp(5.0, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES[AccelProfile.normal])
    assert planner.get_max_accel_override(5.0, 30.0, e2e=True) == expected
    assert planner.accel_controller_active is True

  def test_enabled_acc_uses_python_native_telemetry_types(self):
    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    self.params.put("AccelPersonality", AccelProfile.sport, block=True)
    planner = _bare_planner()
    expected = np.interp(5.0, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES[AccelProfile.sport])
    assert planner.get_max_accel_override(5.0, 30.0, e2e=False) == expected
    assert type(planner.accel_controller_active) is bool
    assert type(planner.accel_controller.is_enabled()) is bool
    assert type(planner.accel_controller.profile) is int

  def test_normal_profile_uses_tuned_limit(self):
    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    self.params.put("AccelPersonality", AccelProfile.normal, block=True)
    planner = _bare_planner()
    expected = np.interp(5.0, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES[AccelProfile.normal])
    assert planner.get_max_accel_override(5.0, 30.0, e2e=False) == expected
    assert planner.accel_controller_active is True

  def test_planner_applies_cruise_settling_only_when_safe(self):
    from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource

    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    planner = _bare_planner()
    expected_limit = planner.accel_controller.get_max_accel(20.0)
    expected_accel_target = planner.accel_controller.get_cruise_target(20.0, 20.5)
    expected_decel_target = planner.accel_controller.get_cruise_target(20.5, 20.0)
    assert np.isclose(planner.get_cruise_target_override(20.0, 20.5, e2e=False), expected_accel_target)
    assert np.isclose(planner.get_cruise_target_override(20.5, 20.0, e2e=False), expected_decel_target)
    assert np.isclose(planner.get_max_accel_override(20.0, 20.5, e2e=False), expected_limit)

    # Without throttle authority the accelerating half must not be shaped at all - shaping it would command
    # gas the throttle-intent gate has just suppressed. The decelerating half is still shaped.
    planner.allow_throttle = False
    assert planner.get_cruise_target_override(20.0, 20.5, e2e=False) == 20.5
    assert np.isclose(planner.get_cruise_target_override(20.5, 20.0, e2e=False), expected_decel_target)
    assert planner.get_max_accel_override(20.0, 20.5, e2e=False) is None
    assert planner.accel_controller_active is False
    assert np.isclose(planner.get_cruise_target_override(20.0, 20.5, e2e=True), expected_accel_target)
    assert np.isclose(planner.get_max_accel_override(20.0, 20.5, e2e=True), expected_limit)
    assert planner.accel_controller_active is True

    planner.source = LongitudinalPlanSource.sccVision
    assert planner.get_cruise_target_override(20.0, 20.5, e2e=True) == 20.5
    assert np.isclose(planner.get_max_accel_override(20.0, 20.5, e2e=True), planner.accel_controller.get_max_accel(20.0))

  def test_scc_and_sla_sources_are_never_softened(self):
    # SCC derives its target from a lateral-acceleration budget and the curve is at a fixed distance, so
    # stretching the maneuver means arriving faster than the budget allows. SLA has the same shape of
    # constraint against the sign's position. Only a pure-preference target may be reshaped.
    # This test is what stops a future change from adding either to COMFORT_SOURCES.
    from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import COMFORT_SOURCES, LongitudinalPlanSource

    assert COMFORT_SOURCES == (LongitudinalPlanSource.cruise,)

    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    planner = _bare_planner()
    speed = 29.0

    for source in (LongitudinalPlanSource.sccVision, LongitudinalPlanSource.sccMap, LongitudinalPlanSource.speedLimitAssist):
      planner.source = source
      for target in (20.0, 25.0, 28.5, 30.0, 33.0):
        assert planner.get_cruise_target_override(speed, target, e2e=False) == target, (source, target)
      # The ceiling is speed-scheduled only, so it is the same for every source.
      assert np.isclose(planner.get_max_accel_override(speed, 33.0, e2e=False), planner.accel_controller.get_max_accel(speed))

  def test_force_decel_is_never_softened(self):
    # forceDecel is captured explicitly in update(). The stock planner also zeroes v_cruise, and the
    # v_target <= 0 guard catches that, but relying on an exact float reaching this class is too fragile a
    # guard for a path that must never be softened - so both are asserted.
    self.params.put_bool("AccelPersonalityEnabled", True, block=True)
    planner = _bare_planner()
    speed = 29.0

    planner.force_decel = True
    for target in (0.0, 5.0, 20.0, 28.9, 33.0):
      assert planner.get_cruise_target_override(speed, target, e2e=False) == target, target

    planner.force_decel = False
    assert planner.get_cruise_target_override(speed, 0.0, e2e=False) == 0.0
    assert planner.get_cruise_target_override(speed, 20.0, e2e=False) != 20.0


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
  planner.force_decel = False
  planner.source = LongitudinalPlanSource.cruise
  return planner
