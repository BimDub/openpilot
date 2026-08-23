"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from collections.abc import Callable

import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.longitudinal_planner import A_CRUISE_MAX_BP, J_CRUISE_VALS, get_cruise_accel
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource, T_IDXS as T_IDXS_MPC
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.accel_controller import (
  AccelController, AccelProfile, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES, TARGET_SPEED_DEADBAND,
)
from openpilot.sunnypilot.selfdrive.test.longitudinal_maneuvers.plant import PlantSP


class CarParams:
  steerRatio = 15.0
  wheelbase = 2.7


def _set_mpc_acceleration(plant: PlantSP, acceleration: float = 2.0) -> None:
  def update(_radar_state, **_kwargs):
    mpc = plant.planner.mpc
    mpc.source = LongitudinalPlanSource.lead0
    mpc.v_solution[:] = mpc.x0[1] + acceleration * T_IDXS_MPC
    mpc.a_solution.fill(acceleration)
    mpc.j_solution.fill(0.0)

  plant.planner.mpc.update = update


def run_profile(profile: int, *, enabled: bool = True, speed: float = 0.0, v_cruise: float = 30.0,
                v_cruise_fn: Callable[[int], float] | None = None, e2e: bool = False, steps: int = 120):
  params = Params()
  params.put_bool("AccelPersonalityEnabled", enabled, block=True)
  params.put("AccelPersonality", profile, block=True)
  controller = AccelController()

  accel = 0.0
  rows = []
  for frame in range(steps):
    target_speed = v_cruise if v_cruise_fn is None else v_cruise_fn(frame)
    use_profile = controller.is_enabled()
    cruise_target = controller.get_cruise_target(speed, target_speed) if use_profile else target_speed
    max_accel_override = controller.get_max_accel(speed, target_speed) if use_profile else None
    accel = get_cruise_accel(e2e, cruise_target, speed, accel, 0.0, CarParams(), DT_MDL, 2.0, True, max_accel_override)
    speed = max(0.0, speed + accel * DT_MDL)
    rows.append((speed, accel, should_stop(speed, accel)))
  return rows


class TestAccelControllerClosedLoop(OpenpilotTestCase):
  def test_blended_positive_model_request_uses_profile_cruise_cap(self):
    params = Params()
    params.put_bool("DynamicExperimentalControl", False, block=True)
    params.put_bool("AccelPersonalityEnabled", True, block=True)
    params.put("AccelPersonality", AccelProfile.eco, block=True)

    def request_acceleration(_current_time: float, _speed: float, _acceleration: float) -> tuple[float, bool]:
      return 2.0, False

    plant = PlantSP(speed=15.0, e2e=True, model_action_fn=request_acceleration)
    _set_mpc_acceleration(plant)
    results = [plant.step(v_cruise=35.0) for _ in range(20)]
    settled = results[-1]
    eco_limit = float(np.interp(settled["published_v_ego"], MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES[AccelProfile.eco]))

    self.assertTrue(settled["controller_active"])
    self.assertEqual(settled["mpc_source"], LongitudinalPlanSource.cruise)
    self.assertAlmostEqual(settled["a_target"], eco_limit, delta=0.01)
    self.assertLess(settled["a_target"], settled["model_action"]["desiredAcceleration"])

  def test_blended_profile_does_not_change_model_braking(self):
    params = Params()
    params.put_bool("DynamicExperimentalControl", False, block=True)
    params.put("AccelPersonality", AccelProfile.eco, block=True)

    def request_braking(_current_time: float, _speed: float, _acceleration: float) -> tuple[float, bool]:
      return -0.8, False

    traces = {}
    for enabled in (False, True):
      params.put_bool("AccelPersonalityEnabled", enabled, block=True)
      plant = PlantSP(speed=20.0, e2e=True, model_action_fn=request_braking)
      _set_mpc_acceleration(plant)
      traces[enabled] = [plant.step(v_cruise=30.0) for _ in range(10)]

    self.assertTrue(all(row["mpc_source"] == LongitudinalPlanSource.e2e for row in traces[True]))
    self.assertTrue(all(row["controller_active"] for row in traces[True]))
    self.assertTrue(all(not row["controller_active"] for row in traces[False]))
    for key in ("a_target", "should_stop", "mpc_source"):
      self.assertEqual([row[key] for row in traces[True]], [row[key] for row in traces[False]])

  def test_profile_does_not_change_model_stop_request(self):
    params = Params()
    params.put_bool("DynamicExperimentalControl", False, block=True)
    params.put("AccelPersonality", AccelProfile.eco, block=True)

    def request_stop(_current_time: float, _speed: float, _acceleration: float) -> tuple[float, bool]:
      return -0.8, True

    traces = {}
    for enabled in (False, True):
      params.put_bool("AccelPersonalityEnabled", enabled, block=True)
      plant = PlantSP(speed=1.0, e2e=True, model_action_fn=request_stop)
      _set_mpc_acceleration(plant)
      traces[enabled] = [plant.step(v_cruise=30.0) for _ in range(10)]

    for key in ("a_target", "should_stop", "mpc_source"):
      self.assertEqual([row[key] for row in traces[True]], [row[key] for row in traces[False]])

  def test_profile_does_not_change_lead_braking(self):
    params = Params()
    params.put_bool("DynamicExperimentalControl", False, block=True)
    params.put("AccelPersonality", AccelProfile.eco, block=True)

    traces = {}
    for enabled in (False, True):
      params.put_bool("AccelPersonalityEnabled", enabled, block=True)
      plant = PlantSP(speed=20.0)
      _set_mpc_acceleration(plant, -0.8)
      traces[enabled] = [plant.step(v_cruise=30.0) for _ in range(10)]

    self.assertTrue(all(row["mpc_source"] == LongitudinalPlanSource.lead0 for row in traces[True]))
    for key in ("a_target", "should_stop", "mpc_source"):
      self.assertEqual([row[key] for row in traces[True]], [row[key] for row in traces[False]])

  def test_blended_cruise_settling_reduces_small_corrections_both_directions(self):
    params = Params()
    params.put_bool("DynamicExperimentalControl", False, block=True)
    params.put("AccelPersonality", AccelProfile.normal, block=True)

    def request_acceleration(_current_time: float, _speed: float, _acceleration: float) -> tuple[float, bool]:
      return 2.0, False

    peak_corrections = {}
    for enabled in (False, True):
      params.put_bool("AccelPersonalityEnabled", enabled, block=True)
      for direction, speed, cruise in (("accelerate", 20.0, 20.5), ("decelerate", 20.5, 20.0)):
        plant = PlantSP(speed=speed, e2e=True, model_action_fn=request_acceleration)
        _set_mpc_acceleration(plant)
        trace = [plant.step(v_cruise=cruise) for _ in range(20)]
        peak_corrections[enabled, direction] = max(abs(row["a_target"]) for row in trace)

    for direction in ("accelerate", "decelerate"):
      self.assertLess(peak_corrections[True, direction], peak_corrections[False, direction])

  def test_normal_is_less_aggressive_than_disabled_stock_path(self):
    stock = run_profile(AccelProfile.normal, enabled=False, speed=4.0, steps=120)
    normal = run_profile(AccelProfile.normal, speed=4.0, steps=120)
    self.assertTrue(all(tuned[1] <= original[1] + 1e-12 for tuned, original in zip(normal, stock, strict=True)))
    self.assertLess(normal[-1][0], stock[-1][0])

  def test_profiles_do_not_change_far_braking(self):
    for e2e in (False, True):
      stock = run_profile(AccelProfile.normal, enabled=False, speed=20.0, v_cruise=0.0, e2e=e2e, steps=100)
      for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport):
        self.assertEqual(run_profile(profile, speed=20.0, v_cruise=0.0, e2e=e2e, steps=100), stock)

  def test_blended_launch_respects_profiles(self):
    traces = {
      profile: run_profile(profile, v_cruise=8.0, e2e=True, steps=180)
      for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport)
    }
    time_to_five = {
      profile: next(frame for frame, row in enumerate(rows) if row[0] >= 5.0) * DT_MDL
      for profile, rows in traces.items()
    }

    self.assertLess(time_to_five[AccelProfile.sport], time_to_five[AccelProfile.normal])
    self.assertLess(time_to_five[AccelProfile.normal], time_to_five[AccelProfile.eco])

  def test_launch_ordering_without_departure_delay(self):
    stock = run_profile(AccelProfile.normal, enabled=False, v_cruise=8.0, steps=160)
    traces = {
      profile: run_profile(profile, v_cruise=8.0, steps=160)
      for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport)
    }
    first_motion = {
      profile: next(frame for frame, row in enumerate(rows) if row[0] > 0.01)
      for profile, rows in traces.items()
    }
    time_to_five = {
      profile: next(frame for frame, row in enumerate(rows) if row[0] >= 5.0) * DT_MDL
      for profile, rows in traces.items()
    }
    stock_first_motion = next(frame for frame, row in enumerate(stock) if row[0] > 0.01)
    stock_time_to_two = next(frame for frame, row in enumerate(stock) if row[0] >= 2.0) * DT_MDL
    stock_time_to_three = next(frame for frame, row in enumerate(stock) if row[0] >= 3.0) * DT_MDL

    self.assertEqual(len(set(first_motion.values())), 1)
    self.assertTrue(all(frame == stock_first_motion for frame in first_motion.values()))
    for rows in traces.values():
      time_to_two = next(frame for frame, row in enumerate(rows) if row[0] >= 2.0) * DT_MDL
      time_to_three = next(frame for frame, row in enumerate(rows) if row[0] >= 3.0) * DT_MDL
      self.assertLessEqual(time_to_two, stock_time_to_two + DT_MDL)
      self.assertLessEqual(time_to_three, stock_time_to_three + 0.1)
    self.assertLessEqual(time_to_five[AccelProfile.sport], time_to_five[AccelProfile.normal])
    self.assertLessEqual(time_to_five[AccelProfile.normal], time_to_five[AccelProfile.eco])
    self.assertLessEqual(time_to_five[AccelProfile.eco], 1.25 * time_to_five[AccelProfile.normal])

  def test_road_speed_catchup_stays_useful(self):
    traces = {
      profile: run_profile(profile, speed=20.0, v_cruise=30.0, steps=100)
      for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport)
    }
    gains = {profile: rows[-1][0] - 20.0 for profile, rows in traces.items()}
    self.assertGreaterEqual(gains[AccelProfile.sport], gains[AccelProfile.normal])
    self.assertGreaterEqual(gains[AccelProfile.eco], 0.72 * gains[AccelProfile.normal])

  def test_catchup_settles_inside_deadband_without_oscillation(self):
    target_speed = 24.0
    rows = run_profile(AccelProfile.normal, speed=20.0, v_cruise=target_speed, steps=800)
    final_deficit = target_speed - rows[-1][0]

    self.assertGreaterEqual(final_deficit, -1e-9)
    self.assertLessEqual(final_deficit, TARGET_SPEED_DEADBAND + 0.01)
    self.assertTrue(all(row[1] >= -1e-12 for row in rows))
    self.assertTrue(all(current[0] <= following[0] for current, following in zip(rows, rows[1:], strict=False)))

  def test_smaller_target_gap_uses_stock_downward_jerk(self):
    def target_speed(frame: int) -> float:
      return 24.0 if frame < 40 else 21.5

    rows = run_profile(AccelProfile.normal, speed=20.0, v_cruise_fn=target_speed, steps=80)
    command_drop = rows[39][1] - rows[40][1]

    self.assertLessEqual(command_drop, max(J_CRUISE_VALS) * DT_MDL + 1e-12)
    self.assertTrue(all(row[1] >= -1e-12 for row in rows[40:]))

  def test_full_catchup_trace_respects_stock_jerk(self):
    for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport):
      rows = run_profile(profile, v_cruise=30.0, steps=300)
      previous_speed = 0.0
      previous_accel = 0.0
      for speed, accel, _should_stop in rows:
        jerk_step = float(np.interp(previous_speed, A_CRUISE_MAX_BP, J_CRUISE_VALS)) * DT_MDL
        self.assertLessEqual(abs(accel - previous_accel), jerk_step + 1e-12)
        previous_speed = speed
        previous_accel = accel

  def test_stop_release_frame_is_profile_independent(self):
    def target_speed(frame: int) -> float:
      return 0.0 if frame < 20 else 8.0

    traces = {
      profile: run_profile(profile, v_cruise_fn=target_speed, steps=80)
      for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport)
    }
    release_frames = {
      profile: next(frame for frame, row in enumerate(rows) if frame >= 20 and not row[2])
      for profile, rows in traces.items()
    }
    stock = run_profile(AccelProfile.normal, enabled=False, v_cruise_fn=target_speed, steps=80)
    stock_release_frame = next(frame for frame, row in enumerate(stock) if frame >= 20 and not row[2])
    self.assertEqual(len(set(release_frames.values())), 1)
    self.assertTrue(all(frame == stock_release_frame for frame in release_frames.values()))
