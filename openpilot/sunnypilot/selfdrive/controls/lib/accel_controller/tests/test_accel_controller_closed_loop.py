"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from collections.abc import Callable

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_cruise_accel
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.accel_controller import AccelController, AccelProfile


class CarParams:
  steerRatio = 15.0
  wheelbase = 2.7


def run_profile(profile: int, *, enabled: bool = True, speed: float = 0.0, v_cruise: float = 30.0,
                v_cruise_fn: Callable[[int], float] | None = None, steps: int = 120):
  params = Params()
  params.put_bool("AccelPersonalityEnabled", enabled, block=True)
  params.put("AccelPersonality", profile, block=True)
  controller = AccelController()

  accel = 0.0
  rows = []
  for frame in range(steps):
    target_speed = v_cruise if v_cruise_fn is None else v_cruise_fn(frame)
    custom_profile = controller.is_enabled() and controller.profile != AccelProfile.normal
    max_accel_override = controller.get_max_accel(speed) if custom_profile else None
    accel = get_cruise_accel(False, target_speed, speed, accel, 0.0, CarParams(), DT_MDL, 2.0, True, max_accel_override)
    speed = max(0.0, speed + accel * DT_MDL)
    rows.append((speed, accel, should_stop(speed, accel)))
  return rows


class TestAccelControllerClosedLoop(OpenpilotTestCase):
  def test_normal_matches_disabled_stock_path(self):
    stock = run_profile(AccelProfile.normal, enabled=False, speed=4.0, steps=120)
    normal = run_profile(AccelProfile.normal, speed=4.0, steps=120)
    self.assertEqual(normal, stock)

  def test_profiles_do_not_change_braking(self):
    stock = run_profile(AccelProfile.normal, enabled=False, speed=20.0, v_cruise=0.0, steps=100)
    for profile in (AccelProfile.eco, AccelProfile.normal, AccelProfile.sport):
      self.assertEqual(run_profile(profile, speed=20.0, v_cruise=0.0, steps=100), stock)

  def test_launch_ordering_without_departure_delay(self):
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

    self.assertEqual(len(set(first_motion.values())), 1)
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
    self.assertEqual(len(set(release_frames.values())), 1)
