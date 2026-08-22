"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import math

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.controls.lib.throttle_intent_controller import ThrottleIntentController


class TestAllowThrottle(OpenpilotTestCase):
  def setUp(self):
    self.controller = ThrottleIntentController()

  def update(self, throttle_prob: float, low_speed_override: bool = False) -> bool:
    return self.controller.update(throttle_prob, low_speed_override=low_speed_override, threshold=0.4)

  def test_short_probability_dip(self):
    self.assertTrue(self.update(1.0))
    self.assertTrue(self.update(0.0))
    self.assertTrue(self.update(0.0))
    self.assertTrue(self.update(0.0))

  def test_probability_chatter(self):
    self.assertTrue(self.update(1.0))
    for _ in range(100):
      self.assertTrue(self.update(0.39))
      self.assertTrue(self.update(0.46))

    self.controller = ThrottleIntentController()
    for _ in range(10):
      self.update(0.0)
    self.assertFalse(self.update(0.0))
    for _ in range(100):
      self.assertFalse(self.update(0.39))
      self.assertFalse(self.update(0.46))

  def test_sustained_probability_changes(self):
    self.assertTrue(self.update(1.0))
    states = [self.update(0.0) for _ in range(8)]
    self.assertTrue(all(states[:-1]))
    self.assertFalse(states[-1])

    for _ in range(20):
      self.assertFalse(self.update(0.0))
    self.assertEqual([self.update(1.0) for _ in range(2)], [False, True])

  def test_threshold_boundaries(self):
    for _ in range(6):
      state = self.update(0.4)
    self.assertFalse(state)
    self.controller.prob_filter.x = 0.44
    self.controller.prob_filter.initialized = True
    self.assertFalse(self.update(0.44))
    self.controller.prob_filter.x = 0.46
    self.assertTrue(self.update(0.46))

  def test_low_speed_override(self):
    for _ in range(20):
      self.assertTrue(self.update(0.0, True))

    self.assertTrue(self.update(1.0))
    self.assertTrue(self.update(math.nan, True))
    self.assertEqual(self.controller.prob_filter.x, 0.45)
    self.assertTrue(self.controller.prob_filter.initialized)
    self.assertTrue(self.update(1.0))
    for _ in range(20):
      self.assertTrue(self.update(0.0, True))
    states = [self.update(0.0) for _ in range(6)]
    self.assertTrue(all(states[:-1]))
    self.assertFalse(states[-1])

  def test_nonfinite_probability(self):
    self.assertTrue(self.update(1.0))
    for value in (math.inf, -math.inf, math.nan):
      self.assertFalse(self.update(value))
      self.assertTrue(math.isfinite(self.controller.prob_filter.x))

    self.assertFalse(self.update(1.0))
    self.assertTrue(self.update(1.0))

  def test_route_probability_trace(self):
    probabilities = (0.941, 0.093, 0.070, 0.429, 0.430, 0.083, 0.509, 0.068)
    states = [self.update(probability) for probability in probabilities]
    self.assertTrue(all(states))

  def test_sustained_low_probability_still_disables_throttle(self):
    self.assertTrue(self.update(1.0))
    states = [self.update(0.0) for _ in range(20)]
    self.assertFalse(all(states))
    self.assertFalse(states[-1])

  def test_filter_updates_once(self):
    self.assertTrue(self.update(1.0))
    self.assertTrue(self.update(0.0))
    self.assertAlmostEqual(self.controller.prob_filter.x, 2.0 / 3.0)

  def test_disable_dwell_uses_wall_time(self):
    self.controller = ThrottleIntentController(dt=0.1)
    states = [self.update(0.0) for _ in range(3)]
    self.assertEqual(states, [True, True, False])
