"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import math

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

THROTTLE_PROB_FILTER_RC = 0.10
THROTTLE_PROB_HYSTERESIS = 0.05
THROTTLE_DISABLE_TIME = 0.30


class ThrottleIntentController:
  def __init__(self, dt: float = DT_MDL):
    self.disable_frames = max(1, math.ceil(THROTTLE_DISABLE_TIME / dt))
    self.allow_throttle = True
    self.disallow_frames = 0
    self.prob_filter = FirstOrderFilter(0.0, THROTTLE_PROB_FILTER_RC, dt, initialized=False)

  def update(self, throttle_prob: float, *, low_speed_override: bool, threshold: float) -> bool:
    if low_speed_override:
      self.allow_throttle = True
      self.disallow_frames = 0
      self.prob_filter.x = threshold + THROTTLE_PROB_HYSTERESIS
      self.prob_filter.initialized = True
      return True

    if not math.isfinite(throttle_prob):
      self.allow_throttle = False
      self.disallow_frames = 0
      self.prob_filter.x = 0.0
      self.prob_filter.initialized = True
      return False

    filtered_throttle_prob = self.prob_filter.update(throttle_prob)
    if self.allow_throttle:
      if filtered_throttle_prob <= threshold:
        self.disallow_frames += 1
        if self.disallow_frames >= self.disable_frames:
          self.allow_throttle = False
          self.disallow_frames = 0
      else:
        self.disallow_frames = 0
    elif filtered_throttle_prob > threshold + THROTTLE_PROB_HYSTERESIS:
      self.allow_throttle = True
      self.disallow_frames = 0

    return self.allow_throttle
