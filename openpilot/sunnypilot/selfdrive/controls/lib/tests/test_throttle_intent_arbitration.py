"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from types import SimpleNamespace
from typing import cast

from openpilot.cereal import messaging, custom, log
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_coast_accel, get_cruise_accel
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.accel_controller import AccelController
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

PlanSource = log.LongitudinalPlan.LongitudinalPlanSource
SunnyPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource


class FakeSubMaster(dict):
  def __init__(self, lead_one: bool = True, lead_two: bool = False, *, valid: bool = True, alive: bool = True):
    lead = {'dRel': 120.8, 'vRel': -16.4, 'aLeadK': -1.7}
    super().__init__(radarState=SimpleNamespace(
      leadOne=SimpleNamespace(present=lead_one, **lead),
      leadTwo=SimpleNamespace(present=lead_two, **lead),
    ), carState=SimpleNamespace(vEgo=17.3),
      selfdriveState=SimpleNamespace(personality=log.LongitudinalPersonality.standard))
    self.valid = {'radarState': valid}
    self.alive = {'radarState': alive}


def arbitrate(sm, mpc_accel=0.19, gated_cruise=-0.26, ungated_cruise=0.5, source=PlanSource.lead0,
              allow_throttle=False, e2e=False, force_decel=False, lead_comfort=False, previous_accel=0.0):
  planner = make_planner(lead_comfort, previous_accel)
  return planner.arbitrate_cruise_candidate(
    cast(messaging.SubMaster, sm), gated_cruise, ungated_cruise, mpc_accel, source,
    allow_throttle=allow_throttle, e2e=e2e, force_decel=force_decel,
  )


def make_planner(lead_comfort: bool = False, previous_accel: float = 0.0):
  planner = object.__new__(LongitudinalPlannerSP)
  planner.accel_controller = AccelController()
  planner.accel_controller._enabled = lead_comfort
  planner.source = SunnyPlanSource.cruise
  planner.output_a_target = previous_accel
  return planner


class TestThrottleIntentArbitration(OpenpilotTestCase):
  def test_coast_gate_cannot_add_braking_a_valid_lead_does_not_need(self):
    selected = arbitrate(FakeSubMaster())
    self.assertEqual(selected, 0.5)
    self.assertEqual(min(selected, 0.19), 0.19)

  def test_counterfactual_must_select_the_lead(self):
    self.assertEqual(arbitrate(FakeSubMaster(), ungated_cruise=0.1), -0.26)

  def test_legitimate_cruise_braking_is_preserved(self):
    self.assertEqual(arbitrate(FakeSubMaster(), gated_cruise=-0.5, ungated_cruise=-0.5), -0.5)

  def test_hard_lead_braking_remains_authoritative(self):
    selected = arbitrate(FakeSubMaster(), mpc_accel=-0.6)
    self.assertEqual(selected, -0.26)
    self.assertEqual(min(selected, -0.6), -0.6)

  def test_lead_comfort_is_only_an_extra_braking_candidate(self):
    for sm, source in ((FakeSubMaster(), PlanSource.lead0), (FakeSubMaster(False, True), PlanSource.lead1)):
      cruise = arbitrate(sm, mpc_accel=0.5, gated_cruise=0.5, ungated_cruise=0.5, source=source,
                         allow_throttle=True, lead_comfort=True, previous_accel=0.2)
      self.assertLess(cruise, 0.2)
      self.assertEqual(min(cruise, -1.2), -1.2)

  def test_lead_comfort_requires_a_live_selected_lead(self):
    for sm, source in ((FakeSubMaster(valid=False), PlanSource.lead0),
                       (FakeSubMaster(alive=False), PlanSource.lead0),
                       (FakeSubMaster(lead_one=False), PlanSource.lead0),
                       (FakeSubMaster(lead_one=True, lead_two=False), PlanSource.lead1)):
      with self.subTest(source=source):
        self.assertEqual(arbitrate(sm, gated_cruise=0.5, ungated_cruise=0.5, source=source,
                                   allow_throttle=True, lead_comfort=True), 0.5)

  def test_policy_requires_live_valid_selected_lead(self):
    cases = (
      FakeSubMaster(valid=False),
      FakeSubMaster(alive=False),
      FakeSubMaster(lead_one=False),
      FakeSubMaster(lead_one=True, lead_two=False),
    )
    sources = (PlanSource.lead0, PlanSource.lead0, PlanSource.lead0, PlanSource.lead1)
    for sm, source in zip(cases, sources, strict=True):
      with self.subTest(source=source, valid=sm.valid, alive=sm.alive):
        self.assertEqual(arbitrate(sm, source=source), -0.26)

  def test_second_lead_is_supported(self):
    self.assertEqual(arbitrate(FakeSubMaster(False, True), source=PlanSource.lead1), 0.5)

  def test_bypass_modes_are_stock_identical(self):
    for allow_throttle, e2e, force_decel in ((True, False, False), (False, True, False), (False, False, True)):
      with self.subTest(allow_throttle=allow_throttle, e2e=e2e, force_decel=force_decel):
        selected = arbitrate(FakeSubMaster(), allow_throttle=allow_throttle, e2e=e2e, force_decel=force_decel)
        self.assertEqual(selected, -0.26)

  def test_nonfinite_inputs_do_not_bypass_gate(self):
    for gated, ungated, mpc_accel in ((float('nan'), 0.5, 0.2),
                                      (-0.26, float('inf'), 0.2),
                                      (-0.26, 0.5, float('nan'))):
      with self.subTest(gated=gated, ungated=ungated, mpc_accel=mpc_accel):
        selected = arbitrate(FakeSubMaster(), mpc_accel, gated, ungated)
        if gated != gated:
          self.assertNotEqual(selected, selected)
        else:
          self.assertEqual(selected, gated)

  def test_5c4_shape_keeps_positive_lead_authoritative_until_it_brakes(self):
    class CarParams:
      steerRatio = 15.0
      wheelbase = 2.7

    planner = make_planner()
    sm = FakeSubMaster()
    v_ego = 6.9
    v_cruise = 30.0
    accel_coast = get_coast_accel(-0.007)
    cruise_accel = 0.185
    mpc_trace = (0.18, 0.175, 0.165, 0.126, 0.06, -0.10, -0.30)
    outputs = []

    for mpc_accel in mpc_trace:
      previous_cruise_accel = cruise_accel
      gated = get_cruise_accel(False, v_cruise, v_ego, previous_cruise_accel, 0.0,
                                CarParams(), DT_MDL, accel_coast, False, 1.45)
      ungated = get_cruise_accel(False, v_cruise, v_ego, previous_cruise_accel, 0.0,
                                  CarParams(), DT_MDL, accel_coast, True, 1.45)
      cruise_accel = planner.arbitrate_cruise_candidate(
        cast(messaging.SubMaster, sm), gated, ungated, mpc_accel, PlanSource.lead0,
        allow_throttle=False, e2e=False, force_decel=False,
      )
      outputs.append(min(cruise_accel, mpc_accel))
      self.assertLessEqual(abs(cruise_accel - previous_cruise_accel), 0.07)

    self.assertEqual(outputs, list(mpc_trace))
