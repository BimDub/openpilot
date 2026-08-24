from types import SimpleNamespace

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.realtime import DT_CTRL
from openpilot.cereal import custom
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, long_control_state_trans


def make_stopping_control(last_output_accel):
  CP = SimpleNamespace(stopAccel=-2.0, longitudinalTuning=SimpleNamespace(kiBP=[0.0], kiV=[0.0]))
  control = LongControl(CP, custom.CarParamsSP.new_message())
  control.long_control_state = LongCtrlState.stopping
  control.last_output_accel = last_output_accel
  CS = SimpleNamespace(aEgo=0.0, brakePressed=False, cruiseState=SimpleNamespace(standstill=False), vEgo=0.0)
  return control, CS


class TestLongControlStateTransition(OpenpilotTestCase):

  def test_stopping_threshold_boundaries(self):
    assert should_stop(0.249, 0.099)
    assert not should_stop(0.250, 0.099)
    assert not should_stop(0.249, 0.1)

  def test_stopping_ramp_rate(self):
    control, CS = make_stopping_control(-0.2)
    output = control.update(True, CS, a_target=-0.2, should_stop=True, accel_limits=(-3.5, 2.0))

    self.assertAlmostEqual(output, -0.2 - 0.3 * DT_CTRL)

  def test_stopping_does_not_release_stronger_braking(self):
    control, CS = make_stopping_control(-2.2)
    output = control.update(True, CS, a_target=-0.2, should_stop=True, accel_limits=(-3.5, 2.0))

    self.assertEqual(output, -2.2)

  def test_stay_stopped(self):
    CP_SP = custom.CarParamsSP.new_message()
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off

  def test_engage(self):
    CP_SP = custom.CarParamsSP.new_message()
    active = True
    current_state = LongCtrlState.off
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
