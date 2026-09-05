"""
BAS Experiment Copilot - Deterministic Protocol FSM Test Runner (train/test_fsm.py)

Feeds hardcoded fake action sequences through the FSM with NO real AI/camera involved.
Verifies pure deterministic decision logic independently of perception models:
1. Correct full sequence, in order -> all CORRECT
2. One step skipped -> OUT_OF_ORDER when later step is attempted
3. Steps done out of order -> OUT_OF_ORDER
4. A repeated/already-done step -> SKIPPED
5. A random unrelated action -> UNKNOWN
6. A low-confidence detection -> LOW_CONFIDENCE_HOLD
"""

import sys
import os

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.protocol_fsm import (
    ProtocolFSM,
    CORRECT,
    OUT_OF_ORDER,
    SKIPPED,
    UNKNOWN,
    LOW_CONFIDENCE_HOLD
)

PROTOCOL_PATH = os.path.join(WORKSPACE_ROOT, "protocols", "fluid_physics.json")


def print_separator(title: str = ""):
    if title:
        print(f"\n{'=' * 75}\n  TEST: {title}\n{'=' * 75}")
    else:
        print("-" * 75)


def test_correct_full_sequence():
    """1. Correct full sequence, in order -> all CORRECT"""
    print_separator("1. Correct full sequence in order (Nominal Run)")
    fsm = ProtocolFSM(protocol_path=PROTOCOL_PATH, confidence_threshold=0.6)

    expected_sequence = [
        ("open_chamber",     0.95),
        ("insert_cartridge", 0.92),
        ("attach_probe",     0.88),
        ("verify_seal",      0.94),
        ("activate",         0.97),
    ]

    for step_num, (action, conf) in enumerate(expected_sequence):
        expected_step_name = fsm.steps[fsm.current_step_index]["label"]
        res = fsm.check(action, confidence=conf, threshold=0.6)
        print(f"  [Step {step_num}] Action: '{action:<18}' | Conf: {conf:.2f} | Expected: '{expected_step_name:<28}' -> Result: {res}")
        assert res == CORRECT, f"Expected {CORRECT}, got {res} at step {step_num}"

    assert fsm.is_completed is True, "Protocol should be marked completed"
    assert fsm.current_step_index == 5, "FSM index should have advanced to 5"
    print("  >>> [PASS] All 5 steps confirmed CORRECT in order. Protocol completed.")


def test_one_step_skipped():
    """2. One step skipped -> OUT_OF_ORDER when the later step is attempted"""
    print_separator("2. One step skipped (Attempting Step 1 before Step 0)")
    fsm = ProtocolFSM(protocol_path=PROTOCOL_PATH, confidence_threshold=0.6)

    # At step 0, expecting "open_chamber", but operator performs "insert_cartridge" (step 1)
    res = fsm.check("insert_cartridge", confidence=0.90, threshold=0.6)
    print(f"  Action: 'insert_cartridge' (Step 1) while expecting Step 0 ('open_chamber') -> Result: {res}")

    assert res == OUT_OF_ORDER, f"Expected {OUT_OF_ORDER}, got {res}"
    assert fsm.current_step_index == 0, "FSM should NOT advance on OUT_OF_ORDER"
    print("  >>> [PASS] Intercepted skipped step as OUT_OF_ORDER; index safely held at 0.")


def test_steps_done_out_of_order():
    """3. Steps done out of order -> OUT_OF_ORDER"""
    print_separator("3. Steps done out of order (Jumping from Step 0 to Step 3)")
    fsm = ProtocolFSM(protocol_path=PROTOCOL_PATH, confidence_threshold=0.6)

    # Complete Step 0 nominally
    res0 = fsm.check("open_chamber", confidence=0.90, threshold=0.6)
    assert res0 == CORRECT
    assert fsm.current_step_index == 1

    # Now expecting Step 1 ("insert_cartridge"), but operator jumps to Step 3 ("verify_seal")
    res_jump = fsm.check("verify_seal", confidence=0.88, threshold=0.6)
    print(f"  Action: 'verify_seal' (Step 3) while expecting Step 1 ('insert_cartridge') -> Result: {res_jump}")

    assert res_jump == OUT_OF_ORDER, f"Expected {OUT_OF_ORDER}, got {res_jump}"
    assert fsm.current_step_index == 1, "FSM should remain at Step 1"
    print("  >>> [PASS] Intercepted out-of-order action as OUT_OF_ORDER; index held at Step 1.")


def test_repeated_already_done_step():
    """4. A repeated/already-done step -> SKIPPED"""
    print_separator("4. Repeated / Already-done step (Re-performing Step 0 at Step 1)")
    fsm = ProtocolFSM(protocol_path=PROTOCOL_PATH, confidence_threshold=0.6)

    # Complete Step 0
    res0 = fsm.check("open_chamber", confidence=0.92, threshold=0.6)
    assert res0 == CORRECT
    assert fsm.current_step_index == 1

    # Operator repeats Step 0 ("open_chamber") instead of Step 1 ("insert_cartridge")
    res_repeat = fsm.check("open_chamber", confidence=0.85, threshold=0.6)
    print(f"  Action: 'open_chamber' (Step 0) repeated while at Step 1 -> Result: {res_repeat}")

    assert res_repeat == SKIPPED, f"Expected {SKIPPED}, got {res_repeat}"
    assert fsm.current_step_index == 1, "FSM should remain at Step 1"
    print("  >>> [PASS] Intercepted repeated past step as SKIPPED; index maintained at Step 1.")


def test_random_unrelated_action():
    """5. A random unrelated action -> UNKNOWN"""
    print_separator("5. Random unrelated action (Action not in protocol definition)")
    fsm = ProtocolFSM(protocol_path=PROTOCOL_PATH, confidence_threshold=0.6)

    # Unrelated action
    res = fsm.check("drink_coffee", confidence=0.99, threshold=0.6)
    print(f"  Action: 'drink_coffee' -> Result: {res}")

    assert res == UNKNOWN, f"Expected {UNKNOWN}, got {res}"
    assert fsm.current_step_index == 0, "FSM should remain at Step 0"

    # Another random action
    res2 = fsm.check("adjust_visor", confidence=0.91, threshold=0.6)
    print(f"  Action: 'adjust_visor' -> Result: {res2}")
    assert res2 == UNKNOWN

    print("  >>> [PASS] Unrelated actions correctly flagged as UNKNOWN.")


def test_low_confidence_detection():
    """6. A low-confidence detection -> LOW_CONFIDENCE_HOLD"""
    print_separator("6. Low-confidence detection (< 0.60 threshold hold)")
    fsm = ProtocolFSM(protocol_path=PROTOCOL_PATH, confidence_threshold=0.6)

    # Action is the correct one ("open_chamber"), but confidence is only 0.42 (< 0.60)
    res = fsm.check("open_chamber", confidence=0.42, threshold=0.6)
    print(f"  Action: 'open_chamber' with Conf: 0.42 (< 0.60 threshold) -> Result: {res}")

    assert res == LOW_CONFIDENCE_HOLD, f"Expected {LOW_CONFIDENCE_HOLD}, got {res}"
    assert fsm.current_step_index == 0, "FSM must hold and NOT advance on low confidence"
    print("  >>> [PASS] Safe-state gate activated: LOW_CONFIDENCE_HOLD; state preserved.")


def run_all_tests():
    print("=" * 75)
    print("  BAS EXPERIMENT COPILOT — DETERMINISTIC PROTOCOL FSM VERIFICATION")
    print(f"  Protocol: {PROTOCOL_PATH}")
    print("  Engine:   Pure Deterministic Logic (Zero AI/ML Dependencies)")
    print("=" * 75)

    test_correct_full_sequence()
    test_one_step_skipped()
    test_steps_done_out_of_order()
    test_repeated_already_done_step()
    test_random_unrelated_action()
    test_low_confidence_detection()

    print("\n" + "=" * 75)
    print("  ALL 6 REQUIRED TEST SCENARIOS PASSED WITH ZERO ERRORS!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_all_tests()
