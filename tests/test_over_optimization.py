import pytest

def check_over_optimization_trap(cai: float, predicted_score: int, folding_energy: float) -> bool:
    """Helper function encapsulating the Over-Optimization Trap rule condition."""
    return cai > 0.85 and predicted_score <= 2 and folding_energy < -15.0

def test_over_optimization_trap_trigger():
    # Exact condition met: CAI > 0.85, score <= 2, MFE < -15.0
    assert check_over_optimization_trap(cai=0.89, predicted_score=1, folding_energy=-18.5) is True
    assert check_over_optimization_trap(cai=0.86, predicted_score=2, folding_energy=-15.1) is True

def test_over_optimization_trap_non_trigger():
    # CAI <= 0.85
    assert check_over_optimization_trap(cai=0.84, predicted_score=1, folding_energy=-18.5) is False
    
    # predicted_score > 2
    assert check_over_optimization_trap(cai=0.89, predicted_score=3, folding_energy=-18.5) is False
    
    # folding_energy >= -15.0
    assert check_over_optimization_trap(cai=0.89, predicted_score=1, folding_energy=-14.9) is False
