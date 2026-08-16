import numpy as np
from marl_spklu.env.history_buffer import HistoryBuffer

def test_gini_calculation():
    # Uniform distribution -> Gini = 0
    util = np.array([0.5, 0.5, 0.5, 0.5])
    gini = HistoryBuffer.calculate_gini(util)
    assert np.isclose(gini, 0.0)
    
    # Perfect inequality -> Gini close to 1
    util2 = np.array([1.0, 0.0, 0.0, 0.0])
    gini2 = HistoryBuffer.calculate_gini(util2)
    assert gini2 > 0.5
    
    # All zeros -> Gini = 0
    util3 = np.array([0.0, 0.0])
    gini3 = HistoryBuffer.calculate_gini(util3)
    assert np.isclose(gini3, 0.0)

def test_history_recording():
    hb = HistoryBuffer(["S1", "S2"], window_size_15m=10)
    hb.record_utilization({"S1": 0.5, "S2": 0.0})
    hb.record_utilization({"S1": 0.5, "S2": 1.0})
    
    recent = hb.get_recent_utilization(steps=2)
    assert np.isclose(recent[0], 0.5) # S1 avg
    assert np.isclose(recent[1], 0.5) # S2 avg

def test_gini_trend():
    hb = HistoryBuffer(["S1", "S2"], window_size_15m=10)
    
    # Increasing inequality over 3 steps
    # Step 0: [0.5, 0.5] -> Gini = 0
    hb.record_utilization({"S1": 0.5, "S2": 0.5})
    # Step 1: [0.7, 0.3] -> Gini > 0
    hb.record_utilization({"S1": 0.7, "S2": 0.3})
    # Step 2: [0.9, 0.1] -> Gini > step1
    hb.record_utilization({"S1": 0.9, "S2": 0.1})
    
    trend = hb.get_gini_trend(steps=3)
    # Slope should be positive because Gini is increasing
    assert trend > 0.0
