from marl_spklu.env.spklu import SPKLU

def test_spklu_initialization():
    spklu = SPKLU("SPKLU_1", {"AC": 2, "DC": 1}, (0, 0))
    
    assert spklu.spklu_id == "SPKLU_1"
    assert spklu.capacities["AC"] == 2
    assert spklu.capacities["DC"] == 1
    assert "AC" in spklu.queues
    assert "DC" in spklu.queues

def test_spklu_utilization():
    spklu = SPKLU("SPKLU_1", {"AC": 2, "DC": 1}, (0, 0))
    
    # AC has cap 2
    spklu.request_connector("u1", "AC")
    spklu.request_connector("u2", "AC")
    spklu.request_connector("u3", "AC")
    spklu.request_connector("u4", "DC")
    
    finished, newly = spklu.step(15.0)
    
    assert spklu.get_utilization("AC") == 1.0
    assert spklu.get_utilization("DC") == 1.0
    
    assert len(spklu.queues["AC"]) == 1 # u3 is in queue

def test_spklu_estimate_wait_time():
    spklu = SPKLU("SPKLU_1", {"AC": 1}, (0, 0))
    
    # 0 in queue, 0 charging -> wait is 0
    assert spklu.estimate_wait_time("AC", avg_charge_time=30.0) == 0.0
    
    # add one, advance step so it charges
    spklu.request_connector("u1", "AC")
    spklu.step(1.0)
    
    # 1 charging, cap is 1 -> wait is 30 for the next
    wait = spklu.estimate_wait_time("AC", avg_charge_time=30.0)
    assert wait == 30.0
    
    # add another to queue
    spklu.request_connector("u2", "AC")
    wait2 = spklu.estimate_wait_time("AC", avg_charge_time=30.0)
    # wait is 60 (2 * 30 / 1)
    assert wait2 == 60.0
