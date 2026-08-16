import numpy as np
import torch
from marl_spklu.rl.rewards import RewardCalculator
from marl_spklu.rl.policy import HPPOPolicy

def test_reward_herding():
    rc = RewardCalculator(alpha_flock=0.3)
    # 3 agents recommending the same station in a window of 3
    assert abs(rc.flocking_penalty(3, 3) - 2/3) < 1e-6
    # 1 agent recommending a unique station
    assert rc.flocking_penalty(1, 3) == 0.0
    # Single agent in window
    assert rc.flocking_penalty(1, 1) == 0.0

def test_policy_input_shapes():
    from marl_spklu.rl.policy import HIST_FEAT_DIM, STATION_FEAT_DIM, CRITIC_STATION_FEAT_DIM

    n_spklu = 50
    obs_dim = STATION_FEAT_DIM * n_spklu + 4          # scalar_dim=4 (arbitrary, >=0)
    critic_obs_dim = CRITIC_STATION_FEAT_DIM * n_spklu + 5   # critic_scalar_dim=5 (arbitrary, >=0)
    policy = HPPOPolicy(obs_dim, critic_obs_dim, n_spklu)

    obs = torch.randn(2, obs_dim)
    critic_obs = torch.randn(2, critic_obs_dim)
    hist = torch.randn(2, 5, HIST_FEAT_DIM)

    logits, value = policy(obs, hist, critic_obs)
    assert logits.shape == (2, n_spklu)
    # Kritik ganda (adopsi MASTER): `value` kini SELALU (B, n_critics) -- tak lagi
    # di-squeeze. n_critics=1 (default) = perilaku lama, hanya bentuknya (B,1).
    assert value.shape == (2, 1)

    # K=2: satu kepala nilai per aliran reward (individual vs global).
    policy2 = HPPOPolicy(obs_dim, critic_obs_dim, n_spklu, n_critics=2)
    logits2, value2 = policy2(obs, hist, critic_obs)
    assert logits2.shape == (2, n_spklu)
    assert value2.shape == (2, 2)


def test_virtual_queue_wait():
    from marl_spklu.env.simulator import Simulator
    from marl_spklu.env.spklu import SPKLU
    from marl_spklu.env.user import User

    # 1. Setup SPKLU with capacities = {'DC': 2}
    spklu = SPKLU("S1", capacities={"DC": 2}, location=(0.0, 0.0))
    sim = Simulator(spklu_dict={"S1": spklu}, users=[], history_buffer=None)
    
    # 2. Case 1: 1 charger in use (remaining 30m), 1 empty. User location such that travel time is 10 mins.
    # Dist_km = 40.0 * (10 / 60) = 6.6667 km. User location at (6.6667, 0.0)
    user = User("U1", w1=-0.15, w2=1.0, w3=0.1, w4=4.0, w5=0.0, connector_types={"DC"})
    user.location = (6.6667, 0.0)
    
    # Put 1 active EV in charging
    spklu.charging["DC"] = [{"user_id": "active1", "remaining_time": 30.0}]
    
    virtual_wait = sim.compute_virtual_wait(user, spklu, current_time=0.0)
    assert abs(virtual_wait) < 1e-4  # Charger 2 is empty and immediately available

    # 3. Case 2: Both chargers in use (remaining 20m and 40m). User arrives in 5 mins.
    # Dist_km = 40.0 * (5 / 60) = 3.3333 km. User location at (3.3333, 0.0)
    user2 = User("U2", w1=-0.15, w2=1.0, w3=0.1, w4=4.0, w5=0.0, connector_types={"DC"})
    user2.location = (3.3333, 0.0)
    
    spklu.charging["DC"] = [
        {"user_id": "active1", "remaining_time": 20.0},
        {"user_id": "active2", "remaining_time": 40.0}
    ]
    
    virtual_wait2 = sim.compute_virtual_wait(user2, spklu, current_time=0.0)
    # arrival_time = 5. Slots available at 20 and 40.
    # earliest slot available at 20. start_time = 20.
    # virtual_wait = start_time - arrival_time = 20 - 5 = 15.0 mins.
    assert abs(virtual_wait2 - 15.0) < 1e-4
    
    # 4. Check isolation: queues and charging states should not change
    assert len(spklu.charging["DC"]) == 2
    assert spklu.charging["DC"][0]["remaining_time"] == 20.0
    assert spklu.charging["DC"][1]["remaining_time"] == 40.0
    assert len(spklu.queues.get("DC", [])) == 0
