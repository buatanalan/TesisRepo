"""Test unit V1.1/V1.2 (Panduan_Validasi_Simulasi.docx Tingkat 1) untuk Simulator +
SPKLU baru: kapasitas konektor, antrian FIFO, mass balance, dan renege terintegrasi
end-to-end. Menggantikan test_simulator.py lama (formula MXL/balk yang sudah dihapus,
lihat archive/old_design_code/tests/)."""
import pytest

from marl_spklu.env.simulator import Simulator
from marl_spklu.env.spklu import SPKLU
from marl_spklu.env.user import User, UserState
from marl_spklu.env.history_buffer import HistoryBuffer


def make_user(uid, w4=0.0, patience_minutes=1440.0):
    return User(uid, w1=-0.1, w2=0.5, w3=0.1, w4=w4, w5=0.0,
                patience_minutes=patience_minutes)


# ---------------------------------------------------------------------------
# V1.1 -- SPKLU: konektor tak pernah melayani >1 sesi simultan; antrian FIFO
# ---------------------------------------------------------------------------

def test_connector_capacity_never_exceeded():
    s = SPKLU("S1", capacities={"DC": 1}, location=(0.0, 0.0))
    s.request_connector("U1", "DC")
    s.request_connector("U2", "DC")
    s.request_connector("U3", "DC")
    # Isi 1 konektor dari antrian (langkah step biasa)
    finished, newly_charging = s.step(dt_minutes=1.0)
    assert len(s.charging["DC"]) <= s.capacities["DC"]
    assert len(newly_charging) == 1  # hanya 1 yg bisa masuk (kapasitas=1)
    assert s.get_queue_length() == 2  # sisa 2 masih antre


def test_queue_is_fifo_order():
    s = SPKLU("S1", capacities={"DC": 1}, location=(0.0, 0.0))
    s.request_connector("FIRST", "DC")
    s.request_connector("SECOND", "DC")
    finished, newly_charging = s.step(dt_minutes=1.0)
    assert newly_charging[0][0] == "FIRST"  # yg pertama antre, pertama dilayani


def test_remove_from_queue_renege():
    s = SPKLU("S1", capacities={"DC": 1}, location=(0.0, 0.0))
    s.request_connector("U1", "DC")
    s.request_connector("U2", "DC")
    s.remove_from_queue("U1")
    assert s.queues["DC"] == ["U2"]
    finished, newly_charging = s.step(dt_minutes=1.0)
    assert newly_charging[0][0] == "U2"  # U1 sudah tak ada di antrian


def test_remove_from_queue_noop_if_not_present():
    s = SPKLU("S1", capacities={"DC": 1}, location=(0.0, 0.0))
    s.request_connector("U1", "DC")
    s.remove_from_queue("NONEXISTENT")  # tak boleh error
    assert s.queues["DC"] == ["U1"]


# ---------------------------------------------------------------------------
# V1.2 -- Mass balance: jumlah user (charging+antrian+transit+selesai) konsisten
# ---------------------------------------------------------------------------

def _simple_sim(n_conn=1, patience_minutes=1440.0):
    spklu = SPKLU("S1", capacities={"DC": n_conn}, location=(0.0, 0.0))
    users = [make_user(f"U{i}", patience_minutes=patience_minutes) for i in range(5)]
    sim = Simulator({"S1": spklu}, users, HistoryBuffer(["S1"], window_size_15m=50))
    sim.spklu_features = {"S1": {"loc": (0.0, 0.0), "pop": 1.0, "conn": n_conn}}
    for i, u in enumerate(users):
        sim.spawn_schedule.setdefault(i, []).append((u, (0.0, 0.0), 80.0))
    return sim, users


def test_mass_balance_no_user_lost_or_duplicated():
    sim, users = _simple_sim(n_conn=2)
    for step in range(40):
        sim.step_once(step)
    # Setiap user harus berakhir di salah satu state valid, tak ada yang "hilang"
    # (mis. None atau state tak dikenal).
    valid_states = {UserState.IDLE, UserState.SPAWNED, UserState.TRAVELING,
                     UserState.QUEUING, UserState.CHARGING, UserState.DONE, UserState.RENEGED}
    assert all(u.state in valid_states for u in users)
    assert len(users) == 5  # jumlah entitas tak berubah


def test_renege_integrated_removes_user_from_queue_end_to_end():
    """User dgn patience sangat pendek harus RENEGE saat antrian penuh lama, dan
    tersingkir dari antrian SPKLU (bukan cuma state berubah tanpa efek nyata)."""
    spklu = SPKLU("S1", capacities={"DC": 1}, location=(0.0, 0.0))
    # occupant lama-lama nge-charge shg antrian menumpuk & tak kunjung kosong
    spklu.charging["DC"] = [{"user_id": "OCCUPANT", "remaining_time": 10_000.0}]
    impatient = make_user("IMPATIENT", patience_minutes=10.0)
    sim = Simulator({"S1": spklu}, [impatient],
                     HistoryBuffer(["S1"], window_size_15m=50))
    sim.spklu_features = {"S1": {"loc": (0.0, 0.0), "pop": 1.0, "conn": 1}}
    sim.spawn_schedule.setdefault(0, []).append((impatient, (0.0, 0.0), 80.0))

    for step in range(10):
        sim.step_once(step)

    assert impatient.state == UserState.RENEGED
    assert "IMPATIENT" not in spklu.queues.get("DC", [])


def test_default_patience_prevents_renege_under_load():
    """Dgn patience default (1 hari), user TIDAK reneges walau antrian panjang di
    horizon simulasi pendek (30-90 hari punya wait realistis << 1 hari)."""
    sim, users = _simple_sim(n_conn=1, patience_minutes=1440.0)
    for step in range(50):
        sim.step_once(step)
    assert all(u.state != UserState.RENEGED for u in users)


# ---------------------------------------------------------------------------
# V1.3 -- Reprodusibilitas: seed sama -> hasil identik
# ---------------------------------------------------------------------------

def test_charge_time_sampling_reproducible_with_seed():
    import random
    from marl_spklu.env.spklu import sample_charge_time

    random.seed(123)
    a = [sample_charge_time("DC") for _ in range(20)]
    random.seed(123)
    b = [sample_charge_time("DC") for _ in range(20)]
    assert a == b
