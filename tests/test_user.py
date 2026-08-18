"""Test unit V1.1 (Panduan_Validasi_Simulasi.docx Tingkat 1) untuk mekanisme User baru:
model pilihan P = (1-T)*P_pref + T*P_rec (Model_Simulasi_Inti.md §3.1-3.2), trust
Beta-count (§3.3), dan renege via patience (§4). Menggantikan test_user.py lama yang
menguji formula MXL/trust-EMA yang sudah dihapus (lihat archive/old_design_code/tests/)."""
import numpy as np
import pytest

from marl_spklu.env.user import User, UserState, feasible_candidates, soc_physical_range_km


def make_user(w1=0.0, w2=0.0, w3=0.0, w4=0.0, w5=0.0, **kw):
    return User("U_TEST", w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, **kw)


# ---------------------------------------------------------------------------
# V1.1a -- Trust: T = alpha/(alpha+beta), inisialisasi default T0=0.5
# ---------------------------------------------------------------------------

def test_trust_init_default():
    u = make_user()
    assert u.trust == pytest.approx(0.5)


def test_trust_property_matches_alpha_beta_ratio():
    u = make_user(trust_alpha0=3.0, trust_beta0=1.0)
    assert u.trust == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# V1.1b -- update_trust: DeltaW ABSOLUT |actual - estimasi|, tiga zona
# (Pemodelan_Variasi_Distribusi.md §7.2 -- simetris, bukan bertanda)
# ---------------------------------------------------------------------------

def test_update_trust_reward_zone_small_deviation_either_direction():
    """|DeltaW| kecil -> zona reward, TERLEPAS dari arah (aktual lebih cepat ATAU lebih
    lambat dari estimasi) -- simetris, beda dari versi bertanda sebelumnya.

    Deviasi uji DITURUNKAN dari DELTAW_TOL_LOW (bukan angka mutlak hardcoded): ambang
    ini proporsional thd `wait_mean` skenario (k1*wait_mean, Model_Simulasi_Inti.md §3.3)
    sehingga BERUBAH tiap kali baseline dikalibrasi ulang. Versi lama memakai |dW|=3.0
    yang valid saat tol_low~5 mnt, tapi jatuh ke zona NETRAL setelah rekalibrasi
    tau=0,68 (tol_low=2,095 mnt) -- test gagal padahal kode sesuai spesifikasi."""
    from marl_spklu.env.user import DELTAW_TOL_LOW
    dev = 0.5 * DELTAW_TOL_LOW  # dijamin di dalam zona reward, berapa pun kalibrasinya

    u_early = make_user()
    a0 = u_early.trust_alpha
    u_early.update_trust(est_wait=20.0, actual_wait=20.0 - dev)  # lebih cepat
    assert u_early.trust_alpha > a0
    assert u_early.trust_beta == pytest.approx(1.0)
    assert u_early.trust > 0.5

    u_late = make_user()
    u_late.update_trust(est_wait=20.0, actual_wait=20.0 + dev)  # lebih lambat
    assert u_late.trust_alpha == pytest.approx(u_early.trust_alpha)  # simetris


def test_update_trust_zona_penalti_mode_abs_simetris(monkeypatch):
    """Mode `abs` (versi ke-2 spesifikasi): |DeltaW| besar menghukum ke DUA arah.

    Dipertahankan sbg tes karena `abs` masih dipakai sbg lengan pembanding pada faktorial
    aturan-trust (`_uji_aturan_trust.py`) -- bukan spesifikasi mati.
    """
    import marl_spklu.env.user as U
    monkeypatch.setattr(U, "TRUST_PENALTY_MODE", "abs")
    big = U.DELTAW_TOL_HIGH + 5.0

    u_much_earlier = make_user()
    b0 = u_much_earlier.trust_beta
    u_much_earlier.update_trust(est_wait=big, actual_wait=0.0)   # jauh lebih CEPAT
    assert u_much_earlier.trust_beta > b0

    u_much_later = make_user()
    u_much_later.update_trust(est_wait=0.0, actual_wait=big)     # jauh lebih LAMBAT
    assert u_much_later.trust_beta == pytest.approx(u_much_earlier.trust_beta)


def test_update_trust_zona_penalti_mode_signed_asimetris(monkeypatch):
    """Mode `signed` (BAKU, versi ke-3): hanya KETERLAMBATAN yang menghukum.

    Over-estimasi (tiba lebih cepat dari janji) bersifat netral pada zona penalti -- inilah
    perubahan yang menghentikan erosi trust (Tahap 4). Dilaporkan sebagai ablasi
    metodologis, bukan kontribusi; lihat RENCANA_PENELITIAN_MENYELURUH.md §4/H3.
    """
    import marl_spklu.env.user as U
    monkeypatch.setattr(U, "TRUST_PENALTY_MODE", "signed")
    big = U.DELTAW_TOL_HIGH + 5.0

    u_much_earlier = make_user()
    b0 = u_much_earlier.trust_beta
    u_much_earlier.update_trust(est_wait=big, actual_wait=0.0)   # jauh lebih CEPAT
    assert u_much_earlier.trust_beta == pytest.approx(b0), (
        "over-estimasi tak boleh menghukum pada mode signed")

    u_much_later = make_user()
    b0_late = u_much_later.trust_beta
    u_much_later.update_trust(est_wait=0.0, actual_wait=big)     # jauh lebih LAMBAT
    assert u_much_later.trust_beta > b0_late


def test_update_trust_mode_baku_adalah_signed():
    """Gerbang regresi: baku BERUBAH dari `abs` ke `signed` (2026). Bila ada yang
    mengembalikannya diam-diam, seluruh hasil Tahap 4 jadi tak dapat ditafsirkan."""
    import marl_spklu.env.user as U
    assert U.TRUST_PENALTY_MODE == "signed"


def test_update_trust_reward_zone_at_boundary():
    """DeltaW = ambang tepat (batas atas zona reward, ~5 menit di skala baseline §7.3) ->
    accuracy_signal=(1-DeltaW/tol)=0, alpha naik minimal (tapi masih masuk cabang alpha,
    bukan netral)."""
    from marl_spklu.env.user import DELTAW_TOL_LOW
    u = make_user()
    a0 = u.trust_alpha
    u.update_trust(est_wait=10.0, actual_wait=10.0 + DELTAW_TOL_LOW)
    assert u.trust_alpha == pytest.approx(a0)  # eps*(1-tol/tol) = 0 -> tak berubah


def test_update_trust_neutral_zone():
    """tol_low < DeltaW < tol_high -> netral, alpha dan beta TIDAK berubah sama sekali."""
    from marl_spklu.env.user import DELTAW_TOL_LOW, DELTAW_TOL_HIGH
    mid = (DELTAW_TOL_LOW + DELTAW_TOL_HIGH) / 2
    u = make_user()
    a0, b0 = u.trust_alpha, u.trust_beta
    u.update_trust(est_wait=0.0, actual_wait=mid)
    assert u.trust_alpha == pytest.approx(a0)
    assert u.trust_beta == pytest.approx(b0)
    assert u.trust == pytest.approx(0.5)


def test_update_trust_penalty_zone():
    """DeltaW >= tol_high (wait jauh lebih lama dari janji) -> zona penalti, beta naik,
    trust turun."""
    from marl_spklu.env.user import DELTAW_TOL_HIGH
    u = make_user()
    b0 = u.trust_beta
    u.update_trust(est_wait=0.0, actual_wait=DELTAW_TOL_HIGH + 5.0)
    assert u.trust_beta > b0
    assert u.trust_alpha == pytest.approx(1.0)
    assert u.trust < 0.5


def test_update_trust_penalty_grows_unbounded_with_delta_w():
    """Beta harus tumbuh MONOTON seiring DeltaW makin besar di zona penalti (tak ada
    plafon) -- beda dari zona reward yang terbatas."""
    from marl_spklu.env.user import DELTAW_TOL_HIGH
    u1 = make_user()
    u1.update_trust(est_wait=0.0, actual_wait=DELTAW_TOL_HIGH * 1.1)
    u2 = make_user()
    u2.update_trust(est_wait=0.0, actual_wait=DELTAW_TOL_HIGH * 3.0)
    assert u2.trust_beta > u1.trust_beta


# ---------------------------------------------------------------------------
# V1.1c -- Model pilihan: U(s) = w1*dist + w2*ln(1+pop) + w3*ln(1+conn) + w4*isPrev
# + w5*soc_urgency, lalu P_pref = softmax(U) (Model_Simulasi_Inti.md §3.1)
# ---------------------------------------------------------------------------

def _two_station_features():
    """Dua SPKLU: A dekat (1km) kurang populer, B jauh (5km) sangat populer."""
    return {
        "A": {"loc": (1.0, 0.0), "pop": 10.0, "conn": 2.0},
        "B": {"loc": (5.0, 0.0), "pop": 1000.0, "conn": 2.0},
    }


def test_decide_spklu_prefers_closer_station_when_distance_dominates():
    """w1 sangat negatif (sensitif jarak), w2 kecil -> harus pilih SPKLU terdekat (A)."""
    u = make_user(w1=-5.0, w2=0.01, w3=0.0, w4=0.0, w5=0.0)
    u.location = (0.0, 0.0)
    chosen = u.decide_spklu([], {}, _two_station_features(), soc_percent=100.0)
    assert chosen == "A"


def test_decide_spklu_prefers_popular_station_when_pop_dominates():
    """w2 sangat besar, w1 kecil -> harus pilih SPKLU jauh tapi jauh lebih populer (B)."""
    u = make_user(w1=-0.01, w2=5.0, w3=0.0, w4=0.0, w5=0.0)
    u.location = (0.0, 0.0)
    chosen = u.decide_spklu([], {}, _two_station_features(), soc_percent=100.0)
    assert chosen == "B"


def test_decide_spklu_isprev_pulls_toward_habit_station():
    """w4 (isPrev) besar & positif -> user kembali ke prev_spklu meski itu bukan pilihan
    terbaik dari sisi jarak/popularitas semata."""
    u = make_user(w1=-0.5, w2=0.1, w3=0.0, w4=10.0, w5=0.0)
    u.location = (0.0, 0.0)
    u.prev_spklu = "B"  # habit ke B, meski B lebih jauh & tak jauh lebih populer
    chosen = u.decide_spklu([], {}, _two_station_features(), soc_percent=100.0)
    assert chosen == "B"


def test_decide_spklu_soc_urgency_increases_distance_penalty_at_low_soc():
    """w5 positif berinteraksi dgn (1-soc/100)*dist -- pada SoC rendah, penalti jarak ke
    SPKLU jauh (B) makin besar drpd SoC tinggi, mendorong balik ke SPKLU dekat (A)."""
    feats = _two_station_features()
    # w1 dibuat netral-lemah, w2 cukup kuat shg high-SoC memilih B, tapi w5 besar shg
    # low-SoC membalik pilihan ke A karena penalti soc_urgency menghukum jarak B yang jauh.
    kwargs = dict(w1=-0.1, w2=0.6, w3=0.0, w4=0.0, w5=-2.0)
    u_high_soc = make_user(**kwargs)
    u_high_soc.location = (0.0, 0.0)
    chosen_high = u_high_soc.decide_spklu([], {}, feats, soc_percent=100.0)

    u_low_soc = make_user(**kwargs)
    u_low_soc.location = (0.0, 0.0)
    chosen_low = u_low_soc.decide_spklu([], {}, feats, soc_percent=5.0)

    assert chosen_high == "B"
    assert chosen_low == "A"


def test_decide_spklu_no_balk_full_queue_still_feasible():
    """Balk DIHAPUS (Model_Simulasi_Inti.md §4): queue_lengths besar TIDAK mengeluarkan
    SPKLU dari kandidat -- beda dari mekanisme lama (BALK_RATIO)."""
    u = make_user(w1=-0.01, w2=5.0, w3=0.0, w4=0.0, w5=0.0)
    u.location = (0.0, 0.0)
    huge_queue = {"A": 999, "B": 999}
    chosen = u.decide_spklu([], {}, _two_station_features(), soc_percent=100.0,
                             queue_lengths=huge_queue)
    # Tetap memilih B (populer) meski antreannya "tak masuk akal" -- karena tak ada balk.
    assert chosen == "B"


def test_decide_spklu_full_mix_with_recommendation_when_trust_high():
    """T tinggi -> pilihan didominasi P_rec (rekomendasi), bukan P_pref, walau P_pref
    kuat condong ke SPKLU lain."""
    u = make_user(w1=-0.01, w2=5.0, w3=0.0, w4=0.0, w5=0.0)  # P_pref condong ke B
    u.location = (0.0, 0.0)
    u.trust_alpha, u.trust_beta = 999.0, 1.0  # T mendekati 1.0
    est_waits = {"A": 5.0, "B": 100.0}  # A direkomendasikan dgn wait pendek
    chosen = u.decide_spklu(["A"], est_waits, _two_station_features(), soc_percent=100.0, gamma=0.5)
    assert chosen == "A"


def test_decide_spklu_probabilities_sum_to_one():
    u = make_user(w1=-0.3, w2=0.5, w3=0.1, w4=1.0, w5=0.0)
    u.location = (0.0, 0.0)
    u.decide_spklu(["A"], {"A": 5.0}, _two_station_features(), soc_percent=80.0)
    assert u.last_final_probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(u.last_final_probs >= 0)


# ---------------------------------------------------------------------------
# V1.1d -- Renege: patience terlampaui saat QUEUING (Model_Simulasi_Inti.md §4)
# ---------------------------------------------------------------------------

def test_no_renege_within_patience():
    u = make_user(patience_minutes=60.0)
    u.state = UserState.QUEUING
    u.wait_time = 0.0
    reneged = u.step(dt_minutes=30.0)
    assert not reneged
    assert u.state == UserState.QUEUING


def test_renege_triggers_when_patience_exceeded():
    u = make_user(patience_minutes=60.0)
    u.state = UserState.QUEUING
    u.wait_time = 50.0
    reneged = u.step(dt_minutes=15.0)  # 50+15=65 >= 60
    assert reneged
    assert u.state == UserState.RENEGED


def test_default_patience_effectively_disables_renege():
    """Default patience = 1 hari (1440 mnt) -- Pemodelan_Variasi_Distribusi.md §9. Sesi
    antrian realistis (puluhan-ratusan menit) tak boleh memicu renege pada default."""
    u = make_user()  # patience default
    u.state = UserState.QUEUING
    u.wait_time = 0.0
    for _ in range(20):  # 20 x 15 menit = 300 menit, jauh di bawah 1440
        reneged = u.step(dt_minutes=15.0)
        assert not reneged
    assert u.state == UserState.QUEUING


# ---------------------------------------------------------------------------
# V1.1e -- feasible_candidates: jangkauan fisik SoC & radius/rasio kemauan
# ---------------------------------------------------------------------------

def test_soc_physical_range_scales_with_soc_and_capacity():
    r_full = soc_physical_range_km(100.0, battery_kwh=40.0, consumption_kwh_km=0.1)
    r_half = soc_physical_range_km(50.0, battery_kwh=40.0, consumption_kwh_km=0.1)
    assert r_full == pytest.approx(400.0)
    assert r_half == pytest.approx(200.0)


def test_feasible_candidates_excludes_unreachable_station():
    """Catatan: tanpa willingness_radius_km/ratio, feasible_candidates mengembalikan
    SEMUA SPKLU tanpa filter jangkauan fisik (perilaku sengaja -- lihat docstring
    fungsi). Filter jangkauan fisik SoC baru aktif begitu salah satu parameter
    kemauan diisi (di sini radius diisi sangat besar agar TAK jadi pembatas
    tambahan -- murni menguji reach fisik dari SoC/baterai)."""
    feats = {"NEAR": {"loc": (1.0, 0.0)}, "FAR": {"loc": (100.0, 0.0)}}
    ids = feasible_candidates((0.0, 0.0), soc_percent=10.0, spklu_features=feats,
                              willingness_radius_km=1e6,
                              battery_kwh=20.0, consumption_kwh_km=0.2)
    # reach = min(0.1*20/0.2, 1e6) = 10km -> FAR (100km) tak terjangkau, NEAR terjangkau.
    assert "NEAR" in ids
    assert "FAR" not in ids


def test_feasible_candidates_always_returns_at_least_one():
    """Limp mode: walau tak ada yg benar2 terjangkau, tetap kembalikan >=1 (terdekat)."""
    feats = {"FAR": {"loc": (1000.0, 0.0)}}
    ids = feasible_candidates((0.0, 0.0), soc_percent=1.0, spklu_features=feats,
                              battery_kwh=10.0, consumption_kwh_km=0.5)
    assert ids == ["FAR"]
