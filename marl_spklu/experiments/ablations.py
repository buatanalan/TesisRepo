"""Context manager utk 4 varian ablasi Bab IV.2 draf tesis (para. 706-710):

  Ablasi-T (tanpa modul trust)         -> constant_trust()
  Ablasi-P (tanpa prediktor sadar-koordinasi) -> load_unaware_forecaster()
  Ablasi-R (tanpa penalti flocking pd reward) -> RewardCalculator(alpha_flock=0.0)
                                                  (bukan context manager -- cukup param)
  Ablasi-E (tanpa encoder riwayat)     -> no_history_encoder()

Pola sama dgn `force_full_compliance` di marl_spklu/experiments/harness.py:
monkeypatch sementara pd satu titik hook, dipulihkan otomatis via try/finally.
Setiap ablasi mengisolasi SATU komponen; kombinasikan context manager kalau perlu
menonaktifkan >1 komponen sekaligus (nested `with`).

Pemakaian tipikal (skenario ablasi, lihat rencana_pengujian_demonstrasi_evaluasi.md
Bagian 3):

    from marl_spklu.experiments.ablations import constant_trust
    from marl_spklu.rl.ppo import TorchContinuingTrainer

    with constant_trust():
        tr = TorchContinuingTrainer(dataset, ...)
        ...  # trust semua user tetap 0.5 sepanjang run ini
"""
import contextlib

from marl_spklu.env import user as _user_mod
from marl_spklu.env.user import User
from marl_spklu.rl import forecaster as _forecaster_mod
from marl_spklu.rl.policy import HPPOPolicy


@contextlib.contextmanager
def constant_trust(value: float = None):
    """Ablasi-T / mu_hat statis PDQN baseline (Tahap 0): trust user dibekukan sepanjang
    simulasi -- update_trust() jadi no-op. Jika `value` None, dibekukan pd nilai inisial
    tiap user (0.5, INIT_TRUST) tanpa mengubahnya (perilaku ablasi lama). Jika `value`
    diberikan, SEMUA user dipaksa ke nilai itu (mis. 0.2/0.5/0.8 utk sensitivity analysis
    Bagian 5.3 spesifikasi_teknis_pdqn_baseline.md) sebelum dibekukan.
    Mengisolasi kontribusi estimasi trust dari observasi kepatuhan thd Objektif 3
    & kaskadenya ke acceptance (Objektif 2)."""
    orig = User.update_trust
    orig_init = User.__init__

    def noop(self, est_wait, actual_wait, alpha=1.0, beta=0.5):
        return None

    if value is not None:
        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            # `trust` adalah @property turunan trust_alpha/trust_beta (T=a/(a+b), model
            # Beta-count) -- BUKAN lagi atribut biasa, jadi tak bisa di-assign langsung.
            # Setel alpha/beta agar rasionya PERSIS `value`; skala absolut tak relevan
            # krn update_trust() sudah di-no-op-kan (dibekukan), rasio takkan bergeser.
            if not (0.0 < value < 1.0):
                raise ValueError(
                    f"constant_trust(value={value}) harus di (0,1) -- trust_alpha/beta "
                    "tak bisa merepresentasikan rasio di luar itu (mis. 0 atau 1 exact).")
            self.trust_alpha = float(value)
            self.trust_beta = float(1.0 - value)

        User.__init__ = patched_init

    User.update_trust = noop
    try:
        yield
    finally:
        User.update_trust = orig
        if value is not None:
            User.__init__ = orig_init


@contextlib.contextmanager
def constant_trust_shadow(value: float):
    """Varian `constant_trust` yang MEMPERTAHANKAN dinamika trust asli (`update_trust`
    TETAP AKTIF, TIDAK di-no-op-kan) -- HANYA trust yang dipakai MENCAMPUR keputusan
    (`decide_spklu`, via `User.trust_effective`) yang dibekukan ke `value`. `User.trust`
    (properti asli, dari trust_alpha/trust_beta yang terus diperbarui normal) jadi murni
    DIAGNOSTIK: "ke arah mana trust SEHARUSNYA bergerak kalau tak dibekukan", tanpa
    memengaruhi dinamika pilihan pengguna yang sungguh terjadi (variabel kontrol
    eksperimen tetap `value`, dipertahankan konstan).

    Beda dgn `constant_trust(value)`: yang itu MEMATIKAN update_trust total (tak ada
    info trust-bayangan sama sekali). Ini menyalakannya lagi tapi firewall penuh dari
    pengambilan keputusan DAN dari observasi kritik CTDE (`rollout.py::_build_critic_obs`
    sudah dipastikan baca `trust_effective`, bukan `trust` mentah -- lihat komentar di
    sana). Cocok utk Tahap 2 revisit: trust statis sbg variabel kontrol, sambil tetap
    mengukur "tekanan performativitas laten" yang terselubung di baliknya."""
    if not (0.0 < value < 1.0):
        raise ValueError(f"constant_trust_shadow(value={value}) harus di (0,1).")
    orig_init = User.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._trust_override = float(value)

    User.__init__ = patched_init
    try:
        yield
    finally:
        User.__init__ = orig_init


@contextlib.contextmanager
def binary_recommendation_mode(mu_hat: float, stochastic: bool = False):
    """Model keputusan pengguna versi PDQN BASELINE (Tahap 0, spesifikasi_teknis_pdqn_
    baseline.md §2.1): kanal rekomendasi berupa VEKTOR f_rec atas seluruh SPKLU (one-hot
    -- tepat satu SPKLU direkomendasikan), dicampur dgn preferensi di skala UTILITAS:

        score(j) = (1 - mu_hat) * u_pref(j) + mu_hat * f_rec(j)
        pilihan  = argmax_j score(j)                       [stochastic=False, DEFAULT]
        pilihan  ~ Categorical(softmax(score))             [stochastic=True]

    `stochastic=True` adalah bentuk yang ditulis dokumen spesifikasi (P(a=j)=softmax(score)
    lalu disampel); argmax adalah kasus limit temperatur->0. Keduanya disediakan karena
    keduanya memberi rezim perilaku berbeda: argmax membuat kepatuhan jadi AMBANG keras
    (patuh persis bila mu/(1-mu) > selisih utilitas ke favorit), sedangkan sampling
    memberi respons bergradasi dan menambah penyebaran alami beban antar-SPKLU.

    Bobot = `mu_hat` MURNI (0.2/0.5/0.8): `trust` maupun `w_i` tidak dibaca. Kanal EstWait
    dimatikan sepenuhnya -- memang sudah tak berfungsi untuk rekomendasi tunggal (P_rec
    ternormalisasi selalu 1 berapa pun EstWait), jadi mode ini membuat desainnya jujur.

    HARUS membungkus SELURUH skenario dalam satu eksperimen (Natural, Greedy, PDQN),
    bukan hanya agen PDQN -- kalau tidak, agen dibandingkan di bawah model keputusan
    pengguna yang berbeda dan perbandingannya tidak sah. Natural otomatis konsisten
    (tanpa rekomendasi -> f_rec nol -> murni preferensi).

    Jalur H-PPO/MARL TIDAK terpengaruh (default `rec_mode="estwait"` tetap berlaku di luar
    context ini), karena artefak usulan membutuhkan trust dinamis yang justru digerakkan
    oleh akurasi EstWait."""
    orig_init = User.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.rec_mode = "binary_utility"
        self.rec_weight = float(mu_hat)
        self.rec_stochastic = bool(stochastic)

    User.__init__ = patched_init
    try:
        yield
    finally:
        User.__init__ = orig_init


@contextlib.contextmanager
def load_unaware_forecaster():
    """Ablasi-P: prediktor waktu tunggu (LearnedForecaster & FormulaForecaster, keduanya
    lewat extract_features()) kehilangan Kelompok fitur 4 (aktivitas sistem / jumlah
    rekomendasi dlm jendela terkini, phi_activity) -- dipaksa 0 apapun input aslinya.
    Sisa fitur (antrean fisik saat ini, karakteristik statis stasiun, dst.) tetap ada.
    Mengisolasi kontribusi antisipasi beban thd koordinasi anti-flocking (Objektif 1,
    constraint C5)."""
    orig = _forecaster_mod.extract_features

    def patched(spklu, time_now_min, user=None, soc=50.0, recent_recs_count=0, dist_override=None):
        return orig(spklu, time_now_min, user=user, soc=soc, recent_recs_count=0,
                   dist_override=dist_override)

    _forecaster_mod.extract_features = patched
    try:
        yield
    finally:
        _forecaster_mod.extract_features = orig


@contextlib.contextmanager
def no_history_encoder():
    """Ablasi-E: encoder riwayat (hist_lstm) diganti representasi observasi sesaat --
    c_t dipaksa nol (tanpa memori interaksi K-langkah), apapun isi `hist` yg masuk.
    Mengisolasi kontribusi inferensi trust berbasis riwayat thd personalisasi
    rekomendasi per-pengguna (Objektif 2 & 3)."""
    orig = HPPOPolicy._encode_hist

    def zeroed(self, hist):
        b = hist.shape[0]
        return hist.new_zeros((b, self.hist_hidden))

    HPPOPolicy._encode_hist = zeroed
    try:
        yield
    finally:
        HPPOPolicy._encode_hist = orig


# Ablasi-R TIDAK butuh monkeypatch -- cukup:
#   from marl_spklu.rl.rewards import RewardCalculator
#   reward_calc = RewardCalculator(alpha_flock=0.0)
# lalu teruskan ke TorchContinuingTrainer(..., reward_calc=reward_calc) seperti biasa.


@contextlib.contextmanager
def initial_trust(value: float):
    """Trust AWAL semua pengguna = `value` (bukan INIT_TRUST=0.5 bawaan), TAPI trust
    tetap DINAMIS sesudahnya (update_trust TIDAK dibekukan -- beda dari `constant_trust`
    yang membekukan). Dipakai utk menguji sensitivitas thd titik awal trust tanpa
    mengubah dinamika performatif itu sendiri. HARUS dipasang SEBELUM
    `Simulator.load_from_dataset()` (mempengaruhi User.__init__)."""
    orig_init = User.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # `trust` adalah @property turunan trust_alpha/trust_beta (T=a/(a+b)) -- BUKAN
        # atribut biasa, tak bisa di-assign langsung (sama spt fix constant_trust()).
        # Setel alpha/beta agar rasionya PERSIS `value`; skala absolut (alpha+beta=1)
        # dipertahankan KECIL spy update_trust() berikutnya (yg TETAP aktif di sini,
        # beda dari constant_trust) bisa langsung menggeser rasio scr wajar, bukan
        # butuh observasi sangat banyak dulu utk mengalahkan skala awal yg besar.
        if not (0.0 < value < 1.0):
            raise ValueError(
                f"initial_trust(value={value}) harus di (0,1) -- trust_alpha/beta "
                "tak bisa merepresentasikan rasio di luar itu (mis. 0 atau 1 exact).")
        self.trust_alpha = float(value)
        self.trust_beta = float(1.0 - value)

    User.__init__ = patched_init
    try:
        yield
    finally:
        User.__init__ = orig_init


@contextlib.contextmanager
def gamma_est_wait(value: float):
    """Ganti sensitivitas P_rec terhadap waktu tunggu yang dijanjikan sistem --
    gamma pada P_rec = softmax(exp(-gamma*wait)) (`user.py::GAMMA_DEFAULT`, dipakai
    `User.decide_spklu`). BUKAN faktor diskon PPO/GAE (`--gamma` CLI pipeline RL,
    parameter ALGORITMA berbeda meski kebetulan sama-sama disebut "gamma" -- lihat
    catatan di `_run_master_pure_hybrid_ppo_pipeline.py`). Semakin besar nilainya,
    semakin tajam pengguna membedakan SPKLU berdasar selisih kecil pada estimasi
    waktu tunggu (mendekati keputusan hampir-deterministik ke SPKLU tercepat);
    semakin kecil, semakin longgar/acak (mendekati acuh thd perbedaan waktu tunggu).
    `user.py::GAMMA_SWEEP` sudah menyediakan titik sapuan baku (x0.5/x1/x2 dari
    GAMMA_DEFAULT, half-life = wait_mean baseline).

    `decide_spklu` dipanggil TANPA argumen `gamma` eksplisit oleh `Simulator`
    (mengandalkan default parameter `gamma: float = GAMMA_DEFAULT` yang terikat
    pada saat fungsi didefinisikan) -- override modul `GAMMA_DEFAULT` SETELAH impor
    karena itu TIDAK berpengaruh. Context manager ini membungkus `User.decide_spklu`
    agar `gamma=value` selalu disuntikkan, terlepas dari apa yang dipanggil caller."""
    orig_decide = User.decide_spklu

    def patched_decide(self, recommendations, estimated_waits, spklu_features,
                       speed_kmh: float = 40.0, gamma: float = None,  # noqa: unused, dipaksa `value`
                       soc_percent: float = 50.0, **kwargs):
        return orig_decide(self, recommendations, estimated_waits, spklu_features,
                           speed_kmh=speed_kmh, gamma=value, soc_percent=soc_percent,
                           **kwargs)

    User.decide_spklu = patched_decide
    try:
        yield
    finally:
        User.decide_spklu = orig_decide


@contextlib.contextmanager
def amplify_preference_weights(pop_mult: float = 2.0, conn_mult: float = 2.0):
    """Perbesar KOEFISIEN preferensi populasi (BETA_POP/BETA_CONN, mean draw individual
    Mixed Logit) sebesar `pop_mult`/`conn_mult` -- BUKAN mengubah kapasitas/popularitas
    SPKLU di dataset, melainkan seberapa KUAT pengguna tertarik pada atribut itu saat
    memilih (User.__init__ menggambar beta_pop/beta_conn dari Normal(BETA_POP*mult,
    BETA_POP_SIGMA*mult) -- sigma ikut diskalakan proporsional supaya rasio variasi/mean
    tetap konsisten, bukan cuma mean yg bergeser).

    Efek: pengguna makin condong ke stasiun POPULER/BANYAK-KONEKTOR terlepas dari
    utilisasi SESAAT -- menciptakan tarikan yg BERTENTANGAN dgn argmin-utilisasi Greedy
    (stasiun populer/besar sering JUSTRU yg paling ramai), uji apakah PDQN (yg bisa belajar
    trade-off ini) unggul saat preferensi personal makin kuat menarik ke arah "salah"
    menurut metrik pemerataan.

    HARUS dipasang SEBELUM `Simulator.load_from_dataset()` (mempengaruhi User.__init__)."""
    orig_pop, orig_pop_sigma = _user_mod.BETA_POP, _user_mod.BETA_POP_SIGMA
    orig_conn, orig_conn_sigma = _user_mod.BETA_CONN, _user_mod.BETA_CONN_SIGMA
    _user_mod.BETA_POP = orig_pop * pop_mult
    _user_mod.BETA_POP_SIGMA = orig_pop_sigma * pop_mult
    _user_mod.BETA_CONN = orig_conn * conn_mult
    _user_mod.BETA_CONN_SIGMA = orig_conn_sigma * conn_mult
    try:
        yield
    finally:
        _user_mod.BETA_POP, _user_mod.BETA_POP_SIGMA = orig_pop, orig_pop_sigma
        _user_mod.BETA_CONN, _user_mod.BETA_CONN_SIGMA = orig_conn, orig_conn_sigma


ABLATIONS = {
    "Ablasi-T": constant_trust,
    "Ablasi-P": load_unaware_forecaster,
    "Ablasi-E": no_history_encoder,
    # "Ablasi-R" sengaja tak dimasukkan -- bukan context manager, lihat catatan di atas.
}
