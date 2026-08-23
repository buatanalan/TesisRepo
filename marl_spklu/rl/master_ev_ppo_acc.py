"""Kepala kritik ke-5 -- STREAM_ACCURACY, uji reward akurasi-janji (`RewardCalculator.
accuracy_reward`). File TERPISAH dari `master_ev_ppo_policy.py` SENGAJA (permintaan
eksplisit "kelas baru, lebih bersih") -- lengan `eq1` (H1a, n_critics=4) yang sudah
dilatih & dianalisis TETAP UTUH, tak ada risiko menyentuh checkpoint yang sudah ada.

LATAR
-----
Diagnosis `_diagnosis_rec_activity_vs_deltaW.py` (2026-08-23): |Delta W| pada trip
patuh melonjak 5,8x saat `rec_activity` tinggi (rho=0,36-0,46). Dua percobaan
memperbaiki FORECASTER (menambah "EV hantu" ke estimasi) GAGAL konsisten -- makin
besar koreksinya, makin buruk (lihat `kalibrasi_congestion_aware_vwf.json`). Pendekatan
di sini BEDA KELAS: bukan memperbaiki janji setelah keputusan diambil, tapi menghukum
KEBIJAKAN saat janjinya (utk trip yg dipatuhi) meleset jauh -- mendorong agen sendiri
menghindari merekomendasikan ke situasi rawan penumpukan.

`MasterEVPPOPolicy` (n_critics generik via `nn.Linear(hidden, n_critics)`) SUDAH
mendukung n_critics=5 tanpa perubahan -- HANYA Transition & RolloutAgent yang perlu
tahu tentang aliran ke-5."""
import numpy as np

from marl_spklu.rl.master_ev_ppo_policy import (MasterEV4Transition, MasterEVPPORolloutAgent,
                                                STREAM_WAIT, STREAM_PROX, STREAM_GLOBAL3,
                                                STREAM_EQUITY)

# Aliran ke-5, HANYA aktif utk n_critics=5 (MasterEV5Transition) -- lihat catatan
# STREAM_EQUITY di master_ev_ppo_policy.py utk alasan kenapa TERPISAH, bukan ditumpuk
# ke stream yg sudah ada (statistik akurasi-janji: jarang tapi ekstrem saat meleset,
# beda profil dari GLOBAL3/EQUITY yg lebih halus/kontinu).
STREAM_ACCURACY = 4


class MasterEV5Transition(MasterEV4Transition):
    """`MasterEV4Transition` + `reward_streams` berukuran 5 (STREAM_ACCURACY=4)."""
    __slots__ = ()

    def __init__(self, obs, critic_obs, hist, mask, chosen_indices, n_rec, logp, value, step,
                pref_hist=None):
        super().__init__(obs, critic_obs, hist, mask, chosen_indices, n_rec, logp, value, step,
                         pref_hist=pref_hist)
        self.reward_streams = np.zeros(5, dtype=np.float64)

    def reward_vec(self, n_critics: int) -> np.ndarray:
        if n_critics == 1:
            return np.array([self.reward_streams.sum()], dtype=np.float64)
        if n_critics != 5:
            raise ValueError(f"n_critics={n_critics} != 5 (MasterEV5Transition)")
        return self.reward_streams


class MasterEVPPOAccRolloutAgent(MasterEVPPORolloutAgent):
    """Override MINIMAL atas `MasterEVPPORolloutAgent`: HANYA `get_recommendation`
    (pilih `MasterEV5Transition` saat n_critics==5) dan `on_charge_complete` (tambah
    suku `accuracy_reward` ke STREAM_ACCURACY). `on_decision`/`on_step_end` DIWARISI
    tanpa perubahan -- logikanya identik lengan `eq1`, cuma aliran reward ke-5 yg baru."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._split_accuracy = (int(getattr(self.policy, "n_critics", 1)) == 5)

    def on_charge_complete(self, user):
        if not self._split_accuracy:
            return super().on_charge_complete(user)
        tr = self._user_trip_tr.pop(user.user_id, None)
        if tr is not None:
            if tr.complied:
                tr.add_reward(
                    self.rc.wait_reward(tr.wait_default, user.wait_time, tr.disp_estwait),
                    STREAM_WAIT)
                if self.rc.alpha_trust != 0.0:
                    delta_trust = float(user.trust) - tr.trust_before
                    tr.add_reward(self.rc.trust_shaping_reward(delta_trust), STREAM_WAIT)
                if getattr(self.rc, "alpha_acc", 0.0) != 0.0:
                    delta_w_abs = abs(user.wait_time - tr.disp_estwait)
                    tr.add_reward(self.rc.accuracy_reward(delta_w_abs), STREAM_ACCURACY)
            tr.resolved = True
            if tr.complied and user.interaction_history:
                complied_v, disp_v, wdef_v, _ = user.interaction_history[-1]
                realized_gap_norm = (user.wait_time - tr.disp_estwait) / self.wait_scale
                user.interaction_history[-1] = (complied_v, disp_v, wdef_v, realized_gap_norm)

    def get_recommendation(self, feasible_spklus: dict):
        if not self._split_accuracy:
            return super().get_recommendation(feasible_spklus)
        # Duplikat MasterEVPPORolloutAgent.get_recommendation, HANYA beda TransitionCls
        # (MasterEV5Transition) -- lihat modul induk utk komentar lengkap tiap baris.
        user = self.sim._current_spawn_user
        soc = self.sim._current_spawn_soc
        time_now = self.sim.current_step * self.sim.dt_minutes
        feasible_ids = list(feasible_spklus.keys())

        _rich_unused, default_idx, wait_hat = self._build_obs(user, soc, feasible_ids, time_now)

        from marl_spklu.rl.master_paper_obs import build_joint_obs_master_ev
        joint_obs = build_joint_obs_master_ev(self.sim, self.sids, time_now, user, soc)
        obs_flat = joint_obs.T.reshape(-1).astype(np.float32)
        mask = self._feasible_mask(feasible_ids)
        hist = self._build_hist(user)

        if self._use_pref:
            pref_hist = self._build_pref_hist(user)
            act = self.policy.act(obs_flat, mask, hist, k=self.k, critic_obs_np=obs_flat,
                                  epsilon=self.epsilon, threshold=self.threshold,
                                  pref_hist_np=pref_hist)
        else:
            pref_hist = None
            act = self.policy.act(obs_flat, mask, hist, k=self.k, critic_obs_np=obs_flat,
                                  epsilon=self.epsilon, threshold=self.threshold)

        chosen_indices = act["chosen_indices"]
        n_rec = act["n_rec"]
        recs = [self.sids[i] for i in chosen_indices]

        baseline = self.forecaster.predict(feasible_spklus, time_now, user=user, soc=soc, sim=self.sim)
        rec_disps = {self.sids[i]: float(baseline.get(self.sids[i], 0.0)) for i in chosen_indices}
        estimated_waits = {sid: rec_disps[sid] if sid in rec_disps else float("inf")
                           for sid in feasible_ids}

        primary_idx = chosen_indices[0] if chosen_indices else int(np.nonzero(mask)[0][0])
        primary_disp = rec_disps.get(self.sids[primary_idx], 0.0)

        idx_arr = np.zeros(self.k, dtype=np.int64)
        idx_arr[:n_rec] = chosen_indices

        tr = MasterEV5Transition(obs_flat, obs_flat, hist, mask, idx_arr, n_rec,
                                 act["logp"], act["value"], self.sim.current_step,
                                 pref_hist=pref_hist)
        tr.disp_estwait = primary_disp
        tr.wait_default = float(self.sim.compute_virtual_wait(
            user, self.sim.spklus[self.sids[default_idx]], time_now)
        ) if default_idx != primary_idx else primary_disp
        self.transitions.append(tr)
        recent_rec_count = float(self.sim.recent_recs.get(self.sids[primary_idx], 0))
        self._pending = (tr, estimated_waits, primary_idx, wait_hat, default_idx, recent_rec_count)
        self._user_trip_tr[user.user_id] = tr
        return recs
