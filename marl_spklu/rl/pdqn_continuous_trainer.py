"""Trainer PDQN KONTINU -- reuse PENUH `TorchContinuingTrainer` (sim persisten lintas
chunk, trust DINAMIS, PPO+GAE differential-return) yang sudah teruji di jalur H-PPO/MARL.
`POLICY_CLS` diarahkan ke `PDQNContinuousPolicy` (lihat pdqn_continuous_policy.py) --
titik-colok yang sengaja disediakan di `TorchPolicyTrainer.POLICY_CLS` (ppo.py) supaya
TIDAK perlu menduplikasi __init__.

Trust TIDAK dibekukan di sini (beda dari PDQN diskrit pdqn_baseline.py yang sengaja
membekukan trust demi keadilan Tahap 0) -- inilah yang membuat setup ini PERFORMATIVE:
janji EstWait yang di-broadcast kebijakan mempengaruhi trust/kepatuhan populasi dari waktu
ke waktu, yang pd gilirannya mengubah distribusi keputusan yang dipelajari kebijakan
berikutnya.

`train()` DI-OVERRIDE (bukan warisan `TorchContinuingTrainer.train()` apa adanya): versi
warisan mereset SELURUH populasi (termasuk trust -> INIT_TRUST) tiap kali horizon dataset
habis (`sim = self._fresh_sim()` di titik batas pass) -- ini menghapus TOTAL efek
performatif yang baru terbentuk, menghasilkan pola gigi-gergaji Gini yang terikat siklus
reset (dibuktikan empiris: lihat catatan eksperimen 7-hari & 30-hari). Di sini, pada
batas pass, HANYA state FISIK (antrean/charging/jadwal spawn) yang di-reset via sim baru;
state KEPERCAYAAN (`trust`, `compliance_history`, `interaction_history`) DIBAWA (carry-
forward) dari populasi lama ke populasi baru (dicocokkan via `user_id`) -- realisasi
"continuing task" yang literal untuk variabel performatif itu sendiri, bukan cuma untuk
state fisik simulator.

`train_rrm()` (BARU): implementasi Repeated Risk Minimization (Perdomo et al. 2020,
"Performative Prediction") yang LEBIH FORMAL drpd `train()` biasa. `train()` memperbarui
kebijakan TIAP chunk SAMBIL distribusi (trust) ikut bergerak BERSAMAAN -- co-adaptasi
online, bukan siklus deploy-beku-ukur-retrain yang terpisah tegas spt didefinisikan RRM.
`train_rrm()` membagi tiap ronde jadi dua fase:
  1. FREEZE  (`freeze_chunks`): kebijakan π_t DIBEKUKAN (tidak ada `ppo.update`), dijalankan
     sampai distribusi (trust) mendekati kesetimbangannya SENDIRI di bawah π_t -- inilah
     D(π_t) yang literal, bukan didekati sambil jalan.
  2. RETRAIN (`retrain_chunks`): `ppo.update` diaktifkan lagi, kebijakan diperbarui jadi
     π_{t+1} dari data yang dikumpulkan di bawah D(π_t) yang sudah menyetimbang.
Barisan metrik akhir-fase-FREEZE tiap ronde (`rrm_trace`) adalah yang diperiksa utk
stabilitas performatif: apakah ia konvergen ke satu titik (π_stabil dgn D(π_stabil)
konsisten), berosilasi, atau divergen.

Pemakaian:
    from marl_spklu.rl.pdqn_continuous_trainer import PDQNContinuousTrainer
    from marl_spklu.rl.forecaster import FormulaForecaster

    tr = PDQNContinuousTrainer("scenario_dataset_30d.json", k=3, rollout_steps=96)
    policy = tr.train(FormulaForecaster(), n_updates=100)

    tr2 = PDQNContinuousTrainer("scenario_dataset_90d_extreme.json", k=3, rollout_steps=96)
    policy, rrm_trace = tr2.train_rrm(FormulaForecaster(), n_rounds=6, freeze_chunks=10, retrain_chunks=10)
"""
import numpy as np

from marl_spklu.env.history_buffer import HistoryBuffer
from marl_spklu.rl.ppo import TorchContinuingTrainer
from marl_spklu.rl.pdqn_continuous_policy import PDQNContinuousPolicy
from marl_spklu.rl.rollout import RLRolloutAgent, _gini


class PDQNContinuousTrainer(TorchContinuingTrainer):
    POLICY_CLS = PDQNContinuousPolicy

    def _fresh_sim_with_history(self, chunk: int):
        sim = self._fresh_sim()
        # HistoryBuffer window = 1 chunk: Simulator.step_once mencatat get_utilization()
        # (fraksi WAKTU konektor terisi, snapshot tiap step) tiap langkah bila `sim.history`
        # terpasang -- inilah utilisasi yg BENAR (bukan served/kapasitas, itu throughput,
        # BUKAN fraksi waktu terpakai; lihat koreksi diskusi). get_recent_utilization(chunk)
        # merata-ratakan snapshot 1 chunk terakhir -> gini atasnya = gini_mean per hari,
        # metrik IDENTIK yg dipakai seluruh eksperimen PDQN diskrit (harness.run_scenario).
        sim.history = HistoryBuffer(list(sim.spklus.keys()), window_size_15m=chunk)
        return sim

    def _carry_forward(self, sim, agent, chunk: int, verbose_tag: str = ""):
        """Buat sim baru (state FISIK bersih) tapi BAWA trust/riwayat populasi lama --
        dipakai baik oleh `train()` maupun `train_rrm()` di titik batas horizon dataset."""
        old_users_by_id = {u.user_id: u for u in sim.users}
        new_sim = self._fresh_sim_with_history(chunk)
        n_matched = 0
        for u in new_sim.users:
            old = old_users_by_id.get(u.user_id)
            if old is not None:
                u.trust = old.trust
                u.compliance_history = list(old.compliance_history)
                u.interaction_history = list(old.interaction_history)
                n_matched += 1
        if self.verbose:
            self._logger.info("  carry-forward trust%s: %d/%d pengguna cocok via user_id",
                              verbose_tag, n_matched, len(new_sim.users))
        agent.sim = new_sim
        agent.sids = list(new_sim.spklus.keys())
        agent.sid_to_idx = {s: i for i, s in enumerate(agent.sids)}
        agent.transitions = []
        agent._pending = None
        agent._user_trip_tr = {}
        agent._prev_gini = None
        return new_sim, agent, 0

    def _run_one_chunk(self, sim, agent, step: int, chunk: int, do_update: bool):
        """Jalankan 1 chunk (~1 hari). do_update=False -> kebijakan DIBEKUKAN (fase FREEZE
        RRM): transisi tetap dikumpulkan (perlu utk trust/reward berjalan wajar) tapi
        TIDAK ADA `ppo.update` -- bobot policy TIDAK berubah. Return (sim, agent, step,
        boundary, metrics|None)."""
        boundary = False
        for _ in range(chunk):
            sim.step_once(step, agent=agent)
            step += 1
            if step >= sim.max_steps:
                boundary = True
                break
        if boundary:
            for t in agent.transitions:
                t.resolved = True
        resolved = [t for t in agent.transitions if t.resolved]
        pending = [t for t in agent.transitions if not t.resolved]

        metrics = None
        stats = {}
        if resolved:
            resolved[-1].done = boundary
            if do_update:
                stats = self.ppo.update(resolved)
            rewards = np.array([t.reward for t in resolved])
            served = np.array([s.total_served for s in sim.spklus.values()], float)
            n_steps_this_chunk = min(chunk, sim.history.current_step or chunk)
            util_avg = sim.history.get_recent_utilization(steps=n_steps_this_chunk)
            trusts = np.array([u.trust for u in sim.users], dtype=float)
            metrics = {"mean_reward": float(rewards.mean()),
                      "acceptance_rate": float(np.mean([t.complied for t in resolved])),
                      "mean_flock_penalty": float(np.mean([t.flock_penalty for t in resolved])),
                      "gini_served": _gini(served), "gini_util": _gini(util_avg),
                      "n_tr": len(resolved), "n_pending": len(pending),
                      "trust_mean": float(trusts.mean()) if trusts.size else 0.5,
                      "trust_std": float(trusts.std()) if trusts.size else 0.0, **stats}

        if boundary:
            sim, agent, step = self._carry_forward(sim, agent, chunk)
        else:
            agent.transitions = pending
        return sim, agent, step, boundary, metrics

    def train(self, forecaster, n_updates: int):
        chunk = self.rollout_steps
        sim = self._fresh_sim_with_history(chunk)
        agent = RLRolloutAgent(self.policy, sim, self.rc, forecaster, self.k,
                               honest_estwait=self.honest_estwait, equity_calc=self.equity_calc,
                               pref_feature_mode=self.pref_feature_mode)
        step = 0
        for _ in range(n_updates):
            it = self._it_global
            sim, agent, step, boundary, info = self._run_one_chunk(sim, agent, step, chunk, do_update=True)
            self._it_global += 1
            if info is not None:
                self.history.append({"iter": it, **info})
                if self.verbose:
                    self._logger.info(
                        "[chunk %3d] R=%+.4f accept=%.2f flock=%.2f | KL=%.4f EV=%.2f | "
                        "gini_srv=%.3f gini_util=%.3f trust=%.3f+-%.3f n=%d pend=%d%s",
                        it, info["mean_reward"], info["acceptance_rate"], info["mean_flock_penalty"],
                        info.get("approx_kl", 0), info.get("explained_var", 0),
                        info["gini_served"], info["gini_util"], info["trust_mean"], info["trust_std"],
                        info["n_tr"], info["n_pending"],
                        " |PASS-BARU(carry-fwd trust)" if boundary else "")
        return self.policy

    def train_rrm(self, forecaster, n_rounds: int = 6, freeze_chunks: int = 10,
                 retrain_chunks: int = 10):
        """Repeated Risk Minimization eksplisit (lihat docstring modul). Return
        (policy, rrm_trace) -- rrm_trace: list of dict per ronde berisi metrik AKHIR
        fase FREEZE (D(pi_t) yg sudah menyetimbang) dan AKHIR fase RETRAIN (pi_{t+1})."""
        chunk = self.rollout_steps
        sim = self._fresh_sim_with_history(chunk)
        agent = RLRolloutAgent(self.policy, sim, self.rc, forecaster, self.k,
                               honest_estwait=self.honest_estwait, equity_calc=self.equity_calc,
                               pref_feature_mode=self.pref_feature_mode)
        step = 0
        rrm_trace = []
        for r in range(n_rounds):
            freeze_last = None
            for _ in range(freeze_chunks):
                sim, agent, step, boundary, info = self._run_one_chunk(
                    sim, agent, step, chunk, do_update=False)
                if info is not None:
                    freeze_last = info
            retrain_last = None
            for _ in range(retrain_chunks):
                sim, agent, step, boundary, info = self._run_one_chunk(
                    sim, agent, step, chunk, do_update=True)
                if info is not None:
                    retrain_last = info
                    self._it_global += 1
            rec = {"round": r,
                  "freeze_gini_util": freeze_last["gini_util"] if freeze_last else None,
                  "freeze_gini_served": freeze_last["gini_served"] if freeze_last else None,
                  "freeze_trust": freeze_last["trust_mean"] if freeze_last else None,
                  "freeze_acceptance": freeze_last["acceptance_rate"] if freeze_last else None,
                  "retrain_gini_util": retrain_last["gini_util"] if retrain_last else None,
                  "retrain_gini_served": retrain_last["gini_served"] if retrain_last else None,
                  "retrain_trust": retrain_last["trust_mean"] if retrain_last else None,
                  "retrain_acceptance": retrain_last["acceptance_rate"] if retrain_last else None}
            rrm_trace.append(rec)
            if self.verbose:
                self._logger.info(
                    "[RRM ronde %2d] FREEZE(pi_%d beku): gini_util=%.4f trust=%.3f  ->  "
                    "RETRAIN(jadi pi_%d): gini_util=%.4f trust=%.3f",
                    r, r, rec["freeze_gini_util"] or -1, rec["freeze_trust"] or -1,
                    r + 1, rec["retrain_gini_util"] or -1, rec["retrain_trust"] or -1)
        return self.policy, rrm_trace
