"""Trainer PDQN mengikuti Algorithm 1 Lin et al. 2024 (IEEE TCE 70(3):5238-5250):
main network + TARGET network, experience replay, target TD

    y = r + eta * max_a' Q_target(s', p'; a')                         (eq. 11)
    L = E[(Q(s,p;a) - y)^2]                                            (eq. 12)

PEMASANGAN (s -> s'): MDP paper didefinisikan ATAS PERMINTAAN PENGISIAN (event-driven),
bukan atas langkah waktu simulasi -- "for each charging request do ... Determine the
next state s'" (Algorithm 1). Jadi s' = state sistem yang dihadapi PERMINTAAN BERIKUTNYA.
Ini sah meski pengguna berikutnya orang lain: komponen state yang di-bootstrap adalah
kondisi JARINGAN SPKLU (antrean/utilisasi/EV dalam perjalanan) yang memang dipengaruhi
aksi sekarang; preferensi pengguna berikutnya p' oleh paper dinyatakan independen dari
(s,a) (eq. 7: "the preference of the next user which is independent of (s,a)").
Karena itu transisi dipasangkan menurut URUTAN KEPUTUSAN (transitions[i] -> [i+1]).

Suku reward wait bersifat DELAYED (baru diketahui saat sesi charging selesai), sehingga
sebuah transisi baru boleh masuk buffer setelah `resolved=True` DAN penerusnya sudah ada.
"""
import json
import logging
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.pdqn_policy import PDQNQNetwork
from marl_spklu.rl.pdqn_agent import PDQNRolloutAgent
from marl_spklu.rl.rewards import RewardCalculator, _gini

NEG_INF = -1e9


def _make_logger(verbose: bool):
    logger = logging.getLogger("marl_spklu.rl.dqn")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger


class ReplayBuffer:
    """Experience replay (paper §IV.B). Menyimpan (s, a, r, s', mask', done)."""

    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, hist, ptype, action, reward, next_obs, next_hist, next_ptype,
             next_mask, done):
        self.buf.append((obs, hist, ptype, action, reward, next_obs, next_hist, next_ptype,
                         next_mask, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        (obs, hist, ptype, action, reward, next_obs, next_hist, next_ptype,
         next_mask, done) = zip(*batch)
        return (np.stack(obs), np.stack(hist), np.array(ptype, dtype=np.int64),
                np.array(action, dtype=np.int64), np.array(reward, dtype=np.float32),
                np.stack(next_obs), np.stack(next_hist), np.array(next_ptype, dtype=np.int64),
                np.stack(next_mask), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DQNContinuingTrainer:
    """Melatih PDQN pada satu Simulator PERSISTEN yang di-step per potongan (chunk),
    mengikuti pola TorchContinuingTrainer (ppo.py) agar dinamika lintas-potongan tidak
    ter-reset. Hyperparameter default = spesifikasi_teknis_pdqn_baseline.md §3.4
    (lr 1e-3, batch 64, gamma 0.95, target sync tiap 100 langkah gradien,
    epsilon 1.0 -> 0.01 linear)."""

    def __init__(self, dataset_path, rollout_steps=384, gamma=0.95, lr=1e-3, batch_size=64,
                 buffer_capacity=100_000, target_update_every=100, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay_frac=0.6, reward_calc=None,
                 seed=0, verbose=True, log_path=None, willingness_ratio=None,
                 willingness_radius_km=None, horizon=None, n_types=None,
                 use_preference: bool = True, pref_feature_mode: bool = False,
                 network_cls=None):
        self.dataset_path = dataset_path
        self.rollout_steps = rollout_steps
        # Panjang satu PASS simulasi sebelum sim di-reset. None -> pakai sim.max_steps
        # (horizon penuh dataset, 2880 = 30 hari). Horizon pendek (mis. 288 = 3 hari)
        # membuat evaluasi jauh lebih murah DAN memisahkan baseline lebih tajam.
        self.horizon = horizon
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_every = int(target_update_every)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        # Peluruhan epsilon dipatok pada PROPORSI ANGGARAN POTONGAN (chunk), bukan jumlah
        # langkah gradien absolut: jumlah pasangan (s,a,r,s') per potongan bervariasi
        # (tergantung berapa sesi charging selesai), sehingga jadwal berbasis langkah
        # gradien membuat epsilon akhir bergantung anggaran -- mis. 100 potongan hanya
        # menghasilkan ~15rb langkah gradien, jadi jadwal 100rb akan berakhir di eps~0.85
        # (agen masih 85% acak saat training usai). Dgn frac=0.6, epsilon mencapai lantai
        # pada 60% potongan pertama untuk anggaran BERAPA PUN.
        self.epsilon_decay_frac = float(epsilon_decay_frac)
        self._total_chunks = None   # diisi di train()
        self.rc = reward_calc or RewardCalculator()
        self.verbose = verbose
        self.log_path = log_path
        self.willingness_ratio = willingness_ratio
        self.willingness_radius_km = willingness_radius_km
        self.seed = seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sim = self._fresh_sim()
        self.N = len(sim.spklus)
        # obs = 6 skalar + 7 blok stasiun (onehot_prev,dist,wait_hat,queue,conn,util,rec_activity)
        self.obs_dim = 6 + 7 * self.N

        # Jumlah LSTM preference-extractor. None -> DETEKSI OTOMATIS dari dataset:
        # kalau pengguna punya `pref_type`, pakai satu LSTM per tipe (desain paper);
        # kalau tidak, satu LSTM bersama. Set eksplisit n_types=1 pd dataset bertipe
        # untuk menjalankan ABLASI "PDQN 1-LSTM" (mengisolasi kontribusi LSTM per-tipe).
        if n_types is None:
            # `pref_type` (5 tipe preferensi diskrit, ablasi Lin et al. 2024) sudah
            # dihapus dari User -- dataset baru (Klaster 31, LCMNL) tak menyediakannya,
            # getattr default None -> selalu 1 LSTM bersama.
            types = {getattr(u, "pref_type", None) for u in sim.users}
            types.discard(None)
            n_types = (max(types) + 1) if types else 1
        self.n_types = int(n_types)

        self.use_preference = bool(use_preference)
        self.pref_feature_mode = bool(pref_feature_mode)
        # network_cls: None -> PDQNQNetwork asli (default). Bisa dioverride mis.
        # PDQNAdvancedQNetwork (pdqn_advanced_policy.py, shared-weight+dueling head) --
        # signature constructor SENGAJA identik supaya drop-in, lihat docstring modul itu.
        net_cls = network_cls or PDQNQNetwork
        self.q_net = net_cls(self.obs_dim, self.N, n_types=self.n_types,
                             use_preference=self.use_preference,
                             pref_feature_mode=self.pref_feature_mode)
        self.target_net = net_cls(self.obs_dim, self.N, n_types=self.n_types,
                                  use_preference=self.use_preference,
                                  pref_feature_mode=self.pref_feature_mode)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.history = []
        self._logger = _make_logger(verbose)
        self._n_updates = 0   # langkah GRADIEN (target sync & epsilon decay)
        self._it_global = 0
        self.config = {"dataset": dataset_path, "N": self.N, "obs_dim": self.obs_dim,
                       "rollout_steps": rollout_steps, "seed": seed, "gamma": gamma, "lr": lr,
                       "batch_size": batch_size, "target_update_every": target_update_every,
                       "epsilon_start": epsilon_start, "epsilon_end": epsilon_end,
                       "epsilon_decay_frac": epsilon_decay_frac, "n_types": self.n_types,
                       "use_preference": self.use_preference,
                       "n_params": sum(p.numel() for p in self.q_net.parameters())}
        self._logger.info("CONFIG %s", self.config)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"event": "config", **self.config}) + "\n")

    def _fresh_sim(self):
        from marl_spklu.rl.training import _fresh_sim
        return _fresh_sim(self.dataset_path, willingness_ratio=self.willingness_ratio,
                          willingness_radius_km=self.willingness_radius_km)

    def _epsilon(self):
        """Linear 1.0 -> 0.01 atas `epsilon_decay_frac` bagian pertama dari anggaran
        potongan; setelah itu bertahan di lantai."""
        horizon = max(1.0, self.epsilon_decay_frac * (self._total_chunks or 1))
        frac = min(1.0, self._it_global / horizon)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # ---------------- Replay: pasangan (s,a,r,s') menurut urutan keputusan ----------------
    def _push_ready_pairs(self, agent, boundary: bool = False) -> int:
        """Masukkan ke buffer transisi yang SUDAH resolved dan penerusnya sudah ada.
        Lalu buang dari depan transisi yang sudah masuk buffer (tak lagi dibutuhkan).

        v3 (2026-08-13): kembali SATU transisi per keputusan (`action_idx`=pasangan
        (i,j), Q_tot=Q(i)+Q(j) -- lihat pdqn_agent.py). Penerus transisi ke-i = i+1
        (urutan keputusan biasa, sama seperti v1/paper asli)."""
        trs = agent.transitions
        n_new = 0
        for i in range(len(trs) - 1):
            t = trs[i]
            if t.resolved and not t.pushed:
                nx = trs[i + 1]
                self.buffer.push(t.obs, t.hist, t.pref_type, t.action_idx, t.reward,
                                 nx.obs, nx.hist, nx.pref_type, nx.mask,
                                 1.0 if t.done else 0.0)
                t.pushed = True
                n_new += 1
        if boundary and trs:
            # Akhir horizon: tak ada permintaan berikutnya -> terminal, bootstrap dinolkan.
            last = trs[-1]
            if last.resolved and not last.pushed:
                self.buffer.push(last.obs, last.hist, last.pref_type, last.action_idx,
                                 last.reward, last.obs, last.hist, last.pref_type,
                                 last.mask, 1.0)
                last.pushed = True
                n_new += 1
        k = 0
        while k < len(trs) and trs[k].pushed:
            k += 1
        agent.transitions = trs[k:]
        return n_new

    def _dqn_update(self):
        """Satu langkah gradien: y = r + gamma*(1-done)*max_{a,b} [Q_target(s',a)+
        Q_target(s',b)] (v3, dekomposisi-nilai/VDN -- lihat pdqn_agent.py docstring),
        loss MSE atas Q_tot(s)=Q(s,i)+Q(s,j). Aksi tak-feasible di s' di-mask sebelum max.

        max atas JUMLAH dua Q independen (tanpa suku interaksi) dicapai dgn memilih
        top-2 nilai Q individual tertinggi lalu dijumlah -- optimal krn tak ada
        pasangan yg secara eksplisit "lebih baik dari jumlah bagian"nya (beda dari
        v1 15-kombinasi yg punya Q gabungan LANGSUNG per pasangan)."""
        if len(self.buffer) < self.batch_size:
            return None
        (obs, hist, ptype, action, reward, next_obs, next_hist, next_ptype,
         next_mask, done) = self.buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        hist_t = torch.as_tensor(hist, dtype=torch.float32)
        # Tipe preferensi memilih LSTM mana yang dipakai tiap sampel (paper: satu LSTM
        # per tipe). Tipe s' ikut disimpan karena pengguna BERIKUTNYA bisa beda tipe.
        ptype_t = torch.as_tensor(ptype, dtype=torch.long)
        next_ptype_t = torch.as_tensor(next_ptype, dtype=torch.long)
        action_t = torch.as_tensor(action, dtype=torch.long)   # (B,2) pasangan (i,j)
        reward_t = torch.as_tensor(reward, dtype=torch.float32)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
        next_hist_t = torch.as_tensor(next_hist, dtype=torch.float32)
        next_mask_t = torch.as_tensor(next_mask, dtype=torch.bool)   # (B,N) level-stasiun
        done_t = torch.as_tensor(done, dtype=torch.float32)

        use_mixer = hasattr(self.q_net, "mixer")   # PDQNQMixQNetwork -- lihat pdqn_advanced_policy.py

        if use_mixer:
            # QMIX: Q_tot = mixer(Q(s,i),Q(s,j); h) NON-LINIER MONOTONIK, ganti
            # penjumlahan VDN tetap. IGM tetap berlaku -> `act()`/top-2 individual tak
            # berubah, HANYA target TD di sini yg beda (§2.1 Kajian_Literatur...).
            q_all, h = self.q_net.forward_with_ctx(obs_t, hist_t, ptype_t)   # (B,N),(B,width)
            qs = torch.stack([q_all.gather(1, action_t[:, 0:1]).squeeze(1),
                              q_all.gather(1, action_t[:, 1:2]).squeeze(1)], dim=1)   # (B,2)
            q_sa = self.q_net.mixer(qs, h)
        else:
            q_all = self.q_net(obs_t, hist_t, ptype_t)   # (B,N)
            q_sa = (q_all.gather(1, action_t[:, 0:1]).squeeze(1) +
                   q_all.gather(1, action_t[:, 1:2]).squeeze(1))   # Q_tot(s) = Q(s,i)+Q(s,j)

        with torch.no_grad():
            if use_mixer:
                q_next, h_next = self.target_net.forward_with_ctx(
                    next_obs_t, next_hist_t, next_ptype_t)
            else:
                q_next = self.target_net(next_obs_t, next_hist_t, next_ptype_t)   # (B,N)
            q_next = q_next.masked_fill(~next_mask_t, NEG_INF)
            n_feasible = next_mask_t.sum(dim=1)   # (B,)
            k_top = min(2, q_next.shape[1])
            top2 = torch.topk(q_next, k=k_top, dim=1).values   # (B, k_top)
            val0 = top2[:, 0]
            if k_top > 1:
                val1 = torch.where(n_feasible >= 2, top2[:, 1], val0)
            else:
                val1 = val0
            if use_mixer:
                # IGM: max joint Q_tot dicapai dgn menggabungkan argmax INDIVIDUAL tiap
                # slot (val0,val1 dari top-2 Q_target), lalu dilewatkan mixer target
                # (mixer monotonik naik thd tiap komponen -> max per-komponen = max joint).
                qs_next = torch.stack([val0, val1], dim=1)
                q_next_max = self.target_net.mixer(qs_next, h_next)
            else:
                q_next_max = val0 + val1
            # Baris tanpa aksi feasible sama sekali -> bootstrap 0 (bukan -1e9).
            q_next_max = torch.where(n_feasible > 0, q_next_max, torch.zeros_like(q_next_max))
            target = reward_t + self.gamma * (1.0 - done_t) * q_next_max

        loss = nn.functional.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.opt.step()

        self._n_updates += 1
        if self._n_updates % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": float(loss.item()), "grad_norm": float(grad_norm),
                "q_mean": float(q_sa.detach().mean())}

    def train(self, n_updates: int):
        """n_updates = jumlah POTONGAN simulasi (bukan langkah gradien)."""
        chunk = self.rollout_steps
        self._total_chunks = int(n_updates)   # patokan jadwal epsilon (lihat _epsilon)
        sim = self._fresh_sim()
        horizon = min(self.horizon or sim.max_steps, sim.max_steps)
        agent = PDQNRolloutAgent(self.q_net, sim, self.rc, epsilon=self._epsilon(),
                                 pref_feature_mode=self.pref_feature_mode)
        step = 0
        for _ in range(n_updates):
            it = self._it_global
            agent.epsilon = self._epsilon()
            boundary = False
            for _ in range(chunk):
                sim.step_once(step, agent=agent)
                step += 1
                if step >= horizon:
                    boundary = True
                    break
            if boundary:
                # Horizon habis: user in-flight tak akan menyelesaikan sesi di sim ini.
                for t in agent.transitions:
                    t.resolved = True
                if agent.transitions:
                    agent.transitions[-1].done = True

            n_new = self._push_ready_pairs(agent, boundary=boundary)
            stats = None
            for _ in range(n_new):
                s = self._dqn_update()
                if s is not None:
                    stats = s

            self._it_global += 1
            if n_new:
                served = np.array([s.total_served for s in sim.spklus.values()], float)
                utils = np.array([s.get_utilization() for s in sim.spklus.values()])
                info = {"n_new": n_new, "n_backlog": len(agent.transitions),
                        "epsilon": agent.epsilon, "buffer_size": len(self.buffer),
                        "gini_served": _gini(served), "gini_util": _gini(utils)}
                rec = {"iter": it, **info, **(stats or {})}
                self.history.append(rec)
                if self.log_path:
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps({"event": "iter", **rec}) + "\n")
                if self.verbose:
                    self._logger.info(
                        "[chunk %3d] new=%d backlog=%d eps=%.3f | loss=%.4f q=%+.3f | "
                        "gini_served=%.3f buf=%d%s",
                        it, n_new, info["n_backlog"], info["epsilon"],
                        (stats or {}).get("loss", 0.0), (stats or {}).get("q_mean", 0.0),
                        info["gini_served"], info["buffer_size"],
                        " |PASS-BARU" if boundary else "")

            if boundary:
                sim = self._fresh_sim()
                agent = PDQNRolloutAgent(self.q_net, sim, self.rc, epsilon=self._epsilon(),
                                 pref_feature_mode=self.pref_feature_mode)
                step = 0
        return self.q_net
