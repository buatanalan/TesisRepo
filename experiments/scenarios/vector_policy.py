import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""VectorWaitPolicy -- salinan HPPOPolicy (marl_spklu/rl/policy.py) TANPA kepala diskrit (a1).
Kebijakan memprediksi VEKTOR modifikasi wait untuk SEMUA N stasiun sekaligus (bukan cuma
stasiun yang "terpilih" lewat argmax/sample kategorikal seperti arsitektur lama). "Rekomendasi"
jadi implisit -- stasiun ber-estimasi-wait terendah otomatis paling menarik lewat p_rec di
decide_spklu (lihat vector_rollout_agent.py & rencana di C:/Users/Lenovo/.claude/plans/
proud-twirling-scone.md).

TIDAK mengedit marl_spklu/rl/policy.py sama sekali -- file ini murni baru, konsisten dgn pola
"nol edit kode inti" yang dipegang sepanjang investigasi.
"""
import warnings

import numpy as np
import torch
import torch.nn as nn

from marl_spklu.rl.ppo import compute_gae   # reuse murni, tak diedit

DELTA_MAX = 10.0


class VectorWaitPolicy(nn.Module):
    """Encoder & critic identik HPPOPolicy; kepala aksi HANYA kontinu (mean per-stasiun),
    tanpa kepala diskrit. act()/evaluate() menyampel & mengevaluasi SELURUH vektor N-dim
    sekaligus (dimask ke kandidat feasible), bukan cuma satu elemen terpilih."""

    def __init__(self, obs_dim: int, critic_obs_dim: int, n_spklu: int, hidden: int = 128,
                delta_max: float = DELTA_MAX):
        super().__init__()
        self.n_spklu = n_spklu
        self.delta_max = float(delta_max)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.cont_mean = nn.Linear(hidden, n_spklu)
        self.cont_logstd = nn.Parameter(torch.zeros(n_spklu))

        self.critic_encoder = nn.Sequential(
            nn.Linear(critic_obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.critic_head = nn.Linear(128, 1)

    def forward(self, obs, critic_obs=None):
        h = self.encoder(obs)
        mean = torch.tanh(self.cont_mean(h)) * self.delta_max
        if critic_obs is not None:
            h_c = self.critic_encoder(critic_obs)
            value = self.critic_head(h_c).squeeze(-1)
        else:
            value = (torch.zeros(obs.shape[0], device=obs.device) if len(obs.shape) > 1
                    else torch.zeros((), device=obs.device))
        return mean, value

    @torch.no_grad()
    def act(self, obs_np, feasible_mask_np, critic_obs_np=None):
        """Sampel VEKTOR delta utk semua stasiun feasible sekaligus. logp dijumlah hanya
        atas entri feasible (konsisten dgn evaluate())."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(feasible_mask_np, dtype=torch.bool).unsqueeze(0)
        critic_obs = (torch.as_tensor(critic_obs_np, dtype=torch.float32).unsqueeze(0)
                     if critic_obs_np is not None else None)

        mean, value = self.forward(obs, critic_obs)
        if not torch.isfinite(mean).all():
            warnings.warn("VectorWaitPolicy.act: mean non-finite -> kemungkinan gradien "
                          "meledak; turunkan lr atau cek observasi.", RuntimeWarning)
            mean = torch.nan_to_num(mean, nan=0.0, posinf=self.delta_max, neginf=-self.delta_max)

        std = torch.exp(self.cont_logstd).clamp(1e-3, self.delta_max)
        dist = torch.distributions.Normal(mean[0], std)
        sample = dist.sample()
        logp_full = dist.log_prob(sample)
        mask_f = mask[0].float()
        logp = (logp_full * mask_f).sum()
        delta_vec = sample.clamp(-self.delta_max, self.delta_max).numpy()

        return {"delta_vec": delta_vec, "logp": float(logp.item()), "value": float(value.item())}

    def evaluate(self, obs_b, mask_b, delta_b, critic_obs_b=None):
        """Batch. obs_b:(B,obs), mask_b:(B,N) bool, delta_b:(B,N) float -> logp,entropy,value
        (logp/entropy dijumlah/dirata-rata HANYA atas entri feasible per baris)."""
        mean, value = self.forward(obs_b, critic_obs_b)
        std = torch.exp(self.cont_logstd).clamp(1e-3, self.delta_max)
        std_b = std.unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std_b)

        logp_full = dist.log_prob(delta_b)
        ent_full = dist.entropy()
        mask_f = mask_b.float()
        n_feas = mask_f.sum(dim=1).clamp(min=1.0)

        logp = (logp_full * mask_f).sum(dim=1)
        entropy = (ent_full * mask_f).sum(dim=1) / n_feas
        return logp, entropy, value


class VectorPPOTrainer:
    """Salinan ringkas PPOTrainer (marl_spklu/rl/ppo.py) diadaptasi utk VectorWaitPolicy.
    GAE (`compute_gae`) dipakai ULANG langsung dari ppo.py -- tak ada duplikasi logika itu."""

    def __init__(self, policy: VectorWaitPolicy, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                epochs=10, minibatch=64, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                target_kl=0.03):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
        self.target_kl = target_kl

    def update(self, transitions):
        if len(transitions) < 2:
            return {"loss": 0.0, "n": len(transitions)}
        returns, adv = compute_gae(transitions, self.gamma, self.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_b = torch.as_tensor(np.stack([t.obs for t in transitions]), dtype=torch.float32)
        critic_obs_b = torch.as_tensor(np.stack([t.critic_obs for t in transitions]), dtype=torch.float32)
        mask_b = torch.as_tensor(np.stack([t.mask for t in transitions]), dtype=torch.bool)
        delta_b = torch.as_tensor(np.stack([t.delta for t in transitions]), dtype=torch.float32)
        old_logp = torch.as_tensor(np.array([t.logp for t in transitions]), dtype=torch.float32)
        ret_b = torch.as_tensor(returns, dtype=torch.float32)
        adv_b = torch.as_tensor(adv, dtype=torch.float32)

        B = len(transitions)
        idx = np.arange(B)
        last = {}
        grad_norm = 0.0
        n_skipped = 0
        epochs_ran = 0
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, B, self.minibatch):
                mb = idx[start:start + self.minibatch]
                logp, ent, value = self.policy.evaluate(
                    obs_b[mb], mask_b[mb], delta_b[mb], critic_obs_b=critic_obs_b[mb])
                ratio = torch.exp(logp - old_logp[mb])
                s1 = ratio * adv_b[mb]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_b[mb]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = nn.functional.mse_loss(value, ret_b[mb])
                ent_loss = -ent.mean()
                loss = pi_loss + self.vf_coef * v_loss + self.ent_coef * ent_loss
                if not torch.isfinite(loss):
                    n_skipped += 1
                    continue
                self.opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                grad_norm = float(gn)
                self.opt.step()
                last = {"pi_loss": pi_loss.item(), "v_loss": v_loss.item(),
                        "entropy": ent.mean().item(), "loss": loss.item()}
            epochs_ran += 1
            if self.target_kl is not None:
                with torch.no_grad():
                    nlp, _, _ = self.policy.evaluate(obs_b, mask_b, delta_b, critic_obs_b=critic_obs_b)
                    kl = float((old_logp - nlp).mean())
                if kl > 1.5 * self.target_kl:
                    break

        with torch.no_grad():
            new_logp, new_ent, new_val = self.policy.evaluate(obs_b, mask_b, delta_b, critic_obs_b=critic_obs_b)
            ratio = torch.exp(new_logp - old_logp)
            approx_kl = float((old_logp - new_logp).mean())
            clip_frac = float((torch.abs(ratio - 1.0) > self.clip).float().mean())
            var_ret = float(ret_b.var())
            ev = float(1.0 - (ret_b - new_val).var() / var_ret) if var_ret > 1e-8 else 0.0
        last.update({
            "approx_kl": approx_kl, "clip_frac": clip_frac, "explained_var": ev,
            "grad_norm": grad_norm, "entropy_final": float(new_ent.mean()),
            "adv_mean": float(adv.mean()), "adv_std": float(adv.std()),
            "n_skipped": n_skipped, "epochs_ran": epochs_ran,
        })
        return last
