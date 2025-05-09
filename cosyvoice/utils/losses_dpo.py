import logging
import torch
import torch.nn.functional as F
from typing import Tuple


def tpr_loss(disc_real_outputs, disc_generated_outputs, tau):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        m_DG = torch.median((dr - dg))
        L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss


def mel_loss(real_speech, generated_speech, mel_transforms):
    loss = 0
    for transform in mel_transforms:
        mel_r = transform(real_speech)
        mel_g = transform(generated_speech)
        loss += F.l1_loss(mel_g, mel_r)
    return loss


class DPOLoss(torch.nn.Module):
    """
    DPO Loss
    """

    def __init__(
        self, 
        beta: float, 
        label_smoothing: float = 0.0, 
        ipo: bool = False, 
        use_emo_dpo = False,
        emo_dpo_epoch: int = 6
    ) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo
        self.use_emp_dpo = use_emo_dpo
        self.emo_dpo_epoch = emo_dpo_epoch

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        jsd = self.get_jsd(policy_chosen_logps - reference_chosen_logps, policy_rejected_logps - reference_rejected_logps)
        logits -= jsd
        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards

    def get_jsd(self, cho_ratio, rej_ratio):
        return torch.log1p(torch.exp(cho_ratio)) - torch.log1p(torch.exp(rej_ratio))


class GRPOLoss(torch.nn.Module):
    """
    GRPO Loss
    """

    def __init__(
        self, 
        beta: float = 0.04, 
        grpo_clip: float = 0.2,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.grpo_clip = grpo_clip

    def forward(
        self,
        reference_logps: torch.Tensor,
        active_logps: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
            reference_logps: shape (B, G, L)
            active_logps: shape (B, G, L)
            advantages: shape (B, G),
            mask: shape (B, G, L)
        """
        # 计算优势函数，shape: (B, G)
        group_mean_advantages = advantages.mean(dim=1, keepdim=True)
        group_std_advantages = advantages.std(dim=1, keepdim=True)
        advantages = (advantages - group_mean_advantages) / (group_std_advantages + 1e-4)
        # shape:(B, G, 1)
        advantages = advantages.unsqueeze(2)

        # 计算KL散度
        per_token_kl = self.grpo_kl(reference_logps, active_logps)

        # 计算ratio
        coef_1 = torch.exp(active_logps - reference_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.grpo_clip, 1 + self.grpo_clip)

        # 被clip样本的比例，越高代表参考策略和当前策略差异越大，或优势函数估计不稳定，或超参数 grpo_clip 设置不合理
        is_low_clipped = (coef_1 < 1 - self.grpo_clip) & (advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.grpo_clip) & (advantages > 0)
        clip_ratio = (is_low_clipped | is_high_clipped).float().mean()
        mean_kl = per_token_kl.mean()

        # 计算总损失
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) + self.beta * per_token_kl
        loss =  ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        metrics = {
            "kl": mean_kl.detach(),
            "clip_ratio": clip_ratio.detach(),
            "clip_ratio/low": is_low_clipped.float().mean().detach(),
            "clip_ratio/high": is_high_clipped.float().mean().detach(),
        }
        return loss, metrics
        
    def grpo_kl(
        self,         
        reference_logps: torch.Tensor,
        active_logps: torch.Tensor,
    ):
        return torch.exp(reference_logps - active_logps) - (reference_logps - active_logps) - 1
