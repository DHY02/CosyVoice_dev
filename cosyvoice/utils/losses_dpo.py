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

