import torch
import torch.nn.functional as F
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def joint_threshold_update_step_vectorized(
    input_ids_1d: torch.Tensor,  # [B*blk]
    full_logits_2d: torch.Tensor,  # [B*blk, V]
    prompt_masks: torch.Tensor,  # [B, blk]
    finished: torch.Tensor,  # [B]
    post_edit_steps: torch.Tensor,  # [B]
    mask_id: int,
    blk: int,
    threshold: float,
    edit_threshold: float,
    max_post_edit_steps: int,
    penalty_lambda: float,
):
    """Batched single denoise step for joint-threshold decoding.

    Advances ``input_ids_1d`` / ``finished`` / ``post_edit_steps`` in place,
    processing every block at once (no per-row Python loop or ``.item()`` sync).
    Finished rows preserve their tokens and state.
    """
    B = input_ids_1d.shape[0] // blk
    V = full_logits_2d.shape[1]

    input_ids = input_ids_1d.view(B, blk)
    logits = full_logits_2d.view(B, blk, V)

    active = ~finished

    # ---------- penalty ----------
    if penalty_lambda > 0:
        prev_ids = input_ids[:, :-1]
        logits[:, 1:, :].scatter_(
            dim=2,
            index=prev_ids.unsqueeze(-1),
            src=torch.full_like(
                prev_ids.unsqueeze(-1), -penalty_lambda, dtype=logits.dtype
            ),
            reduce="add",
        )

    # ---------- argmax + confidence ----------
    # Same ops as the per-row path (argmax over logits, then gather the softmax
    # probability), just batched: keeps decisions bitwise-aligned with it. On
    # NPU this also beats a log-domain max+logsumexp variant (fused softmax).
    x = torch.argmax(logits, dim=-1)
    p = torch.gather(F.softmax(logits, dim=-1), dim=-1, index=x.unsqueeze(-1)).squeeze(
        -1
    )

    mask_pos = input_ids.eq(mask_id)
    has_mask = mask_pos.any(dim=1)

    # ---------- post-edit ----------
    no_mask_active = active & (~has_mask)
    post_edit_steps.add_(no_mask_active.to(post_edit_steps.dtype))
    exceeded = post_edit_steps > max_post_edit_steps
    finished |= no_mask_active & exceeded

    # eligible rows (match original semantics)
    eligible = active & (~(no_mask_active & exceeded))

    # ---------- M2T ----------
    neg_inf = torch.full_like(p, float("-inf"))
    conf_m2t = torch.where(mask_pos, p, neg_inf)

    m2t = (conf_m2t > threshold) & (eligible & has_mask).view(B, 1)

    # force-one if needed
    hit_any = m2t.any(dim=1)
    need_force = (eligible & has_mask) & (~hit_any)

    # topk (not argmax): the per-row fallback picks its forced position with
    # torch.topk, and the two ops can break exact-confidence ties differently.
    best_idx = torch.topk(conf_m2t, k=1, dim=1).indices.squeeze(1)
    rows = torch.arange(B, device=input_ids.device)

    m2t[rows, best_idx] |= need_force

    # ---------- T2T ----------
    edit_mask = (~mask_pos) & (~prompt_masks)
    t2t = (p > edit_threshold) & (input_ids != x) & edit_mask
    t2t = t2t & eligible.view(B, 1)

    # ---------- combine ----------
    transfer = m2t | t2t
    any_transfer_row = transfer.any(dim=1)

    finished |= eligible & (~any_transfer_row)

    # apply update
    input_ids.copy_(torch.where(transfer, x, input_ids))

    return any_transfer_row.any()


class JointThreshold(DllmAlgorithm):
    """Joint-threshold denoising: mask-to-token (M2T) unmasking plus token-to-token
    (T2T) edits, finishing on no-change or an exhausted edit budget. Stateful (edit
    budget + prompt mask), carried across FDFO rounds via ``dllm_algo_state``.
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.penalty_lambda = config.algorithm_config.get("penalty_lambda", 0)
        # All platforms use the tensor path; legacy vectorized_decoding
        # configuration no longer selects a host-synchronizing per-row path.

    def max_steps(self, block_size: int) -> int:
        return block_size + self.max_post_edit_steps + 1

    def init_step_state(self, forward_batch: ForwardBatch) -> dict[str, torch.Tensor]:
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        prompt_masks = input_ids != self.mask_id
        return {
            "prompt_masks": prompt_masks,
            # Fully populated fresh rows are prompt-only blocks, not candidates
            # for post-editing. Continuing rows retain their original masks.
            "finished": prompt_masks.all(dim=1),
            "post_edit_steps": torch.zeros(
                batch_size, dtype=torch.int32, device=input_ids.device
            ),
        }

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        joint_threshold_update_step_vectorized(
            input_ids_1d=forward_batch.input_ids,
            full_logits_2d=full_logits,
            prompt_masks=states["prompt_masks"],
            finished=states["finished"],
            post_edit_steps=states["post_edit_steps"],
            mask_id=self.mask_id,
            blk=self.block_size,
            threshold=self.threshold,
            edit_threshold=self.edit_threshold,
            max_post_edit_steps=self.max_post_edit_steps,
            penalty_lambda=self.penalty_lambda,
        )
        # A terminating step changes no tokens, so its forward already wrote
        # final KV. A later redundant step remains a terminal no-op.
        return states["finished"]


Algorithm = JointThreshold
