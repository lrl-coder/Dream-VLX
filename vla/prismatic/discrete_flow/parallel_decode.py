import torch
import torch.nn.functional as F
import math

from .mask_schedule import schedule as mask_schedule


def mask_by_random_topk(probs: torch.Tensor,
                        mask_len: torch.Tensor,
                        temperature: float = 1.0) -> torch.BoolTensor:
    """
    PyTorch version of mask_by_random_topk:
    probs: [B, L], probability of each position being sampled
    mask_len: [B], number of positions to mask for each sample
    Returns boolean mask matrix [B, L]
    """
    # 1) Sample Gumbel noise
    gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-20) + 1e-20)
    # 2) Compute confidence scores
    confidence = torch.log(probs + 1e-20) + temperature * gumbel  # [B, L]
    # 3) Find the k-th smallest threshold for each row
    sorted_conf, _ = confidence.sort(dim=1)  # [B, L]
    B, L = probs.shape
    k = mask_len.clamp(min=1, max=L-1)      # [B]
    batch_idx = torch.arange(B, device=probs.device)
    threshold = sorted_conf[batch_idx, k]  # [B]

    # 4) Set positions below threshold to True (continue to be masked)
    return confidence < threshold.unsqueeze(1)  # [B, L]


def top_k_logits(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.size(-1))
    val, idx = logits.topk(k, dim=-1)
    mask = torch.full_like(logits, float('-inf')).to(device=logits.device)
    mask.scatter_(-1, idx, val)
    return mask


def decode(
    init_ids: torch.LongTensor,             # [B, L], initial sequence (containing mask_token_id)
    tokens_to_logits,                       # fn(seq_ids: [B, L]) -> logits [B, L, V]
    mask_token_id: int,
    start_iter: int = 0,
    num_iter: int = 12,
    choice_temperature: float = 1.0,
    mask_scheduling_method="cosine",
    use_remask: bool = False,               # Whether to use remask probability
    token_critic: torch.nn.Module = None,  # TokenCritic model, outputs [B, L] scores
    critic_noise_scale: float = 1.0,       # Critic noise scale
):
    """
    Non-autoregressive MaskGIT inference
    Returns final_seqs: [B, num_iter, L], sampled sequences for each iteration
    """
    B, L = init_ids.shape
    device = init_ids.device

    # Record initial unknown (mask) count for scheduling
    unknown_init = (init_ids == mask_token_id).sum(dim=1)  # [B]

    # State init
    cur_seqs = init_ids.clone()                         # [B, L]
    final_seqs = torch.zeros(B, num_iter, L, dtype=init_ids.dtype, device=device)
    final_seqs[:, 0, :] = init_ids

    # Iterative decoding
    for step in range(start_iter, num_iter):
        # 1) Get logits & probability distribution
        logits, actions_hidden_states = tokens_to_logits(cur_seqs)             # [B, L, V]
        probs = F.softmax(logits, dim=-1)               # [B, L, V]

        # 2) Parallel categorical sampling
        #    Flatten, sample, then reshape
        flat_probs = probs.view(-1, probs.size(-1))     # [B*L, V]
        sampled_flat = torch.multinomial(flat_probs, 1)  # [B*L, 1]
        sampled = sampled_flat.view(B, L)               # [B, L]

        # 3) Update only at mask positions
        unknown_map = cur_seqs == mask_token_id         # [B, L]
        sampled = torch.where(unknown_map, sampled, cur_seqs)

        # 4) Compute number of masks for next iteration
        ratio = torch.tensor(float(step + 1) / num_iter, device=device)              # scalar
        # Scheduling function: given ratio and initial unknown count, returns mask_ratio
        mask_ratio = mask_schedule(ratio, unknown_init, mask_scheduling_method)  # [B]
        mask_len = torch.floor(unknown_init.float() * mask_ratio).long()
        # Ensure at least 1 and at most unknown_init-1
        mask_len = torch.clamp(mask_len, min=1, max=(unknown_init - 1).item())

        # —————————————— 4) Compute scores for each position ——————————————
        if token_critic is not None:
            # Use Critic scores + add noise
            # Assume token_critic returns [B, L] raw scores
            raw_crit = token_critic(actions_hidden_states)              # [B, L]
            scores = - raw_crit
            # Add uniform noise, noise decreases as step increases
            scores = scores + (torch.rand_like(scores).cuda() - 0.5) * critic_noise_scale * (1.0 - ratio)
            selected_probs = scores
        else:
            # 5) Compute probability of each position being selected: probs.gather
            selected_probs = probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1)  # [B, L]

        if use_remask:
            # 6) Introduce "remask probability"
            #    p_remask linearly decreases from 1 to 0: easier to remask early, more stable later
            p_remask = 1.0 - ratio
            #    Reduce confidence for known positions (~unknown_map)
            selected_probs = torch.where(
                unknown_map,
                selected_probs,
                selected_probs * p_remask
            )  # [B, L]
        else:
            # Set known positions (initially non-mask) to infinity to avoid being masked in next iteration
            inf = torch.tensor(float("inf"), device=device)
            selected_probs = torch.where(unknown_map, selected_probs, inf)

        # 6) Use Gumbel+top-k strategy to decide positions to be masked in next iteration
        masking = mask_by_random_topk(
            selected_probs,
            mask_len,
            temperature=choice_temperature * (1.0 - ratio),
        )                                               # [B, L]

        # 7) Construct next seqs: positions to be masked continue using mask_token
        next_seqs = torch.where(masking, mask_token_id, sampled)  # [B, L]
        cur_seqs = next_seqs

        # 8) Store results of this iteration
        final_seqs[:, step, :] = sampled

        # TODO: Important: select last iteration final_iters[:, -1, :] as final output, or fuse multi-iteration results

    return final_seqs, actions_hidden_states

