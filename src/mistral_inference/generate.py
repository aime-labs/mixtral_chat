from typing import List, Optional, Tuple

import torch

from .cache import BufferCache
from .model import Transformer
from .tokenizer import ChatFormat


@torch.inference_mode()
def generate(
    prompts,
    model,
    tokenizer,
    callback,
    max_tokens: int,
    max_seq_length: int,
    temperatures: float,
    top_ps: float,
    top_ks: float,
    chunk_size: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    model = model.eval()
    formatter = ChatFormat(tokenizer)
    encoded_prompts = [formatter.encode_dialog_prompt(prompt) for prompt in prompts]
    batch_size, vocab_size = len(encoded_prompts), model.args.vocab_size

    encoded_prompts = [x[-max_seq_length:] if len(x) > max_seq_length else x for x in encoded_prompts] # To limit the max_seq_length
    seqlens = [len(x) for x in encoded_prompts]
    # Cache
    cache_window = max(seqlens) + max(max_tokens)
    #cache_window = max_seq_length
    cache = BufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()

    # Bookkeeping
    logprobs: List[List[float]] = [[] for _ in range(batch_size)]
    num_generated_tokens = torch.tensor([0 for _ in range(batch_size)])
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s : s + chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        prelogits = model.forward(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache,
        )
        logits = torch.log_softmax(prelogits, dim=-1)

        if last_token_prelogits is not None:
            # Pass > 1
            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i_seq in range(batch_size):
                logprobs[i_seq].append(
                    last_token_logits[i_seq, prompt_chunks[i_seq][0]].item()
                )

        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            logprobs[i_seq].extend(
                [
                    logits[offset + i, sequence[i + 1]].item()
                    for i in range(len(sequence) - 1)
                ]
            )
            offset += len(sequence)

        last_token_prelogits = prelogits.index_select(
            0,
            torch.tensor(
                [len(p) for p in prompt_chunks], device=prelogits.device
            ).cumsum(dim=0)
            - 1,
        )
        assert last_token_prelogits.shape == (batch_size, vocab_size)

    # decode
    generated_tensors = []
    is_finished = torch.tensor([False for _ in range(batch_size)])
    
    assert last_token_prelogits is not None
    for _ in range(max(max_tokens)):
        next_tokens = None
        for idx in range(batch_size):
            next_token = sample(last_token_prelogits[idx], temperature=temperatures[idx], top_p=top_ps[idx], top_k = top_ks[idx])
            num_generated_tokens[idx] += 1
            if next_tokens == None:
                next_tokens = next_token
            else:
                next_tokens = torch.cat((next_tokens, next_token), 0)
        is_finished = is_finished ^ (next_token == tokenizer.eos_id).cpu()
        is_finished = is_finished ^ (num_generated_tokens == max_tokens[idx]).cpu()
        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        generated_tensors.append(next_tokens[:, None])
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
        for idx in range(batch_size):
            logprobs[idx].append(last_token_logits[idx, next_tokens[idx]].item())
            generated_text = tokenizer.decode(generated_tokens[idx])
            callback.process_output(idx, generated_text, num_generated_tokens[idx].item(), is_finished[idx].item())

        if is_finished.all():
            break
        last_token_prelogits = model.forward(next_tokens, seqlens=[1] * batch_size, cache=cache)
        assert last_token_prelogits.shape == (batch_size, vocab_size)

    return generated_tokens, logprobs


def sample(logits, temperature, top_p, top_k) -> torch.Tensor:
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_k(probs, top_p, top_k)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)

def sample_top_k(probs, top_p=0.0, top_k=40):
    if top_k > 0:
        probs_sort, probs_idx = torch.topk(probs, top_k)
    else:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    if top_p > 0.0:
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token