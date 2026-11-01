"""Model adapters for causal LM scoring, residual hooks and latent bridge."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

LABEL_ORDER = ("SUPPORTS", "REFUTES", "NEI")
PROMPT_VERSION = "v1_anchor_last_plain"
VERDICT_PROMPT_VERSION = "v2_fact_checking_answer"


def get_transformer_blocks(model) -> list[nn.Module]:
    base = getattr(model, "model", None)
    blocks = getattr(base, "layers", None)
    if blocks is None:
        raise TypeError("Only verified Llama/Qwen-style model.model.layers is supported")
    if len(blocks) != model.config.num_hidden_layers:
        raise ValueError("Block count differs from config.num_hidden_layers")
    return list(blocks)


def load_model(model_id: str, cache_dir: str | None = None, dtype: str = "bfloat16",
               device: str = "cuda"):
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    if device == "cpu" and dt != torch.float32:
        raise ValueError("CPU requires float32 for this implementation")
    source = Path(model_id)
    if not source.exists() and cache_dir:
        snapshots = Path(cache_dir) / ("models--" + model_id.replace("/", "--")) / "snapshots"
        matches = sorted(p for p in snapshots.glob("*") if (p / "config.json").exists())
        if not matches:
            raise FileNotFoundError(f"Model {model_id} has no complete local snapshot in {cache_dir}")
        source = matches[-1]
    tok = AutoTokenizer.from_pretrained(str(source), local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(source), local_files_only=True, torch_dtype=dt).to(device)
    model.config.use_cache = False
    return model, tok


def render_encoder(row: dict, condition: str = "full") -> tuple[str, dict[str, tuple[int, int]]]:
    if condition not in {"full", "full_bias", "claim_only", "evidence_only", "no_evidence"}:
        raise ValueError(condition)
    parts: list[str] = ["You are reading a fact-checking example.\n"]
    spans: dict[str, tuple[int, int]] = {}
    if condition != "evidence_only":
        parts.append("Claim: ")
        start = sum(map(len, parts))
        parts.append(row["claim"])
        spans["claim"] = (start, start + len(row["claim"]))
        parts.append("\n")
    if condition not in {"claim_only", "no_evidence"}:
        parts.append("Evidence:\n")
        for e in row["evidence"]:
            parts.append(f'[{e["evidence_id"]}] ')
            start = sum(map(len, parts))
            parts.append(e["text"])
            spans["evidence"] = (start, start + len(e["text"]))
            parts.append("\n")
    else:
        parts.append("Evidence: [not provided]\n")
    if condition == "full_bias" and row.get("claim_metadata"):
        parts.append("Claim metadata: " + ", ".join(f"{k}={v}" for k, v in
                                                    sorted(row["claim_metadata"].items())) + "\n")
    parts.append("Relation anchor:")
    spans["anchor"] = (sum(map(len, parts)) - len("Relation anchor:"), sum(map(len, parts)))
    return "".join(parts), spans


def render_verdict(row: dict, condition: str = "full") -> str:
    if condition not in {"full", "full_bias", "claim_only", "evidence_only", "no_evidence"}:
        raise ValueError(condition)
    claim = "[not provided]" if condition == "evidence_only" else row["claim"]
    evidence = ("[not provided]" if condition in {"claim_only", "no_evidence"} else
                "\n".join(f'[{e["evidence_id"]}] {e["text"]}' for e in row["evidence"]))
    prompt = (
        "You are a fact-checking assistant.\n\n"
        "Determine whether the evidence supports or refutes\n"
        "the claim, using only the provided evidence.\n\n"
        "Labels:\n"
        "SUPPORTS: The evidence entails the claim.\n"
        "REFUTES: The evidence contradicts the claim.\n"
        "NEI: The evidence is insufficient to determine\n"
        "whether the claim is supported or refuted.\n\n"
        "Return exactly one label.\n\n"
        f"Claim: {claim}\n"
        f"Evidence: {evidence}\n"
    )
    if condition == "full_bias" and row.get("claim_metadata"):
        prompt += "Claim metadata: " + ", ".join(
            f"{key}={value}" for key, value in sorted(row["claim_metadata"].items())) + "\n"
    return prompt + "\nAnswer:"


def render_decoder(row: dict | None = None, guided: bool = False, slot: str = "<latent>",
                   structured: bool = False) -> str:
    if structured and not guided:
        raise ValueError("Structured verification needs claim and evidence")
    text = slot + "\nQuestion: Decide the claim-evidence relation. Answer with one label: SUPPORTS, REFUTES, or NEI."
    if structured:
        text = (slot + "\nQuestion: Verify the claim using the evidence. Return one JSON object with "
                "verdict, evidence_ids, and justification. Cite only supplied evidence IDs.")
    if guided:
        if row is None:
            raise ValueError("Guided decoder requires sample")
        # Gold labels never enter the prompt.
        text += "\nClaim: " + row["claim"] + "\nEvidence:\n"
        text += "\n".join(f'[{e["evidence_id"]}] {e["text"]}' for e in row["evidence"])
    return text + ("\nJSON:" if structured else "\nVerdict:")


def _indices_for_span(offsets: Sequence[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    a, b = span
    indices = [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]
    if not indices:
        raise ValueError(f"No tokens overlap character span {span}")
    return indices


def token_positions(tokenizer, text: str, spans: dict[str, tuple[int, int]]) -> dict[str, list[int]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    result = {name: _indices_for_span(enc["offset_mapping"], span) for name, span in spans.items()}
    result["final_anchor"] = [len(enc["input_ids"]) - 1]
    return result


def last_nonpadding(attention_mask: torch.Tensor) -> torch.Tensor:
    if (attention_mask.sum(1) == 0).any():
        raise ValueError("All-padding sequence")
    return torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask).masked_fill(
        attention_mask == 0, -1).max(dim=1).values


@contextmanager
def temporary_hook(block: nn.Module, hook: Callable):
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _replace_output(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    return new_hidden


def make_patch_hook(index: int, source: torch.Tensor):
    def hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if source.shape != hidden[:, index, :].shape:
            raise ValueError(f"Patch shape {tuple(source.shape)} != {tuple(hidden[:, index, :].shape)}")
        mask = torch.zeros(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
        mask[:, index] = True
        new = torch.where(mask.unsqueeze(-1), source.to(hidden.device, hidden.dtype).unsqueeze(1), hidden)
        return _replace_output(output, new)
    return hook


def make_capture_hook(storage: dict, layer: int, indices: list[int], detach: bool):
    def hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        value = hidden[:, indices, :].mean(dim=1)
        storage[layer] = value.detach().float().cpu() if detach else value
    return hook


def encode_text(tokenizer, text: str, device: torch.device | str) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.shape[1] == 0:
        raise ValueError("Empty encoded prompt")
    return ids.to(device)


def read_activation(model, tokenizer, text: str, layer: int, indices: list[int],
                    detach: bool = True) -> torch.Tensor:
    storage = {}
    hook = make_capture_hook(storage, layer, indices, detach)
    ids = encode_text(tokenizer, text, model.device)
    with temporary_hook(get_transformer_blocks(model)[layer], hook):
        model(input_ids=ids, use_cache=False)
    return storage[layer]


def extract_all(model, tokenizer, text: str, positions: dict[str, list[int]],
                layers: list[int] | None = None) -> dict[str, dict[int, torch.Tensor]]:
    blocks = get_transformer_blocks(model)
    layers = list(range(len(blocks))) if layers is None else layers
    storage = {name: {} for name in positions}
    handles = []
    try:
        for layer in layers:
            def hook(_module, _args, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                for name, idx in positions.items():
                    storage[name][layer] = hidden[:, idx, :].mean(1).detach().float().cpu()
            handles.append(blocks[layer].register_forward_hook(hook))
        with torch.inference_mode():
            model(input_ids=encode_text(tokenizer, text, model.device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return storage


def score_candidate_sequences(model, tokenizer, prefix: str, candidates: Sequence[str] = LABEL_ORDER,
                              hook_block: nn.Module | None = None, hook: Callable | None = None,
                              normalize_by_length: bool = False) -> torch.Tensor:
    """Differentiable teacher-forced log likelihood of *every token* in each answer.

    Candidate tokens are deliberately encoded as separate continuations with a leading
    space. This fixes the boundary convention across all labels and all stages.
    """
    prefix_ids = encode_text(tokenizer, prefix, model.device)
    scores = []
    for label in candidates:
        continuation = encode_text(tokenizer, " " + label, model.device)
        ids = torch.cat([prefix_ids, continuation], dim=1)
        if hook_block is not None:
            with temporary_hook(hook_block, hook):
                logits = model(input_ids=ids, use_cache=False).logits
        else:
            logits = model(input_ids=ids, use_cache=False).logits
        start = prefix_ids.shape[1] - 1
        selected = logits[:, start:start + continuation.shape[1], :].float()
        logp = selected.log_softmax(dim=-1).gather(-1, continuation.unsqueeze(-1)).squeeze(-1).sum(-1)
        if normalize_by_length:
            logp = logp / continuation.shape[1]
        scores.append(logp)
    return torch.stack(scores, dim=-1)


def label_margin(scores: torch.Tensor, label_index: int) -> torch.Tensor:
    mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
    mask[label_index] = False
    return scores[..., label_index] - torch.logsumexp(scores[..., mask], dim=-1)


class LowRankLinear(nn.Module):
    """Self-contained LoRA wrapper; base Linear is frozen and checkpointed separately."""
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        delta = nn.functional.linear(nn.functional.linear(self.dropout(x).float(), self.A), self.B)
        return self.base(x) + (self.scale * delta).to(x.dtype)


def add_lora(model, layers: list[int], modules: Sequence[str] = ("q_proj", "v_proj"),
             rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> list[nn.Parameter]:
    model.requires_grad_(False)
    blocks = get_transformer_blocks(model)
    params = []
    for layer in layers:
        if not 0 <= layer < len(blocks):
            raise ValueError(f"Invalid layer {layer}")
        block = blocks[layer]
        for name in modules:
            parent = getattr(block, "self_attn")
            base = getattr(parent, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"{name} is not a Linear module")
            wrapped = LowRankLinear(base, rank, alpha, dropout)
            setattr(parent, name, wrapped)
            params.extend([wrapped.A, wrapped.B])
    return params


def lora_state(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            if name.endswith(".A") or name.endswith(".B")}


def load_lora_state(model, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = {name for name in model.state_dict() if name.endswith(".A") or name.endswith(".B")}
    if set(state) != expected:
        raise ValueError(f"LoRA checkpoint/module mismatch: missing={sorted(expected - set(state))[:5]}, "
                         f"unexpected={sorted(set(state) - expected)[:5]}")
    model.load_state_dict(state, strict=False)


class LatentBridge(nn.Module):
    def __init__(self, encoder_hidden: int, decoder_hidden: int, decoder, tokenizer,
                 write_layer: int = 0):
        super().__init__()
        self.projector = nn.Linear(encoder_hidden, decoder_hidden, bias=True,
                                   device=decoder.device, dtype=torch.float32)
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.write_layer = write_layer

    def score(self, latent: torch.Tensor, row: dict | None = None, guided: bool = False,
              candidates: Sequence[str] = LABEL_ORDER) -> torch.Tensor:
        slot = self.tokenizer.eos_token
        prompt = render_decoder(row, guided, slot=slot)
        slot_ids = token_positions(self.tokenizer, prompt, {"slot": (0, len(slot))})["slot"]
        if len(slot_ids) != 1:
            raise ValueError("Latent slot must tokenize to exactly one token")
        projected = self.projector(latent.float().to(self.projector.weight.device))
        return score_candidate_sequences(self.decoder, self.tokenizer, prompt, candidates,
                                         get_transformer_blocks(self.decoder)[self.write_layer],
                                         make_patch_hook(slot_ids[0], projected))

    def score_structured_target(self, latent: torch.Tensor, row: dict, target_json: str) -> torch.Tensor:
        slot = self.tokenizer.eos_token
        prompt = render_decoder(row, guided=True, slot=slot, structured=True)
        slot_idx = token_positions(self.tokenizer, prompt, {"slot": (0, len(slot))})["slot"]
        if len(slot_idx) != 1:
            raise ValueError("Latent slot must be one token")
        projected = self.projector(latent.float().to(self.projector.weight.device))
        score = score_candidate_sequences(self.decoder, self.tokenizer, prompt, [target_json],
                                          get_transformer_blocks(self.decoder)[self.write_layer],
                                          make_patch_hook(slot_idx[0], projected))
        length = encode_text(self.tokenizer, " " + target_json, self.decoder.device).shape[1]
        return -score.mean() / length

    def generate_structured(self, latent: torch.Tensor, row: dict, max_new_tokens: int = 128) -> str:
        slot = self.tokenizer.eos_token
        prompt = render_decoder(row, guided=True, slot=slot, structured=True)
        slot_idx = token_positions(self.tokenizer, prompt, {"slot": (0, len(slot))})["slot"]
        if len(slot_idx) != 1:
            raise ValueError("Latent slot must be one token")
        projected = self.projector(latent.float().to(self.projector.weight.device))
        ids = encode_text(self.tokenizer, prompt, self.decoder.device)
        generation_config = copy.deepcopy(self.decoder.generation_config)
        generation_config.do_sample = False
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50
        with temporary_hook(get_transformer_blocks(self.decoder)[self.write_layer],
                            make_patch_hook(slot_idx[0], projected)):
            output = self.decoder.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                           generation_config=generation_config,
                                           max_new_tokens=max_new_tokens, use_cache=False,
                                           pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
