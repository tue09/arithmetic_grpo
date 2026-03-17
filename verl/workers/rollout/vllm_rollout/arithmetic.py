import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class ArithmeticSamplingProcessor(AdapterLogitsProcessor):
    """Select tokens from a deterministic arithmetic code carried in SamplingParams.extra_args."""

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        extra_args = params.extra_args or {}
        if "arithmetic_code" not in extra_args:
            return None

        state = {"code": float(extra_args["arithmetic_code"]) % 1.0}

        def force_token(output_ids, logits):
            probs = torch.softmax(logits.float(), dim=-1)
            cdf = probs.cumsum(dim=-1)
            code = min(max(state["code"], 0.0), 1.0 - 1e-12)
            token = int(
                torch.searchsorted(
                    cdf,
                    torch.tensor(code, device=cdf.device, dtype=cdf.dtype),
                    right=False,
                ).item()
            )
            token = min(token, logits.shape[-1] - 1)
            left = 0.0 if token == 0 else float(cdf[token - 1].item())
            width = max(float(probs[token].item()), 1e-12)
            state["code"] = (code - left) / width
            logits.fill_(float("-inf"))
            logits[token] = 0.0
            return logits

        return force_token
