import random
from functools import lru_cache


@lru_cache(maxsize=256)
def make_arithmetic_codes(group_size: int, seed: int) -> tuple[float, ...]:
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")

    shift = random.Random(seed).random()
    return tuple(((i + 0.5) / group_size + shift) % 1.0 for i in range(group_size))


def get_arithmetic_code(group_size: int, seed: int, rollout_n: int) -> float:
    codes = make_arithmetic_codes(group_size=group_size, seed=seed)
    return codes[rollout_n % len(codes)]
