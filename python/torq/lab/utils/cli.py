# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

def promote_string_val(val: str) -> str | int | float | bool:
    for type_func in (int, float):
        try:
            return type_func(val)
        except (TypeError, ValueError):
            continue
    lowered = val.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    return val


def parse_remainder_args_to_dict(remainder_args: list[str], remainder_name: str) -> dict:
    if not remainder_args:
        return {}
    if not remainder_name.startswith("--"):
        remainder_name = "--" + remainder_name
    if remainder_args[0] == remainder_name:
        remainder_args = remainder_args[1:]
        if not remainder_args:
            return {}
    if len(remainder_args) % 2 != 0:
        raise ValueError(
            f"Trailing or missing value in remainder style args: {remainder_args}"
        )
    args_it = iter(remainder_args)
    return {
        key.lstrip("-"): promote_string_val(val)
        for key, val in zip(args_it, args_it)
    }
