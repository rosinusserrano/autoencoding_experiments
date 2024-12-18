"""Utility functions for metric collection and logging."""


def average_dict_values(d: dict[str, list[float]]) -> dict[str, float]:
    """Update each entry in the dictionary to be its average."""
    for key, values in d.items():
        d[key] = sum(values) / len(values)
    return d


def append_dict_values(
    dest: dict[str, list[float]],
    src: dict[str, float],
) -> dict[str, list[float]]:
    """Append values from src to dest."""
    for key, val in src.items():
        if key not in dest:
            dest[key] = [val]
        else:
            dest[key].append(val)
    return dest
