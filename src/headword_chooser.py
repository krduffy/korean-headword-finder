from typing import List, Tuple


def choose_headword(
    descending_ranking: List[Tuple[int, float]],
    *,
    min_acceptance: float,
    min_delta: float
) -> int | None:

    if len(descending_ranking) == 1:
        return 0

    top_similarity = descending_ranking[0][1]

    if top_similarity < min_acceptance:
        return None

    delta = top_similarity - descending_ranking[1][1]

    if delta < min_delta:
        return None

    return descending_ranking[0][0]
