from typing import List, Tuple


class HeadwordChooser:

    def __init__(self, min_acceptance: float, min_delta: float):
        self.min_acceptance = min_acceptance
        self.min_delta = min_delta

    def choose_headword(
        self, descending_ranking: List[Tuple[int, float]]
    ) -> int | None:

        if len(descending_ranking) == 1:
            return 0

        top_similarity = descending_ranking[0][1]

        if top_similarity < self.min_acceptance:
            return None

        delta = top_similarity - descending_ranking[1][1]

        if delta < self.min_delta:
            return None

        return descending_ranking[0][0]
