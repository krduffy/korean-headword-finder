from typing import List, Tuple


def get_score(
    correct_headword_id: int, headword_similarities: List[Tuple[int, float]]
) -> float:

    for headword_id, similarity in headword_similarities:
        if headword_id == correct_headword_id:
            correct_headword_similarity = similarity
            break

    sum_of_incorrect = (
        sum([headword_similarity[1] for headword_similarity in headword_similarities])
    ) - correct_headword_similarity

    average_of_incorrect = 0

    if len(headword_similarities) >= 2:
        average_of_incorrect = sum_of_incorrect / (len(headword_similarities) - 1)

    return correct_headword_similarity - average_of_incorrect
