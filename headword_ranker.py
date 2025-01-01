from similarity_calculator import SimilarityCalculator


def rank_headwords(
    num_headwords,
    result_tensors,
    definition_weight: float,
    similarity_calculator: SimilarityCalculator,
):
    (
        average_token_embedding_for_unknown_usage,
        average_token_embeddings_for_headword_sense_definitions,
        lemma_embedding_for_unknown_usage,
        lemma_embeddings_for_headword_sense_known_usages,
    ) = result_tensors

    definition_similarities = [0] * num_headwords
    if definition_weight > 0.0:
        definition_similarities = (
            similarity_calculator.get_similarities_of_sense_definitions(
                average_token_embedding_for_unknown_usage,
                average_token_embeddings_for_headword_sense_definitions,
            )
        )

    usage_similarities = [0] * num_headwords
    if 1 - definition_weight > 0.0:
        usage_similarities = (
            similarity_calculator.get_similarities_of_sense_known_usages(
                lemma_embedding_for_unknown_usage,
                lemma_embeddings_for_headword_sense_known_usages,
            )
        )

    weighted_similarities = [
        definition_similarities[index] * definition_weight
        + usage_similarities[index] * (1 - definition_weight)
        for index in range(num_headwords)
    ]

    indices_with_sim = [
        (index, weighted_similarities[index])
        for index in range(len(weighted_similarities))
    ]
    indices_with_sim.sort(key=lambda t: t[1], reverse=True)

    return indices_with_sim
