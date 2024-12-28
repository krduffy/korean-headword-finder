from typing import List, Literal, Tuple
from torch import Tensor, no_grad
from torch.nn.functional import cosine_similarity
from transformers import BertTokenizer, BertModel
from token_weighting import TokenWeightingStrategy

type Language = Literal["english", "korean"]


class SenseDisambiguator:

    def __init__(self, language: Language, weighting_strategy: TokenWeightingStrategy):
        pretrained_model = ""

        if language == "korean":
            pretrained_model = "klue/bert-base"

            from spacy.lang.ko import stop_words

            self.token_weigher = weighting_strategy(stopwords=stop_words.STOP_WORDS)

        elif language == "english":
            pretrained_model = "bert-base-uncased"

            from spacy.lang.en import stop_words

            self.token_weigher = weighting_strategy(stopwords=stop_words.STOP_WORDS)

        else:
            raise ValueError("`language` must be 'english' or 'korean'.")

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertModel.from_pretrained(pretrained_model)

    def get_ordered_sense_indices(
        self, text: str, senses: List[str]
    ) -> List[Tuple[int, float]]:
        text_pt = self._get_embeddings(text)
        sense_pts = [self._get_embeddings(sense) for sense in senses]

        cosine_sims = self._get_cosine_similarities(text_pt, sense_pts)

        indices_with_sim = [
            (index, cosine_sims[index]) for index in range(len(cosine_sims))
        ]
        indices_with_sim.sort(key=lambda t: t[1], reverse=True)

        return indices_with_sim

    def _get_embeddings(self, text: str) -> Tensor:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )

        with no_grad():
            embeddings = self.model(**inputs)

        last_hidden_state = embeddings.last_hidden_state
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        mask = self.token_weigher.get_mask(tokens)

        weighted_embeddings = last_hidden_state[0] * mask.unsqueeze(-1)

        sum_embeddings = weighted_embeddings.sum(dim=0)
        sum_weights = mask.sum()

        return (sum_embeddings / sum_weights).unsqueeze(0)

    def _get_cosine_similarities(
        self, example_tensor: Tensor, sense_tensors: List[Tensor]
    ) -> List[float]:
        return [
            cosine_similarity(example_tensor, sense_tensor).item()
            for sense_tensor in sense_tensors
        ]
