from typing import List
from torch import Tensor, no_grad
from torch.nn.functional import cosine_similarity
from transformers import BertTokenizer, BertModel
import sys


def get_example_from_filename(filename: str) -> str:
    text = None
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def get_senses_from_filename(filename: str) -> List[str]:
    senses = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            senses.append(line)
    return senses


class SenseDisambiguator:

    def __init__(self, english_or_korean: str):
        pretrained_model = ""
        if english_or_korean == "korean":
            pretrained_model = "klue/bert-base"
        elif english_or_korean == "english":
            pretrained_model = "bert-base-uncased"
        else:
            raise ValueError("`english_or_korean` must be 'english' or 'korean'.")

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertModel.from_pretrained(pretrained_model)

    def get_closest_senses(self, text: str, senses: List[str]) -> List[int]:
        text_pt = self.get_embeddings(text)
        sense_pts = [self.get_embeddings(sense) for sense in senses]

        cosine_sims = self.get_cosine_similarities(text_pt, sense_pts)

        return self.get_similarities_above_threshold(cosine_sims)

    def get_embeddings(self, text: str) -> Tensor:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with no_grad():
            embeddings = self.model(**inputs)
        return embeddings.last_hidden_state.mean(dim=1)

    def get_cosine_similarities(
        self, example_tensor: Tensor, sense_tensors: List[Tensor]
    ) -> List[float]:
        return [
            cosine_similarity(example_tensor, sense_tensor)
            for sense_tensor in sense_tensors
        ]

    def get_similarities_above_threshold(self, cosine_similarities: List[float]):
        return [
            (sense_id, cosine_similarities[sense_id])
            for sense_id in range(len(cosine_similarities))
            if cosine_similarities[sense_id] > 0.5
        ]


if __name__ == "__main__":

    args = sys.argv

    if len(args) < 2:
        raise ValueError("You must pass 'english' or 'korean' as the first arg.")

    example_filename = ""
    sense_filename = ""

    if args[1] == "english":
        example_filename = "./bearwiki.txt"
        sense_filename = "./bearsenses.txt"
    elif args[1] == "korean":
        example_filename = "./밤위키.txt"
        sense_filename = "./밤뜻풀이.txt"
    else:
        raise ValueError("The first arg passed must be 'english' or 'korean'.")

    example = get_example_from_filename(example_filename)
    senses = get_senses_from_filename(sense_filename)

    sd = SenseDisambiguator(args[1])

    best_indices = sd.get_closest_senses(example, senses)

    for tuple in best_indices:
        print(tuple, senses[tuple[0]])
