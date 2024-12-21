import spacy


def get_lemmas_with_contexts(filename: str):

    text = None
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    for sent in doc.sents:
        for token in sent:
            print(token.lemma_)


if __name__ == "__main__":
    print(get_lemmas_with_contexts("./bearwiki.txt"))
