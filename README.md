# Headword Finder

## Goal

The goal of this project is to create a tool for finding which of a set of headwords is being used in a specific usage instance for my Korean dictionary.

Users need to be able to input text and have words used in that text automatically derived and added to the correct dictionary entries. This requires finding a list of the
dictionary words (lemmas) in the text and which of the meanings of the word (headword) is specifically being used in the text.

Headwords are made up of multiple senses. For example, [Wikipedia](https://en.wikipedia.org/wiki/Lemma_(morphology)#Headword) uses the headword 'bread' with senses for the 
food 'bread' and another for money 'bread'. Because these senses fundamentally describe the same thing (one is just idiomatic), they are under the same headword.

This is similar to [word-sense disambiguation](https://en.wikipedia.org/wiki/Word-sense_disambiguation) but for my purposes I only need to find which headword is being used, 
which makes things easier.

## Process

### Korean Data

The model is trivially able to pick out which headword is used when one far exceeds all others in frequency, as just picking the most common headword will yield high accuracy rates. As a result, I purposely picked mostly words that have more than one common meaning to find the model that performs best facing harder problems.

Words used include 눈 \[eye / snow\], 못 \[can't / pond / nail\], and 밤 \[chestnut / night\], but most were two syllable Sino Korean words.

The data I have access to is from [Urimalsaem 우리말샘](https://opendict.korean.go.kr/main), which contains over one million headwords.
For each of these headwords, I have a list of definitions paired with any number of example sentences for that headword.

### Scoring Headwords

To find the likelihood that the example sentence matches the definition or any example sentence, I compare the [KoBERT](https://github.com/SKTBrain/KoBERT) embeddings for
the sentences. Example sentences compare the embeddings for the specific lemma being tested, while the definitions compare the overall embedding of the sentence since the
definition does not contain the word (if it's a good definition).

Multiple parameters needed to be tested to find a suitable model for my use case. These include:
- The weighted importances of definition similarity vs. example usage similarity
- How to "score" the similarity of a headword given all of the scores of its senses, each with a definition and list of example usages
- When to confidently return a specific headword vs. return nothing

Here is an example output from running the model on an English sentence with the word 'pen':
```
$ python run_single_test.py english inputs/eng/pen.json 0.2 average max max 
======================================================================
Unknown usage: After putting the cows back into their pen, the farmer worked on repairing the part of the fence they had broken through. | source: me
0.61609 | An enclosure (enclosed area) used to contain domesticated animals, especially sheep or cattle.
0.49334 | A tool, originally made from a feather but now usually a small tubular instrument, containing ink used to write or make marks.

Correct - avg incorrect is 0.12275516986846924
Chosen index (min acceptance 0.5, min delta 0.05) is 1
======================================================================
```
The argument `0.2` is the weight of sense definitions (so `0.8` is the weight of sense example usages).
The next three arguments are for turning lists of similarities into single scores.
The first `average` means that the score of a list of sense example usages is their average.
The second `max` means that the best set of example usages is used for each headword.
The third `max` means that the best definition is used for each headword.

In this case, the headword for an animal pen was over the `min acceptance` score (0.61 > 0.5) and the difference between it and second place was at least `min delta` (0.12 > 0.05), so it is confident that that headword is the correct one.








