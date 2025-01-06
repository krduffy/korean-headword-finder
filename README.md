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

Most of the words I have tested were Korean words with more than one commonly used headword. This includes some native Korean words like 
눈 \[eye / snow\], 못 \[can't / pond / nail\], and 밤 \[chestnut / night\] but most were two syllable Sino Korean words.

The data I have access to is from [Urimalsaem 우리말샘](https://opendict.korean.go.kr/main), which contains over one million headwords.
For each of these headwords, I have a list of definitions paired with any number of example sentences for that headword.

To find the likelihood that the example sentence matches the definition or any example sentence, I compare the [KoBERT](https://github.com/SKTBrain/KoBERT) embeddings for
the sentences. Example sentences compare the embeddings for the specific lemma being tested, while the definitions compare the overall embedding of the sentence since the
definition does not contain the word (if it's a good definition).

Multiple parameters needed to be tested to find a suitable model for my use case. These include:
- The weighted importances of definition similarity vs. example usage similarity
- How to "score" the similarity of a headword given all of the scores of its senses, each with a definition and list of example usages
- When to confidently return a specific headword vs. return nothing

Here is an example output from running
