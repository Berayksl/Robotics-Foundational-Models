from nltk.corpus import wordnet as wn

for syn in wn.synsets("chair"):
    print(syn.name(), "->", syn.definition())
