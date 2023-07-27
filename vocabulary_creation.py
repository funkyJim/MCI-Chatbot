import pandas as pd
import unicodedata as ud
import json
import spacy
import pickle
from autocorrect import AutoCorrect

"""
To be able to use natural language processing techniques in Greek in the chatbot application, it was necessary to have in format
file a Greek dictionary of satisfactory size. To create this dictionary we used a Greek corpus which according to
authors in the reference, is the largest Greek corpus available at internet up to and including the date of writing their article. 
This corpus created by web crawling of websites with Greek content.
Specifically, it extracted the content from 20 million websites in a period of 45 days. 
Then, this text underwent several stages of pre-processing and cleaned of repetitive, non-Greek sentences
characters, advertisements, image text, and any form of code so to come in a form suitable 
for training machine learning models and dictionary production. The result was a corpus of about 50GB in size
ready to use in various natural language processing applications.

Reference:
Outsios Stamatis, Konstantinos Skianis, Polykarpos Meladianos, Christos Xypolopoulos,  and Michalis Vazirgiannis.
"Word embeddings from large-scale greek web content."
arXiv preprint arXiv:1810.06694 (2018).
"""

# Create vocabulary for autocorrect module from lexiko.csv, words in lexiko.csv are all lowercase
df = pd.read_csv('Data/unprocessed/lexiko.csv', encoding='windows-1253', sep=';')
# Some very common words were missing from the corpus, so we had to add them and assign to them a very large word_count
# so that the autocorrect module would work properly
df.loc[len(df)] = ['θα', 200000000]
df.loc[len(df)] = ['να', 200000000]
df.loc[len(df)] = ['σε', 200000000]
df.loc[len(df)] = ['με', 200000000]
df.loc[len(df)] = ['οκ', 200000000]
df.loc[len(df)] = ['βρες', 200000000]
df.sort_values(by=['COUNT'], ascending=False, inplace=True, ignore_index=True)
df.rename(columns={'WORD': 'Word', 'COUNT': 'Count'}, inplace=True)
df.drop(df[df['Count'] < 2000].index, inplace=True)
print(len(df))

d = {ord('\N{COMBINING ACUTE ACCENT}'): None}  # strip accents, ά -> α
df['Word_stripped'] = df['Word'].apply(lambda w: ud.normalize('NFD', w).translate(d))

print(df.head(10))

df.to_pickle('Data/processed/lexiko.pkl')

# -------------------------------------------------------------------
# Create vocabulary from products.json
# This vocabulary is going to be used by the Chatbot to search for products in the user's input message
nlp = spacy.load('el_core_news_lg')
ac = AutoCorrect()

with open('Data/unprocessed/products.json', 'r', encoding='UTF-8') as f:
    products = json.load(f)

lemmas = []
for p in products:
    lemmas.extend([nlp(w)[0].lemma_ for w in ac.autocorrect(p['product_name']).split()])
    lemmas.extend([nlp(w)[0].lemma_ for w in ac.autocorrect(p['category']).split()])
lemmas = [lem for lem in lemmas if len(lem) > 3]  # remove lemmas like 'για', 'σε', etc.
lemmas = set(lemmas)
# print(lemmas)

with open('Data/processed/products_vocab.pkl', 'wb') as f:
    pickle.dump(lemmas, f)
