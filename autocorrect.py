import pickle
from jellyfish import damerau_levenshtein_distance
import unicodedata as ud
import re


class AutoCorrect:
    def __init__(self, word_freq_fpath='Data/processed/lexiko.pkl'):
        with open(word_freq_fpath, 'rb') as f:
            self.df = pickle.load(f)  # dataframe with columns: word, count, word_stripped
        self.vocab = set(self.df['Word_stripped'].tolist())
        self.SIMILARITY_THRESHOLD = 2  # larger than 2 is considered very different

    def autocorrect_word(self, word, print_info=False):
        """
        Algorithm steps:
            1. strip accent, keep only greek characters, lowercase
            2. if word is empty:
                    return empty string
                else:
                    go to step 3
            3. if word is in vocabulary:
                    return word
                else:
                    go to step 4
            4. calculate similarity using Damerau-Levenshtein distance, for each word in the vocabulary
            5. if max_similarity <= threshold:
                    return most similar word with the highest count
                else:
                    return original word
        """
        original_word = word
        if print_info:
            print(word)

        ## Strip accents and lower case
        d = {ord('\N{COMBINING ACUTE ACCENT}'): None}  # strip accents, ά -> α
        word = ud.normalize('NFD', word).translate(d)
        word = re.sub("[^α-ωΑ-Ω]", "", word)
        word = word.lower()

        if word == '':
            return word
        if word in self.vocab:
            if print_info:
                print("Word in Vocabulary\n")
            # there may be two words with different accent, but we only return the first because our df is sorted
            out = self.df.loc[self.df['Word_stripped'] == word]['Word'].tolist()[0]
            return out
        else:
            sim = []
            for w in self.df['Word_stripped'].tolist():
                sim.append(damerau_levenshtein_distance(w, word))

            self.df['Similarity'] = sim
            df_sorted = self.df.sort_values(by=['Similarity', 'Count'], ascending=[True, False])
            if print_info:
                print(df_sorted.head(5))

            out = df_sorted.iloc[0]
            if out['Similarity'] > self.SIMILARITY_THRESHOLD:
                return original_word
            else:
                return out['Word']

    def autocorrect(self, sentence, print_info=False):
        corr_words = []
        for w in sentence.split():
            corr_word = self.autocorrect_word(w, print_info)
            if corr_word != '':
                corr_words.append(corr_word)
        corr_sentence = ' '.join(corr_words)

        return corr_sentence


if __name__ == "__main__":
    print('\nType "exit" to terminate.\n')
    ac = AutoCorrect()
    while True:
        inp = input(">>> ")
        if inp == 'exit':
            exit(0)
        else:
            print(ac.autocorrect(inp, print_info=True))
