import json
import pickle
import random
import numpy as np
from keras.models import load_model
import calendar
import time
import datetime
import spacy

nlp = spacy.load('el_core_news_lg')


class MyChatbot:
    def __init__(self,
                 lemmas_fpath='Data/all_lemmas_from_patterns.pkl',
                 products_fpath='Data/processed/products_vocab.pkl',
                 model_fpath='Models/intent_classification.h5',
                 intents_fpath='Data/intents.json'
                 ):
        with open(lemmas_fpath, 'rb') as f:
            self.lemmas = pickle.load(f)
        with open(products_fpath, 'rb') as f:
            self.products = pickle.load(f)
        self.model = load_model(model_fpath)
        with open(intents_fpath, encoding='utf-8') as f:
            self.intents = json.load(f)
        self.PROBABILITY_THRESHOLD = 0.75

    def bag_of_words(self, msg):
        msg_lemmas = [word.lemma_ for word in nlp(msg)]  # tokenize the message and lemmatize
        bag = [0] * len(self.lemmas)  # bag of words
        for lem in msg_lemmas:
            if lem in self.lemmas:
                bag[self.lemmas.index(lem)] += 1
        return np.array([bag])

    def search_for_product_in_message(self, msg):
        product_words = []
        for w in msg.split():
            tok = nlp(w)[0]
            if tok.lemma_ in self.products:
                product_words.append(tok.text)
        filtered_msg = ' '.join(product_words)

        return filtered_msg

    def get_response(self, msg):
        """
            TODO:

            msg: message received by the user after autocorrection
        """

        # ------------ If there is a product or a product_category in the user's message, we need redirect him to the corresponding e-shop page
        filtered_msg = self.search_for_product_in_message(msg)
        if filtered_msg != '':
            response = "Είναι αυτό το προϊόν που ψάχνετε?"
            action = 'search_for_product'
            tag = 'search_for_product'
        else:
            # ------------ In any other case, the user's message is fed as input to the model
            inp = self.bag_of_words(msg)
            out = self.model.predict(inp, verbose=0)[0]  # model returns a list of lists
            # print(out)

            idx_max = np.argmax(out)
            # print(out[idx_max])
            if out[idx_max] > self.PROBABILITY_THRESHOLD:
                response = random.choice(self.intents['intents'][idx_max]['responses'])
                action = self.intents['intents'][idx_max]['action']
                tag = self.intents['intents'][idx_max]['tag']
            else:  # No answer
                response = 'Με συγχωρείτε, δεν το κατάλαβα αυτό...'
                action = ''
                tag = ''

        return response, action, filtered_msg, tag


class ConversationLogger:
    """
        Python doesn't guarantee that the destructor will be called when the program closes,
        so it is better open and close the file each time a new input arrives.
    """
    def __init__(self):
        ts = calendar.timegm(time.gmtime())  # timestamp
        self.logfile_path = 'logfiles/conversation_' + str(ts) + '.txt'

    def log(self, user_msg, usr_msg_corr, bot_resp, bot_tag):
        with open(self.logfile_path, 'a', encoding="utf-8") as f:
            usr_msg = 'USER MESSAGE:\t\t' + user_msg + '\n'
            usr_msg_corr = 'CORRECTED MESSAGE:\t' + usr_msg_corr + '\n'
            bot_resp = 'BOT RESPONSE:\t\t' + bot_resp + '\n'
            bot_tag = 'BOT TAG:\t\t' + bot_tag + '\n\n'
            # Append logs in the logfile
            f.write(str(datetime.datetime.now())[:-7])
            f.write('\n')
            f.write(usr_msg)
            f.write(usr_msg_corr)
            f.write(bot_resp)
            f.write(bot_tag)


if __name__ == "__main__":
    from autocorrect import AutoCorrect
    print('\nType "exit" to terminate.\n')
    ac = AutoCorrect()
    bot = MyChatbot()
    logger = ConversationLogger()
    while True:
        msg = input(">>> ")
        if msg == 'exit':
            exit(0)
        else:
            msg_corr = ac.autocorrect(msg)
            print(f'Corrected msg: {msg_corr}')
            resp, _, _, tag = bot.get_response(msg_corr)
            logger.log(msg, msg_corr, resp, tag)
            print(f'Predicted tag: {tag}')
            print(f'Chosen response: {resp}')
