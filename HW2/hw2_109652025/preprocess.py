import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)

    # TO-DO 0: Other preprocessing function attemption
    # Begin your code

    text = BeautifulSoup(preprocessed_text, 'lxml').get_text()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = ToktokTokenizer().tokenize(text)
    lemma_words = [lemmatizer.lemmatize(word) for word in text]
    preprocessed_text = ' '.join(lemma_words)

    # End your code

    return preprocessed_text
