from collections import Counter
import warnings

from utils import str_utils

class VocabularyBuilder():
    """Vocabulary builder class to generate vocabulary."""
    
    def __init__(self, max_size = 512):
        self._words_counter = Counter()
        self.max_size = max_size
        self._vocabulary = { '<PAD>':0, '<NUMBER>':1, '<RARE>':2 }
        self.built = False
        
    def add(self, word):
        if not str_utils.is_number(word):
            self._words_counter.update([word.lower()])
            
    def build(self):
        for word, count in self._words_counter.most_common(self.max_size):
            self._vocabulary[word] = len(self._vocabulary)
        print(f"Vocabulary of size {len(self._vocabulary)} built!")
        self.built = True
        return self._vocabulary

    def get_vocab(self):
        if not self.built:
            warnings.warn(
                "The vocabulary is not built. Use VocabularyBuilder.build(). Returning default vocabulary.", Warning)
            return self._vocabulary
        else:
            return self._vocabulary

