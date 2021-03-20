import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog'
    'Do you think my dog is amazing'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
print(word_index)
print(sequences)

test_seq = [
    'I really love my dog',
    'My dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_seq)

print("#"*30)
print(test_seq)

# We need lot of vocabulary to train the data
# In many cases in case of just ignoring the unseen words its better to put a special value