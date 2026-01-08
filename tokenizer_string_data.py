from tensorflow.keras.preprocessing.text import Tokenizer


data = ["The earth is spherical.",    "The earth is a planet.",    "I like to eat at a restaurant."]
# Filter the punctiations, tokenize the words and index them to integers


tokenizer = Tokenizer(num_words=15, filters='''!"#$%&()*+,-./:;<=>?[\\]^_'{|}~\t\n''', lower=True, split=' ')
tokenizer.fit_on_texts(data)

# Translate each sentence into its word-level IDs, and then one-hot encode those IDs
ID_sequences = tokenizer.texts_to_sequences(data)
binary_sequences = tokenizer.sequences_to_matrix(ID_sequences)

print("ID dictionary:\n", tokenizer.word_index)
print("\nID sequences:\n", ID_sequences)
print("\n One-hot encoded sequences:\n", binary_sequences)