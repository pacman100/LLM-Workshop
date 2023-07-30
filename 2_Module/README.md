# Understand the current state of the art LLMs

> "You shall know a word by the company it keeps" - Linguist John Rupert Firth

Natural Language Processing (NLP) domain is currently the centre stage of monumental breakthroughs even though the research has been going on for half a decade now. Now is the best time to get started in the world of Machine Learning (ML)/Deep Learning (DL)/Artifical Intgelligence (AI) as it is akin to **Software 2.0** wherein more and more businesses, instituations and humanity as whole will increasingly use and interact with AI.

**How to represent  and sentences of a language in a way Computer can make sense of it?**

## Word Embeddings

Machine Learning has strong preference for structured fixed-length data. How to model the messiness and unstrutructed traits of textual data? 

### One-hot Encoding

One-hot encoding is a technique used in machine learning and data processing to convert categorical variables into a numerical format. It is commonly employed when dealing with categorical data in various tasks such as classification or neural network inputs, as most machine learning algorithms work with numerical data.

The process of one-hot encoding involves representing each word as a binary vector with all elements set to zero except for the index corresponding to the word, which is set to one. This way, each word becomes a unique vector, and the entire vocabulary of words is transformed into a matrix of binary values.

Let's go through an example to illustrate how one-hot encoding works:

Suppose we have a dataset containing information about the different colours: Alex, Ben, Charlie, Diana, and Eva. The possible favorite colors are "Red," "Blue," "Green," and "Yellow."

Here's the original dataset:

| Person  | Favorite Color |
|---------|----------------|
| Alex    | Blue           |
| Ben     | Green          |
| Charlie | Red            |
| Diana   | Blue           |
| Eva     | Yellow         |

To one-hot encode the "Favorite Color" column, we would follow these steps:

1. Identify the unique categories: In this case, the unique categories are "Red," "Blue," "Green," and "Yellow."

2. Create a binary vector for each category: We represent each category as a binary vector with the length equal to the number of unique categories. Since we have four unique colors, the binary vectors will have a length of four. Each vector will have all elements set to zero except for the element corresponding to the category, which will be set to one.

The binary vectors for each category are as follows:

- "Red" → [1, 0, 0, 0]
- "Blue" → [0, 1, 0, 0]
- "Green" → [0, 0, 1, 0]
- "Yellow" → [0, 0, 0, 1]

3. Replace the categorical values with the binary vectors: Now, we replace the original "Favorite Color" column with these one-hot encoded binary vectors.

The new dataset after one-hot encoding will look like this:

| Person  | Red | Blue | Green | Yellow |
|---------|-----|------|-------|--------|
| Alex    | 0   | 1    | 0     | 0      |
| Ben     | 0   | 0    | 1     | 0      |
| Charlie | 1   | 0    | 0     | 0      |
| Diana   | 0   | 1    | 0     | 0      |
| Eva     | 0   | 0    | 0     | 1      |

Each person's favorite color is now represented by a one-hot encoded binary vector, making it suitable for various machine learning algorithms.

#### Advantages of One-Hot Encoding:

1. **Simple Representation:** One-Hot Encoding is straightforward to implement and understand. It converts categorical data into a binary representation, where each category is represented by a unique binary vector.

2. **Preserves Categorical Information:** One-Hot Encoding preserves the categorical information without introducing any ordinal relationship between categories.

3. **Compatibility with Machine Learning Algorithms:** Many machine learning algorithms, especially those based on numerical calculations, require input data to be in numerical form. One-Hot Encoding is a convenient way to convert categorical data into a format suitable for these algorithms.

4. **Handling Non-Numeric Data:** One-Hot Encoding allows the inclusion of non-numeric data, such as text labels or categorical variables, in machine learning models. This enables the utilization of valuable information from such data in the modeling process.

5. **No Assumptions About Data Distribution:** One-Hot Encoding does not make any assumptions about the distribution of categorical data. It treats each category equally and independently, making it suitable for various data distributions.

6. **Interpretability:** The resulting binary vectors are easily interpretable, making it easier to inspect and understand the relationship between categories and the encoded features.

7. **No Magnitude Impact:** Since each category is represented by a binary vector, there is no magnitude impact on the encoding. All feature values are either 0 or 1, avoiding any magnitude-related biases.

8. **Handling Missing Data:** One-Hot Encoding can handle missing data gracefully. Missing values are simply represented as all 0s in the binary vectors, allowing algorithms to still process and learn from the available information.

9. **Useful for Categorical Features with Low Cardinality:** One-Hot Encoding is particularly useful for categorical features with low cardinality (a small number of unique categories), as it does not introduce significant dimensionality issues in such cases.

#### Limitations of One-Hot Encoding:

1. **High Dimensionality:** One-Hot Encoding creates a binary vector for each word in the dataset. As the number of unique word increases, the dimensionality of the encoded feature space also increases significantly. This can lead to a sparse and high-dimensional representation, making it computationally expensive and memory-intensive.

2. **No Semantic Information:** One-Hot Encoding treats all words as independent and unrelated. It does not capture any semantic similarity or hierarchical relationship between words. As a result, it may not be suitable for tasks that require understanding relationships or similarities between different words.

3. **Curse of Dimensionality:** The high dimensionality of One-Hot Encoding can lead to the curse of dimensionality, where the sparsity of the data negatively impacts the performance of machine learning algorithms, especially for small or limited datasets.



### Bag of words

Instead of a single index being non-zero in One-Hot Encoding, it has multiple non-zero values in the vector representation of a given sequence of words making up a sentence. it is called "Bag of words" due to the lack of preserving the ordering the words in a sentence making it an unordered collection of "bag" of words. It disregards grammar and ordering and is based on the frequency of each word in a given sentence/document. 

The BoW process involves the following steps:

1. **Tokenization:** The text is divided into individual words or tokens. Punctuation and special characters are usually removed, and all words are converted to lowercase to ensure consistency.

2. **Vocabulary Creation:** All unique words across the entire corpus (collection of documents) are collected to form a vocabulary. Each word is assigned a unique index in the vocabulary.

3. **Vectorization:** For each document, a feature vector is created. The length of this vector is equal to the size of the vocabulary, and each entry in the vector represents the frequency of the corresponding word in the document.

4. **Encoding:** The frequency of each word is recorded in the appropriate position of the document's feature vector, according to the word's index in the vocabulary.

Example of Bag of Words

Let's illustrate the Bag of Words technique with a simple example. Consider the following three short documents:

1. Document 1: "The cat chased the mouse."
2. Document 2: "The dog barked at the cat."
3. Document 3: "The mouse ran away from the cat and the dog."

Step 1: Tokenization and Vocabulary Creation

The vocabulary for these three documents will be:

| Index | Word   |
|-------|--------|
| 0     | the    |
| 1     | cat    |
| 2     | chased |
| 3     | mouse  |
| 4     | dog    |
| 5     | barked |
| 6     | at     |
| 7     | ran    |
| 8     | away   |
| 9     | from   |
| 10    | and    |

Step 2: Vectorization

Now, we create the feature vectors for each document based on the word frequencies:

Document 1: [2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
Document 2: [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]
Document 3: [3, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1]

Step 3: Encoding

The vectors show the frequency of each word in the respective documents. For example, in Document 1, the word "the" appears twice, "cat" appears once, "chased" appears once, and so on.

#### Advantages over One-Hot Encoding:
1. **More informative:** In addition to word being present or not, it also encodes the frequency of it providing more information. This can be used for applications like semantic similairty wherein similar documents are likely to have similar word frequencies, which can be used in tasks like document clustering or recommendation systems. 

#### Limitations 
1. **Loss of Word Order:** BoW ignores the word order and the grammatical structure of the text, treating the text as an unordered collection of words. This loss of sequence information can result in the loss of crucial context and meaning, especially in tasks like sentiment analysis or natural language generation.

2. **Fixed Size Representation:** BoW generates a fixed-size vector for each document, where the dimensionality is determined by the size of the vocabulary. This can lead to a loss of information, as long documents may be truncated, and short documents may be padded, leading to unequal representation.

3. **Frequency-based Bias:** BoW relies heavily on the frequency of words in a document. This may cause common words (stopwords) that occur frequently in many documents to dominate the representation, while rare, yet important, words might get overshadowed.

4. **Out-of-Vocabulary Words:** Words that are not present in the vocabulary (words unseen during training) are usually ignored or replaced with a special token. This can result in losing valuable information from the input text, especially for domain-specific or rare terms.

5. **Sparsity:** BoW representation often results in sparse vectors, where most of the entries are zeros. Sparse data can be challenging to handle, especially for certain machine learning algorithms that assume dense feature vectors.


### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a popular technique used in natural language processing and information retrieval to represent the importance of words in a document within a collection of documents. It is an improvement over the Bag of Words (BoW) approach as it considers not only the frequency of words in a document but also their importance in the entire corpus.

#### How TF-IDF Works

The TF-IDF formula consists of two parts: Term Frequency (TF) and Inverse Document Frequency (IDF).

1. **Term Frequency (TF):** It measures the frequency of a word in a document. It is calculated as the ratio of the number of times a word appears in the document to the total number of words in that document. A higher TF value indicates that a word is more important in the document.

   TF = (Number of occurrences of a word in the document) / (Total number of words in the document)

2. **Inverse Document Frequency (IDF):** It measures the rarity of a word across all documents in the corpus. IDF is calculated as the logarithm of the ratio of the total number of documents to the number of documents containing the word. Words that appear in many documents will have a lower IDF value, making them less important.

   IDF = log((Total number of documents) / (Number of documents containing the word))

The TF-IDF score for a word in a document is obtained by multiplying its TF and IDF values.

Let's consider an example of using TF-IDF for retrieving relevant products to a given query in an e-commerce platform.

Suppose we have an e-commerce platform that sells electronic gadgets, and we want to use TF-IDF to retrieve relevant products for a user's search query.

Our product database contains the following three product descriptions:

1. Product 1: "High-quality wireless headphones with noise-canceling technology."
2. Product 2: "Powerful and compact Bluetooth speaker for immersive sound experience."
3. Product 3: "Smartphone with an advanced camera system and long-lasting battery."

Let's say a user enters the query: "wireless headphones sound quality"

Step 1: Preprocessing and Calculating Term Frequencies

We first preprocess the query and product descriptions by tokenizing the text, converting everything to lowercase, and removing any stopwords.

The term frequencies (TF) for the query are as follows:

| Term         | TF(query, Product 1) | TF(query, Product 2) | TF(query, Product 3) |
|--------------|-------------------|--------------|-------------------|
| wireless     | 1                 | 0                 | 0                 |
| headphones   | 1                 | 0                 | 0                 |
| sound        | 0                 | 1                 | 0                 |
| quality      | 1                 | 0                 | 0                 |

Step 2: Calculating Inverse Document Frequencies (IDF)

Next, we calculate the inverse document frequency (IDF) for each term in the query. In this example, since we only have three products, the IDF values for all terms will be the same.

| Term         | IDF("term")      |
|--------------|------------------|
| wireless     | log(3/1) ≈ 0.585 |
| headphones   | log(3/1) ≈ 0.585 |
| sound        | log(3/1) ≈ 0.585 |
| quality      | log(3/1) ≈ 0.585 |

Step 3: Calculate TF-IDF Scores

We calculate the TF-IDF scores for each product based on the query:

| Product      | TF-IDF(query, Product) |
|--------------|----------------------|
| Product 1    | 1 * 0.585 + 1 * 0.585 + 0 * 0.585 + 1 * 0.585 = 1.755 |
| Product 2    | 0 * 0.585 + 0 * 0.585 + 1 * 0.585 + 0 * 0.585 = 0.585 |
| Product 3    | 0 * 0.585 + 0 * 0.585 + 0 * 0.585 + 0 * 0.585 = 0 |

Step 4: Ranking the Products

Based on the TF-IDF scores, we can rank the products in descending order of relevance to the user's query:

1. Product 1: "High-quality wireless headphones with noise-canceling technology." (TF-IDF score: 1.755)
2. Product 2: "Powerful and compact Bluetooth speaker for immersive sound experience." (TF-IDF score: 0.585)
3. Product 3: "Smartphone with an advanced camera system and long-lasting battery." (TF-IDF score: 0)

This example demonstrates how TF-IDF can be used to retrieve relevant products in an e-commerce platform based on a user's search query. The TF-IDF scores help in ranking the products in order of their relevance, making it easier for users to find the most suitable products for their needs.

#### Advantages over Bag of Words:

1. **Term Importance:** TF-IDF considers not only the term frequency in a document (TF) but also the rarity of the term across the entire document collection (IDF). This means that common words that appear frequently in many documents will have a lower TF-IDF score, reducing their impact on the representation. Rare and important words that appear in specific documents will have higher TF-IDF scores, making them more influential in the analysis. This naturally handles stopwords (common words like "the," "and," "is," etc.), which typically have little semantic value

2. **Document Relevance**: TF-IDF helps in identifying relevant documents for a given query. By focusing on important terms and discounting common words, TF-IDF gives higher scores to documents that contain rare and relevant words related to the query. This makes them great for Semantic Similarity. Specifically useful for straighforward queries such as books/novels wherein customers directly just type the exact name of books such as "Harry Potter and the Goblet of Fire" etc. Here, syntactic match is enough.  

#### Limitations

1. **Absence of Semantic Understanding:** TF-IDF treats each term independently and does not capture semantic relationships between words. It cannot understand the meaning or context of words, making it less suitable for tasks that require a deeper understanding of language semantics. For example, it fails to capture that "Geyser" and "Water Heater" mean the same thing. It also fails to capture that "apple" has different meaning in "She ate a juicy red apple." (Fruit) when compared to "I bought a new Apple laptop." (Brand of technology products), i.e., failing to capture contextual representation of words.

2. **Scalability:** For very large document collections, computing the TF-IDF representation for each document can become computationally expensive and memory-intensive. Additionally, maintaining a large vocabulary might also pose challenges.

### Word2vec

To address the aforementioned limitations, word embeddings offer an efficient and compact(dense) way of representing words, ensuring that words with similar meanings have similar encodings. Word Embedding is a vector containing floating point values and hence the term "Dense Representation". The values for the embedding are trainable parameters which are learned similarly
to a model learning the weights for a dense layer. The dimensionality of the word representations is typically much smaller than the number of words in the dictionary. 

One of the foundational work in this regard is the `word2vec`. It proposes 2 methods to learn word embeddings:
1. Continuous Bag of Words: Predicts the current word based on the context 
2. Continuous Skip-Gram: Predicts surrounding words given the current word

![word2vec](../assets/word2vec.png)
[Source [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)]

Below is an example for Skip-Gram model:

Let's consider a small corpus with the following sentence:

"Learning deep learning is essential for artificial intelligence."

Step 1: Prepare Training Data

To train the skip-gram model, we first create training pairs of target and context words. We define a context window size (e.g., 2), which determines how many words on either side of the target word will be considered as context words.

For the sentence above, with a context window size of 2, the training pairs would be:

- (Learning, deep)
- (Learning, learning)
- (deep, Learning)
- (deep, learning)
- (deep, is)
- (learning, Learning)
- (learning, deep)
- (learning, is)
- (learning, essential)
- (is, deep)
- (is, learning)
- (is, essential)
- (essential, learning)
- (essential, is)
- (artificial, is)
- (artificial, essential)
- (intelligence, essential)
- (intelligence, artificial)

Step 2: Train the Skip-Gram Model

The skip-gram model takes these training pairs as input and tries to learn word embeddings by optimizing its parameters to predict the context words given the target word. The goal is to maximize the probability of the context words given the target word.

Step 3: Word Embeddings

Once the skip-gram model is trained, it will generate word embeddings for each word in the vocabulary. These embeddings are dense and lower-dimensional representations of words, capturing the semantic relationships between words based on their co-occurrence patterns in the training data.

The resulting word embeddings might look like:

| Word       | Embedding Vector            |
|------------|----------------------------|
| Learning   | [0.2, 0.5, -0.1, ...]       |
| deep       | [-0.3, 0.7, 0.4, ...]       |
| learning   | [0.1, -0.6, 0.8, ...]       |
| is         | [0.5, 0.3, -0.2, ...]       |
| essential  | [-0.4, 0.2, 0.6, ...]       |
| artificial | [0.7, 0.1, -0.5, ...]       |
| intelligence | [-0.2, -0.3, 0.9, ...]    |
| ...        | ...                        |

Step 4: Word Similarity

The learned word embeddings capture semantic similarities between words. Words with similar meanings or contexts will have similar embeddings. For example, the embeddings for "Learning" and "learning" are likely to be close to each other since they appear in similar contexts in the training data.

With these word embeddings, we can perform various NLP tasks, such as word semantic similarity, document classification, and even text generation, by leveraging the semantic relationships captured in the dense word embeddings.

Word embeddings have interesting properties in terms of geometrical relationships. For example, in the below image we see that linear relationships between semantically similar embeddings, i.e., `vector(Japan)−vector(Tokyo)+vector(Russia) ~ vector(Moscow)`

![linear-relationships](../assets/linear-relationships.svg)

[Source [Embeddings: Translating to a Lower-Dimensional Space](https://developers.google.com/machine-learning/crash-course/embeddings/translating-to-a-lower-dimensional-space)]

#### Limitations:

1. **Absence of Context Understanding:** Since, the word embeddings are static, it lacks the ability to model semantic relationship between words based on context. Again, it fails to capture that "apple" has different meaning in "She ate a juicy red apple." (Fruit) when compared to "I bought a new Apple laptop." (Brand of technology products).

## Encoder-Decoder Models

Let deep dive into the problem of Machine Translation. i.e., translating text from one language to another, e.g., translating Hindi to English. As the input length can be differnt than the ouput length, it demands for a architecture which can handle this. The encoder-decoder architecture is devised for this purpose. Encoder converts the input sentence into fixed representation and decoder takes this fixed representation and converts it into target sentence sequentially. Below is a diagram showing the process.

![encoder-decoder](../assets/encoder-decoder.png)
[Source [cs224n 2022](https://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture05-rnnlm.pdf)]

To encode and decode, we need to capture the semantic and contextual representation of words. One way to capture semantic and contextual relationship is to condition the prediction of next word based on all the words generated so far. This results in a recurrence relationship which is nicely captured by Recurrent neural Networks (RNNs).

### Recurrrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a type of neural network designed to process sequential data, such as time series or natural language. They have feedback connections that allow them to maintain a hidden state, enabling them to retain information about past inputs. This hidden state serves as a memory, allowing RNNs to consider the context of each input in relation to previous inputs. RNNs are particularly effective for tasks involving sequential patterns, like language modeling, machine translation, and speech recognition. 

![RNN](../assets/rnn.png)
[Source [cs224n 2022](https://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture05-rnnlm.pdf)]

#### Limitations:

1. **Exploding or Vanishing Gradients:** They suffer from vanishing or exploding gradient problems, which can make it challenging for them to capture long-range dependencies effectively. It is also due to Backpropagation through time.
2. **Recurrent computation is slow:** Doesn't allow for parallelization of operations leading to suboptimal usage of hardware resources.
3. **Emprically, RNNs underperform to handle long-term dependencies**

### Long Short-Term Memory Models (LSTMs)

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

Instead of just the hidden state, they have an additional state called cell state which makes it easier for information to flow through unchanged, thereby allowing to handle long-term dependencies. It has a "forget gate layer", "input gate layer" and "output gate layer" controlling what information to forget and update and what to ouput as part of next hidden state. 
Below is a diagram of the LSTM architecture:

![LSTM](../assets/LSTM3-chain.png)
[Source [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)]

![LSTM layers](../assets/LSTM_layers.png)
[Source [cs224n 2022](https://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture06-fancy-rnn.pdf)]

#### Limitations of Encoder-Decoder models we have seen so far

Encoder-Decoder models help in Many-to-Many foramt problems but have their own limitations. 
1. **Information Bottleneck Problem:** Compressing whole information of a given sentence into fixed-size vector often leads in loss of valuable information. 
2. **Vanishing Gradients:** Backpropagation through time leads to disappearance of information as it backpropagates. This is because gradients with less than 1 at each steps leading to smaller and smaler and smaller gradients vanishing to 0.

### LSTMs with Attention

Now, let's take inspiration from Human's attention span. **Our brain focuses on a limited part of the whole information. Attention is a mechanism that enables us to selectively focus on a specific part of the whole information as required by the task.** Can this be extended to Deep Learning models? Yes.

Attention techniques allows models to retain complete information from the source sentence without compression. Thereby, **solving the information bottleneck**. It enables the Decoder to selectively focus on particular parts of the source sentence while sequentially generating the output, thereby **being more "human-like" and "somewhat explainable"**. it also **solves the vanishing gradient problem** by providing shorcut for gradients to flow to faraway state. Below is a diagram showing the architecture of using LSTMs with Attention.

![LSTM with attention](../assets/LSTM_with_attention.png)

## Transformer

### Attention is all you Need

### Bi-Directional Encoder Representation From Transformers (BERT)


### (Generative Pre-Trained Models) GPT-1


### GPT-2


### GPT-3


### GPT-3.5 Instruct


### GPT-4 RLHF


### T5

## Types of LLMs

### Continuing the Text

### Instruction Aligned

### LIMA


## References









