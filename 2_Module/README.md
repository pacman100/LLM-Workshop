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

#### Limitations of One-Hot Encoding and Bag of Words 


### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a popular technique used in natural language processing and information retrieval to represent the importance of words in a document within a collection of documents. It is an improvement over the Bag of Words (BoW) approach as it considers not only the frequency of words in a document but also their importance in the entire corpus.

#### How TF-IDF Works

The TF-IDF formula consists of two parts: Term Frequency (TF) and Inverse Document Frequency (IDF).

1. **Term Frequency (TF):** It measures the frequency of a word in a document. It is calculated as the ratio of the number of times a word appears in the document to the total number of words in that document. A higher TF value indicates that a word is more important in the document.

   TF = (Number of occurrences of a word in the document) / (Total number of words in the document)

2. **Inverse Document Frequency (IDF):** It measures the rarity of a word across all documents in the corpus. IDF is calculated as the logarithm of the ratio of the total number of documents to the number of documents containing the word. Words that appear in many documents will have a lower IDF value, making them less important.

   IDF = log((Total number of documents) / (Number of documents containing the word))

The TF-IDF score for a word in a document is obtained by multiplying its TF and IDF values.

#### Example of TF-IDF

Let's illustrate TF-IDF with a small corpus containing three documents:

1. Document 1: "The cat chased the mouse."
2. Document 2: "The dog barked at the cat."
3. Document 3: "The mouse ran away from the cat and the dog."

Step 1: Calculate TF

For example, in Document 1:

- TF("the") = 2 / 5 = 0.4
- TF("cat") = 1 / 5 = 0.2
- TF("chased") = 1 / 5 = 0.2
- TF("mouse") = 1 / 5 = 0.2
- and so on for all words.

Step 2: Calculate IDF

For example, in the entire corpus:

- IDF("the") = log(3 / 3) = 0
- IDF("cat") = log(3 / 2) ≈ 0.176
- IDF("chased") = log(3 / 1) ≈ 1.099
- IDF("mouse") = log(3 / 2) ≈ 0.176
- and so on for all unique words.

Step 3: Calculate TF-IDF

Finally, we calculate the TF-IDF score for each word in each document:

For example, in Document 1:

- TF-IDF("the") ≈ 0.4 * 0 ≈ 0
- TF-IDF("cat") ≈ 0.2 * 0.176 ≈ 0.035
- TF-IDF("chased") ≈ 0.2 * 1.099 ≈ 0.22
- TF-IDF("mouse") ≈ 0.2 * 0.176 ≈ 0.035
- and so on for all words.

### BM25 Algorithm

BM25 (Best Matching 25) is a ranking function used in information retrieval and search engines to evaluate the relevance of a document to a given query. It is an extension of the TF-IDF (Term Frequency-Inverse Document Frequency) model and is designed to address some of its limitations.

#### How BM25 Works

BM25 is based on probabilistic information retrieval and takes into account the term frequency, document length, and term frequency saturation. The formula for BM25 is as follows:

```latex
BM25(q, D) = \sum \left( \frac{{tf(t, D) \cdot (k + 1)}}{{tf(t, D) + k \cdot (1 - b + b \cdot (\frac{{\vert D \vert}}{{avgdl}}))}} \right) \cdot IDF(t, D)
```

Where:
- `q`: Represents the query terms.
- `D`: Denotes the document being scored.
- `tf(t, D)`: Term frequency of term `t` in document `D`.
- `IDF(t, D)`: Inverse Document Frequency of term `t` in the entire collection of documents.
- `|D|`: Length (number of words) of document `D`.
- `avgdl`: Average document length in the entire collection.
- `k` and `b`: Tuning parameters that control the impact of term frequency and document length normalization, respectively.

#### Example of BM25

Let's consider a simple example with a small document collection and a query.

Suppose we have the following three documents:

1. Document 1: "The cat chased the mouse."
2. Document 2: "The dog barked at the cat."
3. Document 3: "The mouse ran away from the cat and the dog."

Let's say our query is: "cat mouse"

Step 1: Calculate Term Frequencies and IDF

First, we calculate the term frequencies and inverse document frequencies for each term in the query and the documents.

| Term   | TF("term", D1) | TF("term", D2) | TF("term", D3) | IDF("term") |
|--------|---------------|---------------|---------------|------------|
| cat    | 1             | 1             | 1             | log(3/2)   |
| mouse  | 1             | 0             | 1             | log(3/2)   |

Step 2: Calculate BM25 Scores

Next, we calculate the BM25 scores for each document with respect to the query.

| Document | BM25(q, D)  |
|----------|-------------|
| D1       | 0.0 + 0.0   |
| D2       | 0.0 + 0.0   |
| D3       | 0.0 + 0.0   |

As this is a very simple example with only two terms in the query, the BM25 scores for all documents are zero.

### Word2vec

## Encoder-Decoder Models

### RNNs

### LSTMs

### LSTMs with Attention

## Transformer

### Attention is all you Need

### BERT


### GPT-1


### GPT-2


### GPT-3


### GPT-3.5 Instruct


### GPT-4 RLHF


### T5

## Types of LLMs

### Continuing the Text

### Instruction Aligned

### LIMA








