---
title: MBAI
separator: <!--s-->
verticalSeparator: <!--v-->
theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Data Intensive Systems
  ## L.05 | OLAP + EDA III

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 10%">

  <iframe src="https://lottie.host/embed/216f7dd1-8085-4fd6-8511-8538a27cfb4a/PghzHsvgN5.lottie" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Welcome to Data Intensive Systems.
  ## Please check in by creating an account and entering the code on the chalkboard.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

## Announcements

- 

<!--s-->

<div class="header-slide">

# OLAP + EDA III
## Modern text data mining (NLP)

</div>

<!--s-->

## Agenda

- Regular Expressions
- Byte-Pair Tokenization
- Word Embeddings
    - Traditional (word2vec)
    - Modern (LLMs)

<!--s-->

<div class="header-slide">

# Regular Expressions

</div>

<!--s-->

## Regular Expressions

Regular expressions (regex) are a powerful tool for working with text data. They allow us to search, match, and manipulate text using a concise and expressive syntax.

You may feel compelled to do basic sting manipulation with Python's built-in string methods. However, regular expressions are much more powerful and flexible. Consider the following example:

> "My phone number is (810) 555-1234."

<!--s-->

## Regular Expressions | Example

> "My phone number is (810)555-1234"

### String Methods

<span class="code-span">phone_number = text.split(" ")[-1]</span>

This method would work for the given example but **not** for "My phone number is (810) 555-1234. Call me!"
or "My phone number is (810) 555-1234. Call me! It's urgent!"

### Regular Expression

<span class="code-span">phone_number = re.search(r'\(\d{3}\)\d{3}-\d{4}', text).group()</span>

This regular expression will match any phone number in the format (810)555-1234, including the additional text above.

<!--s-->

## Regular Expressions | Syntax

Regular expressions are a sequence of characters that define a search pattern. They are used to search, match, and manipulate text strings.

<div style="font-size: 0.8em; overflow-y: scroll; height: 80%;">
  
| Pattern | Description |
|---------|-------------|
| <span class='code-span'>.</span>     | Matches any character except newline |
| <span class='code-span'>^</span>     | Matches the start of a string |
| <span class='code-span'>$</span>     | Matches the end of a string |
| <span class='code-span'>*</span>     | Matches 0 or more repetitions of the preceding element |
| <span class='code-span'>+</span>     | Matches 1 or more repetitions of the preceding element |
| <span class='code-span'>?</span>     | Matches 0 or 1 repetition of the preceding element |
| <span class='code-span'>{n}</span>   | Matches exactly n repetitions of the preceding element |
| <span class='code-span'>{n,}</span>  | Matches n or more repetitions of the preceding element |
| <span class='code-span'>{n,m}</span> | Matches between n and m repetitions of the preceding element |
| <span class='code-span'>[]</span>    | Matches any one of the characters inside the brackets |
| <span class='code-span'> \| </span>     | Matches either the expression before or the expression after the operator |
| <span class='code-span'>()</span>    | Groups expressions and remembers the matched text |
| <span class='code-span'>\d</span>    | Matches any digit (equivalent to <span class='code-span'>[0-9]</span>) |
| <span class='code-span'>\D</span>    | Matches any non-digit character |
| <span class='code-span'>\w</span>    | Matches any word character (equivalent to <span class='code-span'>[a-zA-Z0-9_]</span>) |
| <span class='code-span'>\W</span>    | Matches any non-word character |
| <span class='code-span'>\s</span>    | Matches any whitespace character (spaces, tabs, line breaks) |
| <span class='code-span'>\S</span>    | Matches any non-whitespace character |
| <span class='code-span'>\b</span>    | Matches a word boundary |
| <span class='code-span'>\B</span>    | Matches a non-word boundary |
| <span class='code-span'>\\</span>    | Escapes a special character |

</div>
<!--s-->

## Regular Expressions

Want to practice or make sure your expression works? 

Live regular expression practice: https://regex101.com/

<!--s-->

## Snowflake

Here is an example of searching for phone numbers in a column of a Snowflake table.

```sql

SELECT *
FROM my_table
WHERE REGEXP_LIKE(phone_number, '\\(\\d{3}\\)\\d{3}-\\d{4}');
```

<!--s-->

<div class="header-slide">

# Tokenization

</div>

<!--s-->

## Tokenize

Tokenization is the process of breaking text into smaller units called tokens. Tokens can be words, subwords, or characters. Tokenization is a crucial step in NLP because it allows us to work with text data in a structured way.

Some traditional tokenization strategies include:

- **Word Tokenization**. E.g. <span class="code-span">"Hello, world!" -> ["Hello", ",", "world", "!"] -> [12, 4, 56, 3]</span>
- **Subword Tokenization**. E.g. <span class="code-span">"unbelievable" -> ["un", "believable"] -> [34, 56]</span>
- **Character Tokenization**. E.g. <span class="code-span">"Hello!" -> ["H", "e", "l", "l", "o", "!"] -> [92, 34, 56, 56, 12, 4]</span>

These simple tokenization strategies are often not sufficient for modern NLP tasks. For example, word tokenization can lead to a large vocabulary size, which can be computationally expensive. Subword tokenization can help reduce the vocabulary size, but it can still lead to out-of-vocabulary words. Character tokenization can handle out-of-vocabulary words, but it can lead to a loss of semantic meaning. One common, modern tokenization strategy is **Byte Pair Encoding(BPE)**, which is used by many large language models.

<!--s-->

## Tokenize | Byte Pair Encoding (BPE)

BPE is a subword tokenization algorithm that builds a vocabulary of subwords by iteratively merging the most frequent pairs of characters.

BPE is a powerful tokenization algorithm because it can handle rare words and out-of-vocabulary words. It is used by many large language models, including GPT-4. The algorithm is as follows:

```text
1. Initialize the vocabulary with all characters in the text.
2. While the vocabulary size is less than the desired size:
    a. Compute the frequency of all character pairs.
    b. Merge the most frequent pair.
    c. Update the vocabulary with the merged pair.
```

<!--s-->

## Tokenize | Byte Pair Encoding (BPE) with TikToken

One BPE implementation can be found in the `tiktoken` library, which is an open-source library from OpenAI.

```python

import tiktoken
enc = tiktoken.get_encoding("cl100k_base") # Get specific encoding used by GPT-4.
enc.encode("Hello, world!") # Returns the tokenized text.

>> [9906, 11, 1917, 0]

```

<!--s-->

## Chunking

<div style = "font-size: 0.8em;">

Chunking is the process of creating windows of text that can be indexed and searched. Chunking is essential for information retrieval systems because it allows us to break down large documents into smaller, searchable units. It differs from tokenization in that it is not concerned with the individual tokens, but rather with the larger units of text.

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; height: 100%;">


### Sentence Chunking

Sentence chunking is the process of breaking text into sentences.

E.g. <span class="code-span">"Hello, world! How are you?" -> ["Hello, world!", "How are you?"]</span>

### Paragraph Chunking

Paragraph chunking is the process of breaking text into paragraphs.

E.g. <span class="code-span">"Hello, world! \n Nice to meet you." -> ["Hello, world!", "Nice to meet you."]</span>

### Agent Chunking

Agent chunking is the process of breaking text down using an LLM.

</div>

<div class="c2" style = "width: 50%; height: 100%;">

### Sliding Word / Token Window Chunking

Sliding window chunking is a simple chunking strategy that creates windows of text by sliding a window of a fixed size over the text.

E.g. <span class="code-span">"The cat in the hat" -> ["The cat in", "cat in the", "in the hat"]</span>

### Semantic Chunking

Semantic chunking is the process of breaking text into semantically meaningful units.

E.g. <span class="code-span">"The cat in the hat. One of my favorite books." -> ["The cat in the hat.", "One of my favorite books."]</span>

</div>
</div>

<!--s-->

## Chunk | NLTK Sentence Chunking

NLTK is a powerful library for natural language processing that provides many tools for text processing. NLTK provides a sentence tokenizer that can be used to chunk text into sentences.

### Chunking with NLTK

```python
from nltk import sent_tokenize

# Load Academic Integrity document.
doc = open('/Users/joshua/Desktop/academic_integrity.md').read()

# Split the document into sentences.
chunked_data = sent_tokenize(doc)
```

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; height: 100%;">

### Input: Original Text

```text
The purpose of this guide is to set forth the terms under which academic work is pursued at Northwestern and
throughout the larger intellectual community of which we are members. Please read this booklet carefully,
as you will be held responsible for its contents. It describes the ways in which common sense and decency apply
to academic conduct. When you applied to Northwestern, you agreed to abide by our principles of academic integrity;
these are spelled out on the first three pages. The balance of the booklet provides information that will help you avoid
violations, describes procedures followed in cases of alleged violations of the guidelines, and identifies people who 
can give you further information and counseling within the undergraduate schools.
```

</div>
<div class="c2" style = "width: 50%; height: 100%;">

### Output: Chunked Text (by Sentence)
```text
[
    "The purpose of this guide is to set forth the terms under which academic work is pursued at Northwestern and throughout the larger intellectual community of which we are members."
    "Please read this booklet carefully, as you will be held responsible for its contents."
    "It describes the ways in which common sense and decency apply to academic conduct."
    "When you applied to Northwestern, you agreed to abide by our principles of academic integrity; these are spelled out on the first three pages."
    "The balance of the booklet provides information that will help you avoid violations, describes procedures followed in cases of alleged violations of the guidelines, and identifies people who can give you further information and counseling within the undergraduate schools."
]

```
</div>
</div>

<!--s-->

<div class="header-slide">

# Word Embeddings

</div>

<!--s-->

## Embed

Word embeddings are dense vector representations of words that capture semantic information. Word embeddings are essential for many NLP tasks because they allow us to work with words in a continuous and meaningful vector space.

**Traditional embeddings** such as Word2Vec are static and pre-trained on large text corpora.

**Contextual embeddings** such as those used by BERT and GPT are dynamic and trained on large language modeling tasks.

<img src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*SYiW1MUZul1NvL1kc1RxwQ.png" style="margin: 0 auto; display: block; width: 80%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Google</span>

<!--s-->

## Embed | Traditional Word Embeddings

Word2Vec is a traditional word embedding model that learns word vectors by predicting the context of a word. Word2Vec has two standard architectures:

- **Continuous Bag of Words (CBOW)**. Predicts a word given its context.
- **Skip-gram**. Predicts the context given a word.

Word2Vec is trained on large text corpora and produces dense word vectors that capture semantic information. The result of Word2Vec is a mapping from words to vectors, where similar words are close together in the vector space.

<img src="https://storage.googleapis.com/slide_assets/word2vec.png" style="margin: 0 auto; display: block; width: 50%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Braun 2017</span>

<!--s-->

## Embed | Contextual Word Embeddings

Contextual word embeddings are word embeddings that are dependent on the context in which the word appears. Contextual word embeddings are essential for many NLP tasks because they capture the *contextual* meaning of words in a sentence.

For example, the word "bank" can have different meanings depending on the context:

- **"I went to the bank to deposit my paycheck."**
- **"The river bank was covered in mud."**

[HuggingFace](https://huggingface.co/spaces/mteb/leaderboard) contains a [MTEB](https://arxiv.org/abs/2210.07316) leaderboard for some of the most popular contextual word embeddings:

<img src="https://storage.googleapis.com/cs326-bucket/lecture_14/leaderboard.png" style="margin: 0 auto; display: block; width: 50%;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">HuggingFace, 2024</span>

<!--s-->

## OLAP | Embed

Similar to the previous examples, we can use Snowflake to embed text data.

```sql

-- Create embedding vectors for wiki articles (only do once)
ALTER TABLE wiki ADD COLUMN vec VECTOR(FLOAT, 768);
UPDATE wiki SET vec = SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', content);

```

<!--s-->

## OLAP | Embed

We can even create RAG pipelines in Snowflake.

```sql

-- Create embedding vectors for wiki articles (only do once)
ALTER TABLE wiki ADD COLUMN vec VECTOR(FLOAT, 768);
UPDATE wiki SET vec = SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', content);

-- Embed incoming query
SET query = 'in which year was Snowflake Computing founded?';
CREATE OR REPLACE TABLE query_table (query_vec VECTOR(FLOAT, 768));
INSERT INTO query_table SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', $query);

-- Do a semantic search to find the relevant wiki for the query
WITH result AS (
    SELECT
        w.content,
        $query AS query_text,
        VECTOR_COSINE_SIMILARITY(w.vec, q.query_vec) AS similarity
    FROM wiki w, query_table q
    ORDER BY similarity DESC
    LIMIT 1
)

-- Pass to large language model as context
SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-7b',
    CONCAT('Answer this question: ', query_text, ' using this text: ', content)) FROM result;

```
<!--s-->