# Maritime Accident Corpus Quality Report

This report summarizes the size, linguistic quality, and validation health of the generated maritime NLP pretraining corpus.

## 1. Corpus Summary Metrics

* **Total Documents (Occurrences)**: 73316
* **Total Words (Tokens)**: 2555505
* **Total Characters**: 16218036
* **Average Document Length**: 34.86 words
* **Median Document Length**: 31.00 words
* **Vocabulary Size**: 42702 unique words
* **Type-Token Ratio (Lexical Diversity)**: 0.01671
* **Shannon Entropy**: 8.7457 bits

### Document Length Distribution
* **0-50 words**: 58189 documents
* **50-100 words**: 14229 documents
* **100-200 words**: 872 documents
* **200-500 words**: 24 documents
* **500-1000 words**: 2 documents
* **1000+ words**: 0 documents

---

## 2. Redundancy & Deduplication Metrics

* **Duplicate Sentence Ratio**: 29.77%
* **Duplicate Paragraph Ratio**: 0.01%

---

## 3. Top N-Grams

### Top 15 Terms
| Rank | Term | Frequency |
|---|---|---|
| 1 | the | 131430 |
| 2 | in | 99092 |
| 3 | and | 55021 |
| 4 | on | 51663 |
| 5 | a | 50597 |
| 6 | of | 48865 |
| 7 | was | 46582 |
| 8 | while | 45843 |
| 9 | underway | 44426 |
| 10 | to | 43952 |
| 11 | vessel | 43492 |
| 12 | with | 41137 |
| 13 | reported | 35504 |
| 14 | an | 28826 |
| 15 | fishing | 28614 |


### Top 10 Bigrams
| Rank | Bigram | Frequency |
|---|---|---|
| 1 | involved in | 26411 |
| 2 | was involved | 26268 |
| 3 | in an | 26258 |
| 4 | an occurrence | 26206 |
| 5 | the vessel | 25939 |
| 6 | registered in | 25270 |
| 7 | while underway | 24923 |
| 8 | with a | 24463 |
| 9 | a gross | 22289 |
| 10 | gross tonnage | 22289 |


### Top 10 Trigrams
| Rank | Trigram | Frequency |
|---|---|---|
| 1 | was involved in | 26267 |
| 2 | involved in an | 26208 |
| 3 | in an occurrence | 26204 |
| 4 | with a gross | 22289 |
| 5 | a gross tonnage | 22289 |
| 6 | gross tonnage of | 22289 |
| 7 | gt registered in | 22260 |
| 8 | while underway underway | 19412 |
| 9 | registered in canada | 18481 |
| 10 | in canada was | 18481 |


---

## 4. Data Validation and Join Integrity

* **Validation Status**: **WARNING**
* **Value Consistency Warnings**: 2

### Raw Table Joining Orphan Statistics
| Relationship | Count |
|---|---|
| Vessels Without Occurrence | 0 |
| Injuries Without Occurrence | 0 |
| Injuries Without Vessel | 2 |
| Lsa Without Occurrence | 0 |
| Lsa Without Vessel | 0 |
| Nav Without Occurrence | 0 |
| Nav Without Vessel | 0 |
| Rec Without Occurrence | 0 |
| Rec Without Vessel | 0 |


---

## 5. Sample Generated Documents

### Occurrence ID: 1
```text
FELL OVERBOARD AND DROWNED WHILE ON DUTY. Note: formerly OccNo : 9704-7
```

### Occurrence ID: 3
```text
DAMAGED DOCK AND BULBOUS BOW. Note: formerly OccNo : 9708-26-1
```

### Occurrence ID: 3
```text
The weather at the time of the occurrence was clear. The sea state was described as ice covered - heavy.
```


