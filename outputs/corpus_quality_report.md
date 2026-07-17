# Maritime Accident Corpus Quality Report

This report summarizes the size, linguistic quality, and validation health of the generated maritime NLP pretraining corpus.

## 1. Corpus Summary Metrics

* **Total Documents (Occurrences)**: 74548
* **Total Words (Tokens)**: 3143904
* **Total Characters**: 20081181
* **Average Document Length**: 42.17 words
* **Median Document Length**: 35.00 words
* **Vocabulary Size**: 42853 unique words
* **Type-Token Ratio (Lexical Diversity)**: 0.01363
* **Shannon Entropy**: 8.3799 bits

### Document Length Distribution
* **0-50 words**: 51868 documents
* **50-100 words**: 17875 documents
* **100-200 words**: 4739 documents
* **200-500 words**: 64 documents
* **500-1000 words**: 2 documents
* **1000+ words**: 0 documents

---

## 2. Redundancy & Deduplication Metrics

* **Duplicate Sentence Ratio**: 51.32%
* **Duplicate Paragraph Ratio**: 0.01%

---

## 3. Top N-Grams

### Top 15 Terms
| Rank | Term | Frequency |
|---|---|---|
| 1 | the | 249506 |
| 2 | vessel | 99258 |
| 3 | of | 98831 |
| 4 | a | 79935 |
| 5 | in | 66073 |
| 6 | was | 63290 |
| 7 | with | 59674 |
| 8 | to | 50488 |
| 9 | on | 49339 |
| 10 | conditions | 47631 |
| 11 | at | 39488 |
| 12 | reported | 38849 |
| 13 | and | 36977 |
| 14 | occurrence | 35398 |
| 15 | time | 33722 |


### Top 10 Bigrams
| Rank | Bigram | Frequency |
|---|---|---|
| 1 | the vessel | 65510 |
| 2 | vessel was | 36658 |
| 3 | of the | 36215 |
| 4 | the occurrence | 35310 |
| 5 | at the | 34339 |
| 6 | the time | 33453 |
| 7 | time of | 33367 |
| 8 | is a | 26532 |
| 9 | registered in | 25713 |
| 10 | with a | 24789 |


### Top 10 Trigrams
| Rank | Trigram | Frequency |
|---|---|---|
| 1 | the vessel was | 36033 |
| 2 | at the time | 33441 |
| 3 | the time of | 33360 |
| 4 | time of the | 33351 |
| 5 | of the occurrence | 28533 |
| 6 | the occurrence the | 24610 |
| 7 | occurrence the vessel | 24603 |
| 8 | with a gross | 22615 |
| 9 | a gross tonnage | 22615 |
| 10 | gross tonnage of | 22615 |


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


