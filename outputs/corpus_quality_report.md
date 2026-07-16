# Maritime Accident Corpus Quality Report

This report summarizes the size, linguistic quality, and validation health of the generated maritime NLP pretraining corpus.

## 1. Corpus Summary Metrics

* **Total Documents (Occurrences)**: 48441
* **Total Words (Tokens)**: 3315982
* **Total Characters**: 21023833
* **Average Document Length**: 68.45 words
* **Median Document Length**: 61.00 words
* **Vocabulary Size**: 43905 unique words
* **Type-Token Ratio (Lexical Diversity)**: 0.01324
* **Shannon Entropy**: 8.5288 bits

### Document Length Distribution
* **0-50 words**: 19486 documents
* **50-100 words**: 18281 documents
* **100-200 words**: 10194 documents
* **200-500 words**: 478 documents
* **500-1000 words**: 2 documents
* **1000+ words**: 0 documents

---

## 2. Redundancy & Deduplication Metrics

* **Duplicate Sentence Ratio**: 52.30%
* **Duplicate Paragraph Ratio**: 31.21%

---

## 3. Top N-Grams

### Top 15 Terms
| Rank | Term | Frequency |
|---|---|---|
| 1 | the | 266462 |
| 2 | of | 93350 |
| 3 | in | 77150 |
| 4 | a | 74120 |
| 5 | was | 72083 |
| 6 | vessel | 70537 |
| 7 | reported | 56818 |
| 8 | to | 50255 |
| 9 | with | 47566 |
| 10 | as | 45595 |
| 11 | on | 43871 |
| 12 | and | 43417 |
| 13 | were | 42693 |
| 14 | conditions | 41376 |
| 15 | weather | 32348 |


### Top 10 Bigrams
| Rank | Bigram | Frequency |
|---|---|---|
| 1 | the vessel | 52444 |
| 2 | the incident | 30775 |
| 3 | the occurrence | 28358 |
| 4 | is a | 27864 |
| 5 | registered in | 26810 |
| 6 | at the | 26418 |
| 7 | of the | 26147 |
| 8 | were reported | 26059 |
| 9 | with a | 25388 |
| 10 | constructed of | 23961 |


### Top 10 Trigrams
| Rank | Trigram | Frequency |
|---|---|---|
| 1 | at the time | 23290 |
| 2 | with a gross | 23211 |
| 3 | a gross tonnage | 23211 |
| 4 | gross tonnage of | 23211 |
| 5 | the time of | 23209 |
| 6 | time of the | 23199 |
| 7 | registered in canada | 19742 |
| 8 | note formerly occno | 19209 |
| 9 | the incident occurred | 17477 |
| 10 | in canada constructed | 17252 |


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

### Occurrence ID: 2
```text
INJURED DURING MOORING OPERATIONS. Note: formerly OccNo : 9704-7
```

### Occurrence ID: 3
```text
DAMAGED DOCK AND BULBOUS BOW. Note: formerly OccNo : 9708-26-1

The weather at the time of the occurrence was clear. The sea state was described as ice covered - heavy.
```


