# Maritime Accident Corpus Quality Report

This report summarizes the size, linguistic quality, and validation health of the generated maritime NLP pretraining corpus.

## 1. Corpus Summary Metrics

* **Total Documents (Occurrences)**: 310035
* **Total Words (Tokens)**: 8273295
* **Total Characters**: 52586651
* **Average Document Length**: 26.69 words
* **Median Document Length**: 17.00 words
* **Vocabulary Size**: 43933 unique words
* **Type-Token Ratio (Lexical Diversity)**: 0.00531
* **Shannon Entropy**: 8.3590 bits

### Document Length Distribution
* **0-50 words**: 277440 documents
* **50-100 words**: 20180 documents
* **100-200 words**: 11604 documents
* **200-500 words**: 792 documents
* **500-1000 words**: 19 documents
* **1000+ words**: 0 documents

---

## 2. Redundancy & Deduplication Metrics

* **Duplicate Sentence Ratio**: 35.07%
* **Duplicate Paragraph Ratio**: 0.05%

---

## 3. Top N-Grams

### Top 15 Terms
| Rank | Term | Frequency |
|---|---|---|
| 1 | the | 684559 |
| 2 | of | 255784 |
| 3 | was | 238789 |
| 4 | a | 215375 |
| 5 | vessel | 196725 |
| 6 | and | 164389 |
| 7 | on | 148372 |
| 8 | in | 148128 |
| 9 | with | 117024 |
| 10 | radio | 104647 |
| 11 | compass | 104257 |
| 12 | reported | 101401 |
| 13 | radar | 89212 |
| 14 | gt | 88453 |
| 15 | gross | 88439 |


### Top 10 Bigrams
| Rank | Bigram | Frequency |
|---|---|---|
| 1 | the vessel | 150795 |
| 2 | a gross | 88439 |
| 3 | gross tonnage | 88439 |
| 4 | tonnage of | 88439 |
| 5 | on board | 76332 |
| 6 | constructed of | 71058 |
| 7 | magnetic compass | 67735 |
| 8 | the occurrence | 65586 |
| 9 | vhf radio | 63655 |
| 10 | with a | 59013 |


### Top 10 Trigrams
| Rank | Trigram | Frequency |
|---|---|---|
| 1 | a gross tonnage | 88439 |
| 2 | gross tonnage of | 88439 |
| 3 | at the time | 52066 |
| 4 | the time of | 51985 |
| 5 | time of the | 51975 |
| 6 | with a gross | 47135 |
| 7 | of the occurrence | 42595 |
| 8 | has a gross | 41304 |
| 9 | registered in canada | 40996 |
| 10 | underway moving ahead | 37929 |


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
DAMAGED DOCK AND BULBOUS BOW. Note: formerly OccNo : 9708-26-1 The weather at the time of the occurrence was clear. The sea state was described as ice covered - heavy.
```

### Occurrence ID: 3
```text
During the occurrence, weather was reported as clear. The sea state was described as ice covered - heavy.
```


