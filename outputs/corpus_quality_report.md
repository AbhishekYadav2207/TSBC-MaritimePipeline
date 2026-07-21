# Comprehensive Maritime NLP Corpus Quality Report

This report evaluates the scale, document length, structural diversity, scaffolding influence, BERT tokenizer compatibility, Maritime Information Density (MID), and pretraining readiness of the maritime corpus.

---

## 1. Corpus Scale & Information Density
* **Total Documents**: 96,714
* **Total Words (Tokens)**: 3,277,542
* **Total Characters**: 20,671,676
* **Unique Vocabulary**: 42,762 terms
* **Maritime Information Density (MID)**: **1.00** concepts / 100 words

---

## 2. Document Length Distribution
* **Mean Length**: 33.89 words
* **Median Length**: 36.00 words
* **Standard Deviation**: 15.00 words
* **Percentiles**: P10=10, P25=25, P50=36, P75=44, P90=50, P95=53
* **Min / Max**: 4 / 513 words

### Length Buckets
* `<20 words`: 18,259 (18.9%)
* `20–50 words`: 68,640 (71.0%)
* `50–100 words`: 9,615 (9.9%)
* `100–200 words`: 176 (0.2%)
* `200–512 words`: 23 (0.0%)
* `>512 words`: 1 (0.0%)

---

## 3. Linguistic Diversity
* **Type-Token Ratio (TTR)**: 0.01305
* **Shannon Entropy**: 8.2438 bits
* **Unique Sentences**: 127,025
* **Unique Paragraphs**: 96,709

---

## 4. Duplication & Near-Duplicate Analysis
* **Sentence Duplicate Ratio**: 7.41%
* **Paragraph Duplicate Ratio**: 0.01%
* **Scaffold-Reduced Near-Duplicate Rate (MinHash LSH)**: 20.58%
* **Template Pattern Concentration**: 41.89% (Top pattern: `raw_tsb_summary`)

---

## 5. Maritime Domain Coverage
* **Top Domain Bigrams**: 'the vessel', 'the fishing', 'fishing vessel', 'on board', 'people on'
* **Top Domain Trigrams**: 'the fishing vessel', 'people on board', 'on board reported', 'the vessel was', 'reported being disabled'
* **Top Domain 4-Grams**: 'people on board reported', 'the canadian coast guard', 'on board reported being', 'board reported being disabled', 'the vessel was towed'

---

## 6. Template Influence
* **Template Scaffolding Token Ratio**: 66.42%
* **Domain-Derived Token Ratio**: 33.58%

---

## 7. BERT Tokenizer Compatibility
* **BERT Model**: `bert-base-uncased`
* **Tokenizer Fertility (Subwords/Word)**: 1.3450
* **Maritime Fragmentation Rate**: 25.45%
* **OOV / [UNK] Rate**: 0.0000%

---

## 8. BERT MLM Baseline Diagnostic
* **MLM Evaluation Model**: `bert-base-uncased`
* **General Tokens Top-1 Accuracy**: 39.75%
* **Maritime Tokens Top-1 Accuracy**: 28.18%
* **Performance Gap**: 11.57%

---

## 9. Multi-Dimensional Readiness Dimensions
* ✅ **Relational Integrity**: PASS
* ✅ **Linguistic Quality**: PASS
* ⚠️ **Semantic Density**: WARN
* ⚠️ **Duplication**: WARN
* ⚠️ **Template Influence**: WARN
* ✅ **Domain Coverage**: PASS
* ✅ **Bert Compatibility**: PASS

---

## 10. Pretraining Readiness Assessment

# Status: **NEEDS IMPROVEMENT**

* **Assessment Summary**: Corpus evaluation across 7 quality dimensions.

---

## 11. BEFORE (v2 Baseline) vs. AFTER (Optimized) Ablation Comparison

| Metric | BEFORE (v2 Baseline) | AFTER (Optimized) | Delta / Change |
| :--- | :--- | :--- | :--- |
| **Total Documents** | 117,889 | 96,714 | -21,175 |
| **Total Tokens (Words)** | 4,033,904 | 3,277,542 | -756,362 |
| **Maritime Information Density (MID)** | 0.00 | **1.00** | **+1.00 concepts/100w** |
| **Mean Doc Length (words)** | 34.22 | 33.89 | -0.33 words |
| **Median Doc Length (words)**| 35.00 | 36.00 | +1.00 words |
| **Sentence Duplication Rate** | 16.21% | 7.41% | **-8.80%** |
| **Near-Duplicate Rate (MinHash)**| 16.24% | 20.58% | **+4.34%** |
| **Template Scaffolding Ratio**| 54.82% | 66.42% | **+11.60%** |
| **Top Pattern Concentration** | 42.83% | 41.89% | **-0.94%** |
| **Tokenizer Fertility** | 1.4858 | 1.3450 | -0.1408 |
| **Maritime Fragmentation** | 27.60% | 25.45% | **-2.15%** |
