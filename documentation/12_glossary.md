# Comprehensive Terminology & Methodology Glossary

This glossary provides formal definitions for technical, domain-specific, statistical, and architectural terms used throughout the **TSBC Maritime Pipeline** documentation suite.

---

### A
- **Administrative Noise**: Non-semantic system boilerplate, audit timestamps, database primary key labels (e.g., `"Formerly OccNo: M14P0003"`), or database status headers present in raw relational database exports that degrade language model pretraining quality.
- **AutoModelForMaskedLM**: PyTorch class from the Hugging Face `transformers` library providing a pretrained language model architecture with a Masked Language Modeling prediction head on top.

### B
- **Banded Locality-Sensitive Hashing (Banding LSH)**: An algorithmic optimization technique that divides a $K$-dimensional MinHash signature vector into $b$ bands of $r$ rows. Documents sharing identical signature sub-vectors within at least one band are hashed into the same bucket, reducing near-duplicate search complexity from $O(N^2)$ to sub-linear $O(N)$.
- **Bernoulli Masking**: A stochastic token masking algorithm where each candidate token in an input sequence is independently selected for masking with a probability $p$ (typically $p = 0.15$ or 15%), drawn from a Bernoulli distribution $\text{Bernoulli}(p)$.
- **Byte-Pair Encoding (BPE)**: A subword tokenization algorithm that iteratively merges the most frequent adjacent pairs of characters or bytes in a corpus until a predefined vocabulary size is reached (used by RoBERTa and ModernBERT).

### C
- **Cartesian Product Guard**: Algorithmic checks implemented in relational table merging (`05_merge_tables.py`) that aggregate child tables independently before joining onto parent entities, preventing multiplicative record expansion.
- **Cliff's Delta ($\delta$)**: A non-parametric effect size statistic measuring the probability that a randomly selected score from one distribution is higher than a score from another, without assuming normal distribution.
- **Cohen's $d$**: A parametric standardized effect size statistic expressing the difference between two sample means in units of pooled standard deviation.
- **Concept Gain Calculator**: An algorithmic component in Stage 06 that evaluates candidate text clauses and appends them to a document only if they introduce at least one novel domain concept relative to existing document text ($\text{Gain} \ge 1$).
- **Composite Key**: A primary key composed of two or more database attributes (e.g., `(VesselID, OccID)`) required to uniquely identify a vessel involvement unit across occurrences.

### D
- **Domain-Adaptive Pretraining (DAPT)**: The machine learning strategy of continuing self-supervised pretraining of an existing pretrained language model (e.g., BERT) on a specialized in-domain text corpus before fine-tuning on downstream tasks.
- **DAPT-Vect (Domain-Adaptive Pretraining with Vocabulary Extension)**: A hybrid adaptation strategy where new domain-specific terms are explicitly inserted into a model's token embedding layer prior to executing continued DAPT.

### E
- **Exponential MLM Loss ($\exp(\mathcal{L})$)**: The exponential of the Cross-Entropy loss $\exp(\mathcal{L}_{\text{MLM}})$, representing the model's perplexity-equivalent token uncertainty across masked prediction sites.

### F
- **Fertility (Subword Fertility)**: The average number of subword tokens produced per raw whitespace word by a tokenizer ($\Phi = W_{\text{subword}} / W_{\text{raw}}$). Lower fertility indicates better alignment with domain vocabulary.
- **Foreign Key (FK)**: A database column or set of columns in one table that references the primary key of another table, establishing a relational link.

### J
- **Jaccard Similarity Coefficient**: A statistical metric measuring the similarity between two sample sets, defined as the size of their intersection divided by the size of their union ($J(A, B) = \|A \cap B\| / \|A \cup B\|$).

### K
- **Knowledge Tier**: A classification category (High Knowledge, Medium Knowledge, Low Knowledge, Redundant, Noisy) assigned to documents based on their 10-feature Semantic Importance Score.
- **Knowledge Unit Graph Engine**: A graph-based document synthesis framework (`06_generate_documents.py`) that combines primary operational narratives with equipment and casualty clauses.

### M
- **Maritime Information Density (MID)**: A normalized metric measuring the number of extracted maritime domain concepts per 100 words of text ($\text{MID} = \text{Concepts} / (\text{Words} / 100)$).
- **Maritime Understanding Index (MUI)**: A composite mathematical score ($0.0 \le \text{MUI} \le 100.0$) ranking language models across Top-1 accuracy, rare term accuracy, MLM loss, subword fragmentation, OOV rate, and category balance.
- **Masked Language Modeling (MLM)**: A self-supervised pretraining objective where a percentage of input tokens are masked, and the model is trained to predict the original tokens using bidirectional context.
- **MinHash**: A probabilistic technique for estimating the Jaccard similarity between sets by computing the minimum hash values under $K$ independent hash functions.

### O
- **Out-Of-Vocabulary (OOV) Rate**: The proportion of subword tokens in a dataset mapped to the generic unknown token `[UNK]` by a tokenizer ($\eta_{\text{oov}} = W_{\text{unk}} / W_{\text{subword}}$).
- **Oxford Comma**: A comma placed immediately before the coordinating conjunction in a series of three or more terms (enforced by `text_sanitizer.join_words_grammatical`).

### P
- **Primary Key (PK)**: A column or set of columns that uniquely identifies each row in a database table without null values.
- **Provenance Spans**: Metadata annotations attached to rendered text tracking whether specific spans originate from static template scaffolding or source database attributes.

### S
- **Shannon Entropy ($H(X)$)**: A metric measuring the average information content or uncertainty in a token frequency distribution, expressed in bits per token.
- **Subword Fragmentation Rate**: The proportion of domain vocabulary terms split into 2 or more subword pieces by a tokenizer ($F_{\text{frag}} = (T - N_{\text{single}}) / T$).

### T
- **Type-Token Ratio (TTR)**: The ratio of unique vocabulary words (types) to total words (tokens) in a text, serving as a measure of lexical diversity ($\text{TTR} = \|V\| / N$).

### W
- **WordPiece**: A subword tokenization algorithm used by BERT that selects subword merges maximizing the likelihood of a unguided language model.
