# Section 01: Data Ingestion & Preprocessing (Stages 01–05a)

This document provides complete technical specifications for the data ingestion, dictionary parsing, dataset profiling, schema relationship discovery, semantic attribute selection, hierarchical table merging, and record validation stages.

---

## Shared Utility Modules

### 1. `pipeline_utils.py`
- **File**: [`scripts/pipeline_utils.py`](file:///c:/--Files--/Programming/pipeline/scripts/pipeline_utils.py)
- **Key Functions**:
  - `get_project_root() -> Path`: Dynamically resolves the project root directory.
  - `load_config() -> dict`: Loads and parses `config/config.json`.
  - `setup_logging(stage_name: str) -> logging.Logger`: Configures dual console (`stdout`) and file loggers writing to `outputs/logs/pipeline.log`.
  - `read_csv_safe(file_path: Path, **kwargs) -> pd.DataFrame`: Auto-detects encoding (`utf-8-sig`, fallback to `latin-1`), strips leading/trailing spaces from column headers, validates `usecols`, and sets `low_memory=False`.
  - `detect_datasets() -> dict`: Matches files in `data/*.csv` to standard table stems (`MDOTW_VW_OCCURRENCE_PUBLIC`, `MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC`, etc.).

### 2. `text_sanitizer.py`
- **File**: [`scripts/text_sanitizer.py`](file:///c:/--Files--/Programming/pipeline/scripts/text_sanitizer.py)
- **Key Functions**:
  - `strip_administrative_noise(text: str) -> str`: Scrubs administrative leakage tags (`formerly occno: X`, `data extraction status pending`, `record id: 12345`).
  - `join_words_grammatical(words: list, conjunction: str = "and") -> str`: Joins token lists with Oxford commas.
  - `format_cargo_description(cargo_prod, cargo_qty)`: Prevents duplicate phrases like `"cargo cargo"`.
  - `format_damage_description(degree, location)`: Prevents duplicate phrases like `"damaged damage"`.
  - `format_casualty_count(count, singular_term, plural_term)`: Formats casualty counts with singular/plural agreement.

---

## Stage 01: Parse Data Dictionary
- **Script**: [`scripts/01_parse_dictionary.py`](file:///c:/--Files--/Programming/pipeline/scripts/01_parse_dictionary.py)
- **Execution Command**: `python run_pipeline.py --stage 01`
- **Algorithmic Logic**:
  1. Ingests `MDOTW-MARSIS-Master-dataset-inventory-and-dictionary-English.csv`.
  2. `map_display_columns()` groups by table name and pairs numeric ID/Enum/IND columns with corresponding `DisplayEng` columns (e.g., `WeatherConditionEnum` $\rightarrow$ `WeatherConditionDisplayEng`).
  3. `categorize_column()` categorizes columns into functional NLP types (`admin`, `temporal`, `spatial`, `environmental`, `vessel_spec`, `casualty`, `equipment`, `narrative`).
- **Input File**: Master dictionary CSV in `data/`
- **Output File**: [`outputs/dictionary_metadata.json`](file:///c:/--Files--/Programming/pipeline/outputs/dictionary_metadata.json)

---

## Stage 02: Profile Datasets
- **Script**: [`scripts/02_profile_dataset.py`](file:///c:/--Files--/Programming/pipeline/scripts/02_profile_dataset.py)
- **Execution Command**: `python run_pipeline.py --stage 02`
- **Algorithmic Logic**:
  1. Scans 6 raw relational table CSVs in `data/`:
     - `MARSISdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv` (87,760 rows)
     - `MARSISdb_MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC.csv` (73,926 rows)
     - `MARSISdb_MDOTW_VW_INJURIES_PUBLIC.csv` (23,004 rows)
     - Equipment tables (LSA: 75,257 rows, NAV: 314,447 rows, REC: 78,399 rows)
  2. Computes per-column metrics: missing value ratio, data types, cardinality (unique count), top 5 frequent categories, and candidate primary/foreign key flags.
- **Output File**: [`outputs/profiling_report.json`](file:///c:/--Files--/Programming/pipeline/outputs/profiling_report.json)

---

## Stage 03: Discover Schema Relationships
- **Script**: [`scripts/03_discover_relationships.py`](file:///c:/--Files--/Programming/pipeline/scripts/03_discover_relationships.py)
- **Execution Command**: `python run_pipeline.py --stage 03`
- **Algorithmic Logic**:
  1. Evaluates primary key `OccID` across child tables (`VW_OCCURRENCE_VESSEL`, `VW_INJURIES`, LSA, NAV, REC).
  2. Quantifies cardinalities (1-to-Many vessel joins, 1-to-Many equipment joins) to build the relational schema graph.
- **Output File**: [`outputs/relationships.json`](file:///c:/--Files--/Programming/pipeline/outputs/relationships.json)

---

## Stage 04: Select Semantic Columns
- **Script**: [`scripts/04_select_semantic_columns.py`](file:///c:/--Files--/Programming/pipeline/scripts/04_select_semantic_columns.py)
- **Execution Command**: `python run_pipeline.py --stage 04`
- **Algorithmic Logic**:
  1. Evaluates columns based on non-null ratio, string length, and descriptive relevance.
  2. Discards low-information administrative columns (GUIDs, entry dates, row IDs, French duplicates).
  3. Preserves high-information semantic attributes (weather, location, vessel specs, gross tonnage, incident class).
- **Output File**: [`outputs/selected_semantic_columns.json`](file:///c:/--Files--/Programming/pipeline/outputs/selected_semantic_columns.json)

---

## Stage 05: Merge Datasets
- **Script**: [`scripts/05_merge_tables.py`](file:///c:/--Files--/Programming/pipeline/scripts/05_merge_tables.py)
- **Execution Command**: `python run_pipeline.py --stage 05`
- **Algorithmic Logic**:
  1. Deduplicates parent occurrence records by `OccID`, aggregating duplicate text fields using custom semicolon concatenation (`val1; val2`).
  2. Deduplicates vessel records by composite key `(VesselID, OccID)`.
  3. Executes a Left Outer Join grouped by `OccID`. Merges parent occurrences with nested arrays of child vessels, injuries, and equipment.
  4. Maps orphaned child records to a synthetic placeholder vessel (`VesselID: 999999999`, `VesselName: "UNSPECIFIED VESSEL"`).
- **Output File**: [`outputs/merged_records.jsonl`](file:///c:/--Files--/Programming/pipeline/outputs/merged_records.jsonl) (346 MB, 96,714 records)

---

## Stage 05a: Validate Records
- **Script**: [`scripts/05a_validate_records.py`](file:///c:/--Files--/Programming/pipeline/scripts/05a_validate_records.py)
- **Execution Command**: `python run_pipeline.py --stage 05a`
- **Algorithmic Logic**:
  1. Evaluates data integrity across `merged_records.jsonl`.
  2. Verifies `OccID` completeness, key structural integrity, non-empty vessel lists, and detects orphaned records.
  3. Checks numeric bounds (`max_vessel_speed_knots`: 100.0, `max_tonnage`: 300,000, `max_crew`: 1,000).
- **Output File**: [`outputs/validation_report.json`](file:///c:/--Files--/Programming/pipeline/outputs/validation_report.json)
