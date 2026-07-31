# Section 01: Data Ingestion & Preprocessing (Stages 01–05a)

This document details the data ingestion, schema profiling, relationship discovery, attribute selection, table merging, and record validation steps.

---

## Shared Utility Modules

### 1. `pipeline_utils.py`
- **File**: [scripts/pipeline_utils.py](file:///c:/--Files--/Programming/pipeline/scripts/pipeline_utils.py)
- **Functions**:
  - `get_project_root()`: Resolves workspace absolute path.
  - `load_config()`: Ingests `config/config.json`.
  - `setup_logging(name)`: Configures standard console and file loggers writing to `outputs/logs/pipeline.log`.
  - `load_csv_safely(file_path)`: Auto-detects file encoding (UTF-8, ISO-8859-1, Windows-1252) and loads pandas DataFrames cleanly.

### 2. `text_sanitizer.py`
- **File**: [scripts/text_sanitizer.py](file:///c:/--Files--/Programming/pipeline/scripts/text_sanitizer.py)
- **Functions**:
  - `sanitize_text(text)`: Stricter regex normalization replacing non-ASCII noise, collapsing spaces, and standardizing quotes/dashes.
  - `mask_pii(text)`: Scrubs administrative occurrence reference tags and internal database IDs.

---

## Stage 01: Parse Data Dictionary
- **Script**: [scripts/01_parse_dictionary.py](file:///c:/--Files--/Programming/pipeline/scripts/01_parse_dictionary.py)
- **Logic**: Reads `MDOTW-MARSIS-Master-dataset-inventory-and-dictionary-English.csv`. Maps table names to column specifications, database data types, descriptions, and discrete value lookup translations.
- **Output File**: [outputs/dictionary_metadata.json](file:///c:/--Files--/Programming/pipeline/outputs/dictionary_metadata.json)

---

## Stage 02: Profile Datasets
- **Script**: [scripts/02_profile_dataset.py](file:///c:/--Files--/Programming/pipeline/scripts/02_profile_dataset.py)
- **Logic**: Scans 6 relational CSV files in `data/`:
  - `MARSISdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv` (87,760 rows)
  - `MARSISdb_MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC.csv` (73,926 rows)
  - `MARSISdb_MDOTW_VW_INJURIES_PUBLIC.csv` (23,004 rows)
  - Equipment tables (LSA: 75,257 rows, NAV: 314,447 rows, REC: 78,399 rows)
- **Metrics Computed**: Missingness ratios, unique cardinalities, data types, sample values.
- **Output File**: [outputs/profiling_report.json](file:///c:/--Files--/Programming/pipeline/outputs/profiling_report.json)

---

## Stage 03: Discover Schema Relationships
- **Script**: [scripts/03_discover_relationships.py](file:///c:/--Files--/Programming/pipeline/scripts/03_discover_relationships.py)
- **Logic**: Matches primary key `OccID` across child tables (`VW_OCCURRENCE_VESSEL`, `VW_INJURIES`, equipment tables). Quantifies cardinalities (1-to-Many vessel joins, 1-to-Many equipment joins) to build the relational execution graph.
- **Output File**: [outputs/relationships.json](file:///c:/--Files--/Programming/pipeline/outputs/relationships.json)

---

## Stage 04: Select Semantic Columns
- **Script**: [scripts/04_select_semantic_columns.py](file:///c:/--Files--/Programming/pipeline/scripts/04_select_semantic_columns.py)
- **Logic**: Evaluates columns based on non-null ratio, string length, and descriptive relevance. Retains key attributes (weather, location, vessel type, gross tonnage, incident class) while dropping internal flags and administrative IDs.
- **Output File**: [outputs/selected_semantic_columns.json](file:///c:/--Files--/Programming/pipeline/outputs/selected_semantic_columns.json)

---

## Stage 05: Merge Datasets
- **Script**: [scripts/05_merge_tables.py](file:///c:/--Files--/Programming/pipeline/scripts/05_merge_tables.py)
- **Logic**: Performs a relational join grouped by `OccID`. Merges occurrence details with nested lists of vessels, injuries, LSA equipment, navigation aids, and voyage recorders into unified JSON objects.
- **Output File**: [outputs/merged_records.jsonl](file:///c:/--Files--/Programming/pipeline/outputs/merged_records.jsonl) (346 MB)

---

## Stage 05a: Validate Records
- **Script**: [scripts/05a_validate_records.py](file:///c:/--Files--/Programming/pipeline/scripts/05a_validate_records.py)
- **Logic**: Checks record integrity across `merged_records.jsonl`. Verifies `OccID` completeness, key structural integrity, non-empty vessel lists, and detects orphaned records.
- **Output File**: [outputs/validation_report.json](file:///c:/--Files--/Programming/pipeline/outputs/validation_report.json)
