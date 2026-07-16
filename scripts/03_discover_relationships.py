import os
import json
from pathlib import Path
import networkx as nx
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("03_discover_relationships")

def build_relationship_graph(profiling_report: dict) -> dict:
    """Discovers relationships and builds a networkx relationship graph."""
    table_names = list(profiling_report.keys())
    
    # 1. Identify all identifier columns (containing ID, No, Key) in each table
    table_ids = {}
    for table_name, profile in profiling_report.items():
        ids = []
        for col_name, col_metrics in profile["columns"].items():
            col_lower = col_name.lower()
            if any(k in col_lower for k in ["id", "no", "key"]):
                # Skip secondary flags/attributes that happen to have "no" in them (e.g., "officialno" is fine, but "noLSE" is a count)
                if col_lower == "nolse" or "dontknow" in col_lower:
                    continue
                ids.append(col_name)
        table_ids[table_name] = ids
        
    logger.info("Identifier columns per table:")
    for t, ids in table_ids.items():
        logger.info(f"  {t}: {ids}")
        
    # 2. Find shared keys between tables and identify the parent table
    # The parent table for a key is the one that has the maximum number of unique values for that key
    key_parents = {}
    all_keys = set()
    for ids in table_ids.values():
        all_keys.update(ids)
        
    for key in all_keys:
        tables_with_key = []
        for table_name, ids in table_ids.items():
            if key in ids:
                tables_with_key.append(table_name)
                
        if len(tables_with_key) > 1:
            # Determine which table is the parent (highest nunique)
            parent_table = None
            max_unique = -1
            
            for t in tables_with_key:
                nunique = profiling_report[t]["columns"][key]["nunique"]
                if nunique > max_unique:
                    max_unique = nunique
                    parent_table = t
                    
            key_parents[key] = {
                "parent_table": parent_table,
                "child_tables": [t for t in tables_with_key if t != parent_table],
                "nunique": max_unique
            }
            logger.info(f"Key '{key}' parent inferred as '{parent_table}' with {max_unique} unique values.")
            
    # 3. Create relationship graph using networkx
    G = nx.DiGraph()
    
    # Add tables as nodes
    for t in table_names:
        G.add_node(t, row_count=profiling_report[t]["row_count"])
        
    relationships = []
    
    # Add edges representing relationships (Parent -> Child)
    for key, info in key_parents.items():
        parent = info["parent_table"]
        for child in info["child_tables"]:
            # Determine relationship type (usually One-to-Many from Parent to Child)
            # We verify this by looking at unique values vs row counts
            G.add_edge(parent, child, key=key, type="one_to_many")
            relationships.append({
                "parent_table": parent,
                "child_table": child,
                "join_key": key,
                "relationship_type": "one_to_many"
            })
            logger.info(f"Inferred relationship: '{parent}' --({key})--> '{child}' (One-to-Many)")
            
    # Export relationship structure
    # Serialize networkx graph info
    graph_data = {
        "nodes": [{"id": node, "row_count": G.nodes[node]["row_count"]} for node in G.nodes],
        "edges": [{"source": u, "target": v, "key": d["key"], "type": d["type"]} for u, v, d in G.edges(data=True)],
        "relationships": relationships,
        "key_parents": key_parents
    }
    
    return graph_data

def main():
    root = get_project_root()
    config = load_config()
    
    report_path = root / config.get("output_dir", "outputs") / "profiling_report.json"
    if not report_path.exists():
        logger.error(f"Profiling report not found at {report_path}! Run Step 2 first.")
        return
        
    with open(report_path, "r", encoding="utf-8") as f:
        profiling_report = json.load(f)
        
    logger.info("Discovering table relationships...")
    graph_data = build_relationship_graph(profiling_report)
    
    out_path = root / config.get("output_dir", "outputs") / "relationships.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)
        
    logger.info(f"Relationships discovered and saved successfully to {out_path}")

if __name__ == "__main__":
    main()
