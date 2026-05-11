import polars as pl
import os
import zipfile
import glob

# Configuration
BASE_DIR = "/Users/akash/dev/DATA245/Project"
TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = BASE_DIR
YEARS = list(range(2018, 2026))

# Essential columns for ML
CORE_COLUMNS = [
    "STATE_CODE_001",
    "STRUCTURE_NUMBER_008",
    "YEAR_BUILT_027",
    "ADT_029",
    "YEAR_ADT_030",
    "SERVICE_ON_042A",
    "STRUCTURE_KIND_043A",
    "STRUCTURE_TYPE_043B",
    "STRUCTURE_LEN_MT_049",
    "ROADWAY_WIDTH_MT_051",
    "DECK_COND_058",
    "SUPERSTRUCTURE_COND_059",
    "SUBSTRUCTURE_COND_060",
    "CHANNEL_COND_061",
    "CULVERT_COND_062",
    "DATE_OF_INSPECT_090",
    "INSPECT_FREQ_MONTHS_091",
    "HIGHWAY_SYSTEM_104",
    "YEAR_RECONSTRUCTED_106",
    "PERCENT_ADT_TRUCK_109"
]

def extract_and_load(year):
    zip_path = os.path.join(TEMP_DIR, f"{year}.zip")
    print(f"Processing year {year}...")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find the text file in the zip
        txt_files = [f for f in z.namelist() if f.endswith('.txt')]
        if not txt_files:
            return None
        
        with z.open(txt_files[0]) as f:
            try:
                # NBI data is often encoded in Latin-1 or similar
                content = f.read()
                df = pl.read_csv(content, ignore_errors=True, truncate_ragged_lines=True, encoding="latin1")
            except Exception as e:
                print(f"Error reading year {year}: {e}")
                # Fallback to lossy utf8
                df = pl.read_csv(content, ignore_errors=True, truncate_ragged_lines=True, encoding="utf8-lossy")
            
            # Standardize column names to uppercase
            df.columns = [c.upper() for c in df.columns]
            
            # Select columns that exist in this year
            available_cols = [c for c in CORE_COLUMNS if c in df.columns]
            df = df.select(available_cols)
            
            # Ensure Structure Number is a string and padded
            df = df.with_columns([
                pl.col("STRUCTURE_NUMBER_008").cast(pl.Utf8).str.strip_chars().alias("BRIDGE_ID"),
                pl.col("STATE_CODE_001").cast(pl.Utf8).alias("STATE_ID")
            ])
            
            # Create a unique key
            df = df.with_columns(
                (pl.col("STATE_ID") + "_" + pl.col("BRIDGE_ID")).alias("UNIQUE_KEY")
            )
            
            # Add year
            df = df.with_columns(pl.lit(year).alias("YEAR"))
            
            return df

def main():
    all_dfs = []
    bridge_sets = []
    
    for year in YEARS:
        df = extract_and_load(year)
        if df is not None:
            all_dfs.append(df)
            # Collect unique keys for intersection
            bridge_sets.append(set(df["UNIQUE_KEY"].to_list()))
    
    if not bridge_sets:
        print("No data loaded.")
        return

    # Find bridges present in ALL years
    consistent_bridges = set.intersection(*bridge_sets)
    print(f"Found {len(consistent_bridges)} bridges consistent across all {len(YEARS)} years.")
    
    # Filter each year's dataframe
    filtered_dfs = []
    for df in all_dfs:
        # Filter rows
        df_filtered = df.filter(pl.col("UNIQUE_KEY").is_in(list(consistent_bridges)))
        filtered_dfs.append(df_filtered)
    
    # Combine all
    full_data = pl.concat(filtered_dfs, how="diagonal")
    
    # Deriving Bridge Condition (Good, Fair, Poor)
    # FHWA Condition Ratings: 0-9 (N for Not Applicable)
    # Good: Min rating >= 7
    # Fair: Min rating 5 or 6
    # Poor: Min rating <= 4
    
    rating_cols = ["DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060", "CULVERT_COND_062"]
    
    # Convert to numeric, handle 'N' as null
    for col in rating_cols:
        full_data = full_data.with_columns(
            pl.col(col).cast(pl.Utf8).str.strip_chars()
            .map_elements(lambda x: int(x) if x and x.isdigit() else None, return_dtype=pl.Int64)
            .alias(f"{col}_NUM")
        )
    
    # Calculate lowest rating
    full_data = full_data.with_columns(
        pl.min_horizontal([f"{col}_NUM" for col in rating_cols]).alias("LOWEST_RATING")
    )
    
    # Assign Condition
    full_data = full_data.with_columns(
        pl.when(pl.col("LOWEST_RATING") >= 7).then(pl.lit("Good"))
        .when(pl.col("LOWEST_RATING") >= 5).then(pl.lit("Fair"))
        .when(pl.col("LOWEST_RATING").is_not_null()).then(pl.lit("Poor"))
        .otherwise(None)
        .alias("BRIDGE_CONDITION")
    )
    
    # Split into train and test
    train_data = full_data.filter(pl.col("YEAR") <= 2024)
    test_data = full_data.filter(pl.col("YEAR") == 2025)
    
    print(f"Train records: {train_data.height}")
    print(f"Test records: {test_data.height}")
    
    # Save to CSV
    train_path = os.path.join(OUTPUT_DIR, "nbi_train_2018_2024.csv")
    test_path = os.path.join(OUTPUT_DIR, "nbi_test_2025.csv")
    
    print(f"Saving to {train_path}...")
    train_data.write_csv(train_path)
    print(f"Saving to {test_path}...")
    test_data.write_csv(test_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
