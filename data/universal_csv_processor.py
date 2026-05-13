"""
Universal Data Processor for CSV/XLSX files
Generates output compatible with PrivPGD pipeline using DataHandler

INPUTS:
    - file_path: Path to CSV or XLSX file
    - label_column: Name of target column (or None to auto-detect)
    - disc_k: Number of discretization bins (default: 32)

OUTPUTS (same format as ACS processing):
    - data_disc.csv: Discretized training data
    - testdata_disc.csv: Discretized test data
    - data_original.csv: Original training data
    - testdata_original.csv: Original test data
    - domain.json: Domain specification
    - inverse_mapping.pkl: Reverse mapping
"""

import json
import os
import pickle
import re
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .data_handler import DataHandler


class UniversalDataProcessor:
    """
    Process any CSV/XLSX dataset to PrivPGD-compatible format.
    Handles categorical encodings like "0-17", "18-25" automatically.
    """

    def __init__(self, output_base_path: str = "datasets"):
        self.output_base_path = output_base_path
        os.makedirs(output_base_path, exist_ok=True)

    def process_file(
        self,
        file_path: str,
        dataset_name: Optional[str] = None,
        label_column: Optional[str] = None,
        columns_to_drop: Optional[List[str]] = None,
        disc_k: int = 32,
        test_size: float = 0.2,
        random_state: int = 42,
        sheet_name: int = 0,
    ) -> str:
        """
        Process a CSV or XLSX file to PrivPGD format.

        Args:
            file_path: Path to input file (.csv or .xlsx)
            dataset_name: Name for output directory (default: derived from filename)
            label_column: Name of target column (default: last column)
            columns_to_drop: List of column names to exclude (e.g., IDs, timestamps)
            disc_k: Number of discretization bins
            test_size: Proportion of data for testing (0.0-1.0)
            random_state: Random seed for reproducibility
            sheet_name: For Excel files, sheet index or name

        Returns:
            Path to output directory containing processed files
        """

        # Step 1: Load data
        print(f"\n{'='*80}")
        print(f"PROCESSING: {file_path}")
        print(f"{'='*80}\n")

        df_raw = self._load_file(file_path, sheet_name)
        print(f"✓ Loaded: {len(df_raw)} rows × {len(df_raw.columns)} columns")

        # Step 2: Handle special categorical encodings and preprocessing
        df_preprocessed = self._preprocess_dataframe(
            df_raw,
            columns_to_drop=columns_to_drop
        )
        print(f"✓ Preprocessed: {len(df_preprocessed.columns)} columns retained")

        # Step 3: Identify and move label column to end
        if label_column is None:
            label_column = df_preprocessed.columns[-1]
            print(f"⚠ No label specified, using last column: '{label_column}'")
        else:
            if label_column not in df_preprocessed.columns:
                raise ValueError(f"Label column '{label_column}' not found!")
            # Move label to end
            cols = [c for c in df_preprocessed.columns if c != label_column]
            df_preprocessed = df_preprocessed[cols + [label_column]]
            print(f"✓ Label column: '{label_column}'")

        # Step 4: Use DataHandler for discretization (same as ACS pipeline)
        print(f"\n{'='*80}")
        print(f"DISCRETIZING WITH DataHandler (k={disc_k})")
        print(f"{'='*80}\n")

        # Debug: Show column types and unique values before DataHandler
        print("Column info before DataHandler:")
        for col in df_preprocessed.columns:
            n_unique = df_preprocessed[col].nunique()
            dtype = df_preprocessed[col].dtype
            print(f"  {col}: dtype={dtype}, unique_values={n_unique}")
        print()

        datahandler = DataHandler(df_preprocessed)
        df_processed, domain, inverse_mapping = datahandler.forward(disc_k)

        print(f"✓ Discretization complete")
        print(f"✓ Domain: {domain}")

        # Step 5: Train/test split (same indices for processed and original)
        train_indices, test_indices = train_test_split(
            df_processed.index,
            test_size=test_size,
            random_state=random_state
        )

        train_df_processed = df_processed.iloc[train_indices]
        test_df_processed = df_processed.iloc[test_indices]
        train_df_original = df_preprocessed.iloc[train_indices]
        test_df_original = df_preprocessed.iloc[test_indices]

        print(f"✓ Split: {len(train_df_processed)} train, {len(test_df_processed)} test")

        # Step 6: Create output directory and save (same format as ACS)
        if dataset_name is None:
            dataset_name = Path(file_path).stem

        output_path = os.path.join(
            self.output_base_path,
            f"{dataset_name}_disc{disc_k}"
        )
        os.makedirs(output_path, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"SAVING TO: {output_path}")
        print(f"{'='*80}\n")

        # Save training data
        self._save_data(
            train_df_processed,
            train_df_original,
            output_path,
            datahandler,
            is_test=False
        )

        # Save test data
        self._save_data(
            test_df_processed,
            test_df_original,
            output_path,
            datahandler,
            is_test=True
        )

        print(f"\n{'='*80}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"\nOutput files created:")
        print(f"  📄 data_disc.csv          - Training data (for PrivPGD)")
        print(f"  📄 domain.json            - Domain specification (for PrivPGD)")
        print(f"  📄 testdata_disc.csv      - Test data")
        print(f"  📄 data_original.csv      - Original training data")
        print(f"  📄 testdata_original.csv  - Original test data")
        print(f"  📄 inverse_mapping.pkl    - Reverse mapping")

        return output_path

    def _load_file(self, file_path: str, sheet_name) -> pd.DataFrame:
        """Load CSV or XLSX file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == '.csv':
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except (UnicodeDecodeError, UnicodeError):
                    continue
            raise ValueError("Could not decode CSV file with standard encodings")

        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, sheet_name=sheet_name)

        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .csv, .xlsx, or .xls")

    def _preprocess_dataframe(
        self,
        df: pd.DataFrame,
        columns_to_drop: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Preprocess dataframe: handle special encodings, missing values, etc.
        Handles categorical encodings like "0-17", "18-25", etc.
        """
        df = df.copy()

        # Drop specified columns
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')

        # Ensure column names are strings
        df.columns = [str(col) for col in df.columns]

        for col in df.columns:
            # Handle missing values
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    # For non-numeric, fill with mode or a placeholder
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                    else:
                        df[col].fillna('Unknown', inplace=True)

            # Handle categorical encodings like "0-17", "18-25", "55+"
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = self._convert_range_encoding(df[col])

        return df

    def _convert_range_encoding(self, series: pd.Series) -> pd.Series:
        """
        Convert categorical range encodings to numeric ordinal values.

        Examples:
            "0-17" → 0
            "18-25" → 1
            "26-35" → 2
            "55+" → 3
            "A" → 0
            "B" → 1

        If series is already numeric or can be converted, returns as-is.
        Otherwise, uses ordinal encoding based on natural sorting.
        """
        # Try converting to numeric directly
        try:
            numeric_series = pd.to_numeric(series, errors='raise')
            # Ensure it's not boolean type
            if series.dtype == bool or numeric_series.dtype == bool:
                return numeric_series.astype(int)
            return numeric_series
        except (ValueError, TypeError):
            pass

        # Check if it's a range encoding (e.g., "0-17", "18-25", "55+")
        sample_val = str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else ""

        if self._is_range_encoding(sample_val):
            # Sort by the starting number of each range
            unique_vals = series.unique()
            sorted_vals = sorted(unique_vals, key=self._extract_range_start)
            mapping = {val: idx for idx, val in enumerate(sorted_vals)}
            # Return as int (not bool) to preserve cardinality
            return series.map(mapping).astype('int64')

        # Default: ordinal encoding with natural sorting
        # This preserves the actual number of unique values
        unique_vals = sorted(series.dropna().unique(), key=lambda x: str(x))
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        # Add NaN mapping if present
        result = series.map(mapping)
        if series.isna().any():
            result = result.fillna(-1)
        # Return as int64 to preserve cardinality
        return result.astype('int64')

    def _is_range_encoding(self, value: str) -> bool:
        """Check if a value represents a range encoding like "0-17" or "55+"."""
        if pd.isna(value):
            return False
        value_str = str(value).strip()
        # Pattern: number-number, number+, or <number
        patterns = [
            r'^\d+-\d+$',      # "0-17"
            r'^\d+\+$',        # "55+"
            r'^<\d+$',         # "<18"
            r'^\d+$',          # Just a number
        ]
        return any(re.match(pattern, value_str) for pattern in patterns)

    def _extract_range_start(self, value: str) -> float:
        """Extract the starting number from a range encoding for sorting."""
        if pd.isna(value):
            return float('inf')

        value_str = str(value).strip()

        # Extract first number
        match = re.search(r'\d+', value_str)
        if match:
            return float(match.group())

        return float('inf')

    def _save_data(
        self,
        df_processed: pd.DataFrame,
        df_original: pd.DataFrame,
        output_path: str,
        datahandler: DataHandler,
        is_test: bool
    ):
        """Save processed and original data (same format as ACS pipeline)."""
        name = "testdata" if is_test else "data"

        # Save processed dataframe to CSV
        df_processed.to_csv(
            os.path.join(output_path, f"{name}_disc.csv"),
            index=False
        )

        # Save original dataframe to CSV
        df_original.to_csv(
            os.path.join(output_path, f"{name}_original.csv"),
            index=False
        )

        # Save domain and inverse_mapping only for training data
        if not is_test:
            # Save domain as JSON (convert numpy types to Python types)
            domain_serializable = {
                k: int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else v
                for k, v in datahandler.domain.items()
            }
            with open(os.path.join(output_path, "domain.json"), "w") as f:
                json.dump(domain_serializable, f, indent=2)

            # Save inverse_mapping using pickle
            with open(os.path.join(output_path, "inverse_mapping.pkl"), "wb") as f:
                pickle.dump(datahandler.inverse_mapping, f)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def process_csv_for_privpgd(
    file_path: str,
    dataset_name: Optional[str] = None,
    label_column: Optional[str] = None,
    columns_to_drop: Optional[List[str]] = None,
    disc_k: int = 32,
) -> str:
    """
    Quick function to process a CSV/XLSX file for PrivPGD.

    Args:
        file_path: Path to CSV or XLSX file
        dataset_name: Output directory name (default: filename)
        label_column: Target column name (default: last column)
        columns_to_drop: List of columns to exclude (e.g., ["User_ID", "Product_ID"])
        disc_k: Number of discretization bins

    Returns:
        Path to output directory

    Example:
        # Black Friday Dataset
        output_path = process_csv_for_privpgd(
            file_path="BlackFriday.csv",
            label_column="Purchase",
            columns_to_drop=["User_ID", "Product_ID"]
        )

        # NYC Taxi Dataset
        output_path = process_csv_for_privpgd(
            file_path="green_tripdata_2016-12.csv",
            label_column="fare_amount",
            columns_to_drop=["VendorID", "lpep_pickup_datetime"]
        )

        # Any dataset
        output_path = process_csv_for_privpgd(
            file_path="my_data.csv",
            label_column="target"
        )
    """
    processor = UniversalDataProcessor()
    return processor.process_file(
        file_path=file_path,
        dataset_name=dataset_name,
        label_column=label_column,
        columns_to_drop=columns_to_drop,
        disc_k=disc_k
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """
    Usage instructions and examples.
    """
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           UNIVERSAL CSV/XLSX PROCESSOR FOR PRIVPGD                         ║
╚════════════════════════════════════════════════════════════════════════════╝

This processor converts any CSV/XLSX file to the format required by PrivPGD.
Output format is identical to the ACS data processing pipeline.

BASIC USAGE:
───────────────────────────────────────────────────────────────────────────────
from universal_csv_processor import process_csv_for_privpgd

output_path = process_csv_for_privpgd("your_data.csv")

REQUIRED INPUT:
───────────────────────────────────────────────────────────────────────────────
  file_path        : Path to CSV or XLSX file

OPTIONAL INPUTS:
───────────────────────────────────────────────────────────────────────────────
  dataset_name     : Output folder name (default: filename)
  label_column     : Target column (default: last column)
  columns_to_drop  : List of columns to exclude (default: none)
  disc_k           : Discretization bins (default: 32)

OUTPUT:
───────────────────────────────────────────────────────────────────────────────
  datasets/{dataset_name}_disc{k}/
    ├── data_disc.csv          ← For PrivPGD training
    ├── domain.json            ← For PrivPGD training
    ├── testdata_disc.csv      ← For evaluation
    ├── data_original.csv      ← Original training data
    ├── testdata_original.csv  ← Original test data
    └── inverse_mapping.pkl    ← Reverse mapping

SPECIAL FEATURES:
───────────────────────────────────────────────────────────────────────────────
  ✓ Handles range encodings: "0-17", "18-25", "55+"
  ✓ Handles missing values automatically
  ✓ Works with both CSV and XLSX files
  ✓ Uses DataHandler (same as ACS pipeline)
  ✓ Output format compatible with PrivPGD

NEXT STEP:
───────────────────────────────────────────────────────────────────────────────
  Use output with PrivPGD:
  
  from run_privpgd import run_privpgd
  
  result = run_privpgd(
      train_dataset=f"{output_path}/data_disc.csv",
      domain=f"{output_path}/domain.json",
      savedir=output_path
  )
    """)