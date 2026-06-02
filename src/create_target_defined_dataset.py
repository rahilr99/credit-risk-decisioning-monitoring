from pathlib import Path

from data_ingestion import iter_raw_lendingclub_data_chunks
from target_definition import add_bad_loan_target, filter_target_defined_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
TARGET_DEFINED_DATA_PATH = INTERIM_DATA_DIR / "lendingclub_target_defined.csv.gz"

def create_target_defined_dataset_in_chunks(
        chunksize: int = 100_000,
        output_path: Path = TARGET_DEFINED_DATA_PATH, 
) -> None:
    """
    Create a target-defined interim dataset from the raw LendingClub file. 

    This function: 
    1. Reads the raw dataset in chunks.
    2. Adds the bad_loan target. 
    3. Keeps only rows where bad_loan is defined. 
    4. Saves the result to data/interim as a compressed CSV.

    The output file is intended for local project use and should not be committed to Github. 

    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    total_raw_rows = 0
    total_saved_rows = 0
    chunk_count = 0

    print("Creating target-defined interim dataset...")
    print(f"Chunk size: {chunksize:,} rows")
    print(f"Output path: {output_path}")

    for chunk in iter_raw_lendingclub_data_chunks(chunksize=chunksize):
        chunk_count += 1
        total_raw_rows += len(chunk) 

        chunk_with_target = add_bad_loan_target(chunk)
        target_defined_chunk = filter_target_defined_rows(chunk_with_target)

        write_header = chunk_count == 1

        target_defined_chunk.to_csv(
            output_path, mode="w" if write_header else "a", 
            header = write_header, 
            index= False, 
            compression = "gzip", 
        )

        total_saved_rows += len(target_defined_chunk)

        print(
            f"Processed chunk {chunk_count:,} | "
            f"Raw rows so far: {total_raw_rows:,} | "
            f"Saved rows so far: {total_saved_rows:,}"
        )

    print("\nTarget-defined interim dataset created.")
    print(f"Total raw rows processed: {total_raw_rows:,}")
    print(f"Total target-defined rows saved: {total_saved_rows:,}")
    print(f"Rows excluded from supervised dataset: {total_raw_rows - total_saved_rows:,}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__": 
    create_target_defined_dataset_in_chunks(chunksize=100_000)