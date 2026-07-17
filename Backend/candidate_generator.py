import pandas as pd

# Path to alloy dataset
DATASET_PATH = "Dataset/Alloy Properties/alloys.csv"


def generate_candidates(min_strength, min_melting_point, top_n=5):
    """
    Generate candidate alloys based on user-defined properties.
    
    Parameters:
    - min_strength: minimum tensile strength (psi)
    - min_melting_point: minimum melting temperature
    - top_n: number of candidates to return
    """

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Rename columns for easier access
    df = df.rename(columns={
        "Tensile Strength: Ultimate (UTS) (psi)": "UTS",
        "Melting Completion (Liquidus)": "MeltingPoint"
    })

    # Drop rows with missing values
    df = df.dropna(subset=["UTS", "MeltingPoint"])

    # Filter candidates
    candidates = df[
        (df["UTS"] >= min_strength) &
        (df["MeltingPoint"] >= min_melting_point)
    ]

    # Sort by strength
    candidates = candidates.sort_values(by="UTS", ascending=False)

    # Select top results
    top_candidates = candidates.head(top_n)

    return top_candidates[["Alloy", "UTS", "MeltingPoint"]]


if __name__ == "__main__":
    # Realistic values based on dataset
    min_strength = 600
    min_melting_point = 1400

    results = generate_candidates(min_strength, min_melting_point)

    print("\nCandidate Materials:")
    print(results)

