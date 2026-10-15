#import libraries
import numpy as np 
import numpy.typing as npt
import pandas as pd 
from analyzer import load_data
# Constants

SUBJECTS=["Math","Geography","Physics","Biology","Chemistry"]

def validate_scores(
    scores: npt.NDArray[np.float64],
    n_subjects: int,
    low: float | npt.NDArray = 0.0,
    high: float | npt.NDArray = 20.0,
) -> None:
    """Validate numeric data using configurable column limits."""

    if scores.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {scores.ndim}D."
        )

    if not np.issubdtype(scores.dtype, np.floating):
        raise ValueError(
            f"Expected floating dtype, got {scores.dtype}."
        )

    if scores.shape[1] != n_subjects:
        raise ValueError(
            f"Expected {n_subjects} columns, got {scores.shape[1]}."
        )

    lower = np.asarray(low, dtype=np.float64)
    upper = np.asarray(high, dtype=np.float64)

    # Each bound must be a scalar or one value per column.
    allowed_shapes = ((), (n_subjects,))

    if lower.shape not in allowed_shapes:
        raise ValueError("Invalid shape for lower bounds.")

    if upper.shape not in allowed_shapes:
        raise ValueError("Invalid shape for upper bounds.")

    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or not np.all(lower <= upper)
    ):
        raise ValueError("Invalid validation bounds.")

    valid = np.isnan(scores) | (
        np.isfinite(scores)
        & (scores >= lower)
        & (scores <= upper)
    )

    if not np.all(valid):
        count = np.count_nonzero(~valid)
        raise ValueError(f"Found {count} invalid values.")


def main():
    """Main function to generate and display student scores."""

    data = load_data()

    print(f"first 10 rows:\n{data.head(10)}")
    print("*"*100)
    print( f"shape of df:\n {data.shape}")
    print("*"*100)
    print( f"columns:\n{data.columns}")
    print("*"*100)
    print( f"type of data: \n {data.dtypes}")
    print("*"*100)  
    data.info() # For missing values 


    columns = [
        "G1", "G2", "G3",
        "studytime", "failures", "absences"
    ]

    X = data[columns].to_numpy(dtype=np.float64)
    print(X.min(axis=0))
    print(X.max(axis=0))
    
    low = np.array([0, 0, 0, 1, 0, 0])
    high = np.array([20, 20, 20, 4, 4, 93])

    validate_scores(
        scores=X,
        n_subjects=len(columns),
        low=low,
        high=high
    )

    print("Validation passed!")









if __name__=="__main__":
    main()





