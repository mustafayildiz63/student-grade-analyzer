import numpy as np 
import numpy.typing as npt
from data import load_data


def validate_scores(
    scores: npt.NDArray[np.float64],
    n_subjects: int,
    low: float | npt.NDArray = 0.0,
    high: float | npt.NDArray = 20.0,
) -> None:
    """Validate numeric data using configurable column limits."""
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D array, got {scores.ndim}D.")

    if not np.issubdtype(scores.dtype, np.floating):
        raise ValueError(f"Expected floating dtype, got {scores.dtype}.")

    if scores.shape[1] != n_subjects:
        raise ValueError(f"Expected {n_subjects} columns, got {scores.shape[1]}.")

    lower = np.asarray(low, dtype=np.float64)
    upper = np.asarray(high, dtype=np.float64)

    if lower.shape not in ((), (n_subjects,)) or upper.shape not in ((), (n_subjects,)):
        raise ValueError("Invalid shape for validation bounds.")

    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or not np.all(lower <= upper):
        raise ValueError("Invalid validation bounds.")

    valid = np.isnan(scores) | (np.isfinite(scores) & (scores >= lower) & (scores <= upper))

    if not np.all(valid):
        count = np.count_nonzero(~valid)
        raise ValueError(f"Found {count} invalid values.")


def main():
    """Load, validate, and analyze student data."""
    data = load_data()

    columns = ["G1", "G2", "G3", "studytime", "failures", "absences"]
    X = data[columns].to_numpy(dtype=np.float64)

    # Validate inputs
    low = np.array([0, 0, 0, 1, 0, 0])
    high = np.array([20, 20, 20, 4, 4, 93])
    validate_scores(scores=X, n_subjects=len(columns), low=low, high=high)
    print("Validation passed!\n" + "="*50)

    # Column mappings
    G1, G2, G3, STUDYTIME, FAILURES, ABSENCES = range(6)
    total_students = X.shape[0]
    print(f"Total students: {total_students}")

    # --- M4 Task 1: Academic Groupings ---
    zero_grade = X[:, G3] == 0
    nonzero_failure = (X[:, G3] > 0) & (X[:, G3] < 10)
    passed = X[:, G3] >= 10

    total_zero = zero_grade.sum()
    total_nonzero_fail = nonzero_failure.sum()
    total_passed = passed.sum()

    print(f"Zero final grades       : {total_zero}")
    print(f"Nonzero failing grades  : {total_nonzero_fail}")
    print(f"Passing grades          : {total_passed}")

    # Overlap and integrity checks
    assert total_zero + total_nonzero_fail + total_passed == total_students
    assert not np.any(zero_grade & nonzero_failure), "Overlap between zero and nonzero failure!"
    assert not np.any(zero_grade & passed), "Overlap between zero and passed!"
    assert not np.any(nonzero_failure & passed), "Overlap between nonzero failure and passed!"

    # --- M4 Task 2: Scenario Analysis ---
    print("\n" + "="*20 + " Scenario Analysis " + "="*20)
    
    # Scenario A: Including zeros
    failure_rate_a = ((total_zero + total_nonzero_fail) / total_students) * 100
    print(f"Scenario A (Includes Zeros) Failure Rate : {failure_rate_a:.2f}%")

    # Scenario B: Excluding zeros
    students_excluding_zeros = total_students - total_zero
    failure_rate_b = (total_nonzero_fail / students_excluding_zeros) * 100
    print(f"Scenario B (Excludes Zeros) Failure Rate : {failure_rate_b:.2f}%")

    # --- M4 Task 3: Compare Academic Groups ---
    print("\n" + "="*20 + " Group Mean Comparisons " + "="*20)
    
    masks = {
        "Zero Grade Group": zero_grade,
        "Nonzero Failure Group": nonzero_failure,
        "Passed Group": passed,
    }

    for name, mask in masks.items():
        print(f"[{name}]")
        print(f"  G1 Mean        : {X[mask, G1].mean():.2f}")
        print(f"  Studytime Mean : {X[mask, STUDYTIME].mean():.2f}")
        print(f"  Failures Mean  : {X[mask, FAILURES].mean():.2f}")
        print(f"  Absences Mean  : {X[mask, ABSENCES].mean():.2f}")
        print("-" * 30)


if __name__ == "__main__":
    main()



