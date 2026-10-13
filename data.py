#import libraries
import numpy as np 
import numpy.typing as npt

# Constants
N_STUDENTS=30
SUBJECTS=["Math","Geography","Physics","Biology","Chemistry"]



def generate_scores(rng: np.random.Generator, n_students:int,subject_means:npt.NDArray ,std:float=12.0)->   npt.NDArray[np.floating]:
    """
    Generate random scores for students in different subjects.
    """
    scores = rng.normal(loc=subject_means,scale=std,size=(n_students   ,len(subject_means))) # (30,5)
    clip_scores = np.clip(scores,0,100).round()# (30,5)
    
    return clip_scores


def inject_missing_values(scores: npt.NDArray[np.floating], n_missing:int,rng:np.random.Generator) -> npt.NDArray[np.floating]:
    """
    Inject missing values into the scores array.
    """
    

    missing_indices = rng.choice(scores.size, size=n_missing, replace=False)


    # Get the 2D indices corresponding to the flattened indices
    missing_positions = np.unravel_index(missing_indices, scores.shape)


    result=scores.copy() 
    # Inject missing values (NaN) at the selected positions
    result[missing_positions] = np.nan

    return result



def validate_scores(
    scores: npt.NDArray[np.float64],
    n_subjects: int
) -> None:
    """
    Validate a 2D array of student scores.

    Missing values (NaN) are allowed.

    Args:
        scores: A 2D NumPy array of floating-point scores.
        n_subjects: Expected number of subject columns.

    Raises:
        ValueError: If the array has the wrong dimensions,
            dtype, column count, or invalid score values.
    """

    if scores.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, got {scores.ndim}D."
        )

    if not np.issubdtype(scores.dtype, np.floating):
        raise ValueError(
            f"Expected a floating-point dtype, got {scores.dtype}."
        )

    if scores.shape[1] != n_subjects:
        raise ValueError(
            f"Expected {n_subjects} subjects, got {scores.shape[1]}"
        )

    valid = np.isnan(scores) | (
        np.isfinite(scores) & (scores >= 0) & (scores <= 100)
    )

    if not np.all(valid):
        raise ValueError(
            f"Found {np.count_nonzero(~valid)} invalid scores."
        )


def main():
    """Main function to generate and display student scores."""
    subject_means = np.array([60, 55, 65, 70, 75])

    rng=np.random.default_rng(42)

    scores_full = generate_scores(rng=rng,n_students=N_STUDENTS,subject_means=subject_means)

    n_missing=rng.integers(1,10)

    scores=inject_missing_values(scores_full,n_missing,rng)
    validate_scores(scores, n_subjects=len(SUBJECTS))
    print("Validation passed ✅")
    scores_bad=scores_full.copy()

    scores_bad[0,0]=150 # ValueError: All values inside the scores should be between the 0 and 100. 
    scores_bad_1D=scores_full[0] #ValueError: Expected a 2D array , got 1D
    int_scores=scores_full.astype(np.int64) # ValueError: Expected data type is float , got int64
    bad_subjects=scores_full.T  # ValueError: Number of subjects in scores does not match expected number.
    bad_inputs = [
        ("1D array",   scores_bad_1D),
        ("int dtype",  int_scores),
        ("transposed", bad_subjects),
        ("value 150",  scores_bad),
    ]

    for name, bad in bad_inputs:
        try:
            validate_scores(bad, n_subjects=len(SUBJECTS))
            print(f"❌ {name}: NOT caught!")
        except ValueError as e:
            print(f"✅ {name}: caught → {e}")
   
    
    
    mean=np.nanmean(scores,axis=0) #(5,)
    std=np.nanstd(scores,axis=0) # (5,)
    mins=np.nanmin(scores,axis=0) #(5,)
    maxs=np.nanmax(scores,axis=0) #(5,)


    print(f"{'Subjects':<10} {'Mean':<6} {'Std':<6} {'Min':<6} {'Max':<6}")
    print("-" * 40)
    for subject, mean_val, std_val, min_val, max_val in zip(SUBJECTS, mean, std, mins, maxs):
        print(f"{subject:<10} {mean_val:<6.2f} {std_val:<6.2f} {min_val:<6.2f} {max_val:<6.2f}")


    valid_counts = np.isfinite(scores).sum(axis=0)  
    missing_counts = np.isnan(scores).sum(axis=0)

    print("Valid counts:", valid_counts)
    print("Missing counts:", missing_counts)

    if not np.all(valid_counts + missing_counts == N_STUDENTS):
        raise ValueError("Sanity check failed!")

    print("Sanity check passed ✅")



if __name__=="__main__":
    main()







