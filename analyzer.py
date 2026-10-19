#import libraries
import numpy as np 
import numpy.typing as npt
import pandas as pd 
from data import load_data
# Constants



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

    columns = [
        "G1", "G2", "G3",
        "studytime", "failures", "absences"
    ]
 
    X = data[columns].to_numpy(dtype=np.float64)
    low = np.array([0, 0, 0, 1, 0, 0])
    high = np.array([20, 20, 20, 4, 4, 93])
    validate_scores(
        scores=X,
        n_subjects=len(columns),
        low=low,
        high=high
    )

    print("Validation passed!")
    G1, G2, G3, STUDYTIME, FAILURES, ABSENCES = range(6)

    mean_X=X.mean(axis=0)
    std_X=X.std(axis=0)
    min_X=X.min(axis=0)
    max_X=X.max(axis=0)

    data_new=[
        mean_X,
        std_X,
        max_X,
        min_X
    ]

    description_table=pd.DataFrame(
        data=data_new,
        columns=columns,
        index=["mean","std","max","min"]
    )

    print("*"*40 + "task-1" + "*"*40)
    print(description_table)
    print("*"*100)
    print(data.describe())


    #  Task 2 

    change_12 = X[: , G2] - X[: , G1]
   
    change_23= X[: , G3] - X[: , G2]


    changes = [("G1→G2", change_12), ("G2→G3", change_23)]
    for name, change in changes:

        print(f"{name} : mean: {change.mean()} | improved most :  {change.argmax()} | dropped most : {change.argmin()} |  same: {change[ change == 0 ].size} ")


    #  Task 3
    Y = X[:,G3]
    mask = (Y == 0)
    g3_zeros=mask.sum()
    print(g3_zeros) #There are 38 zeros they may include student did not enter the exam

    idx=idx = np.flatnonzero(mask)
    print(len(idx))
    
    print(X[idx, G1])      # their G1 grades
    print(X[idx, G2])      # their G2 grades


    # task 4
    unique_studytimes = np.unique(X[:,STUDYTIME])
    for st in unique_studytimes:
        mask = (X[:,STUDYTIME] == st)

        mean_g3 = X[mask,G3].mean()

        print(f"Studytime: {st}, G3 mean : {mean_g3:.4f}")
              
    print(f"Pandas groupby Check: ")
    studytime_g3_mean = data.groupby("studytime")["G3"].mean()
    print(studytime_g3_mean)

    mask1=X[:,FAILURES] == 0
    failures_equal_zero_mean = X[mask1,G3].mean()
    print(failures_equal_zero_mean)

    mask2=X[:,FAILURES] >= 1
    failures_at_least_one_mean = X[mask2,G3].mean()
    print(failures_at_least_one_mean)

    # Task 5

    highest_G3=X[:,G3].max()
    highest_indices=np.where(X[:,G3] == highest_G3)[0]
    count_highest= len(highest_indices)
    print(f"The highest value is: {highest_G3}")
    print(f"It appears {count_highest} time(s) at indices: {highest_indices}")










if __name__=="__main__":
    main()





