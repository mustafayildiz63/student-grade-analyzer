#import libraries
import numpy as np 
import numpy.typing as npt

# Constants
N_STUDENTS=30
SUBJECTS=["Math","Geography","Physics","Biology","Chemistry"]



def generate_scores(rng: np.random.Generator, n_students:int,subject_means:npt.NDArray ,std:float=12.0)->   npt.NDArray[np.float64]:
    """
    Generate random scores for students in different subjects.
    """
    scores = rng.normal(loc=subject_means,scale=std,size=(n_students   ,len(subject_means))) # (30,5)
    clip_scores = np.clip(scores,0,100).round()# (30,5)
    
    return clip_scores


def inject_missing_values(scores: npt.NDArray[np.float64], n_missing:int,rng:np.random.Generator) -> npt.NDArray[np.float64]:
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


def main():
    """Main function to generate and display student scores."""
    subject_means = np.array([60, 55, 65, 70, 75])

    rng=np.random.default_rng(42)

    scores_full = generate_scores(rng=rng,n_students=N_STUDENTS,subject_means=subject_means)
    
    n_missing=rng.integers(1,10)

    scores=inject_missing_values(scores_full,n_missing,rng)
    print(f"NaNs in scores:{np.isnan(scores).sum()}")  # Count of NaN values in scores
    print(f"NaNs in scores_full:{np.isnan(scores_full).sum()}")  # Count of NaN values in scores_full
    print(f"NaN is Subjects :{np.isnan(scores).sum(axis=0)}")  # Subjects with NaN values

if __name__=="__main__":
    main()







