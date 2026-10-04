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
    clip_scores = np.clip(scores,0,100).round(1)# (30,5)
    
    return clip_scores


def main():
    """Main function to generate and display student scores."""
    subject_means = np.array([60, 55, 65, 70, 75])

    rng=np.random.default_rng(42)
    scores = generate_scores(rng=rng,n_students=N_STUDENTS,subject_means=subject_means)
    shape_scores= scores.shape
    print(f"Shape of scores : {shape_scores}")
    type_scores=scores.dtype
    print(f"Type of scores : {type_scores}")
    print(scores[:3])



if __name__=="__main__":
    main()

    

