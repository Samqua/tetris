import numpy as np
import itertools
from numba import jit
import time

def rotate(piece,num_rots):
    coords=[np.array(p) for p in piece] # each of these is a 2-element vector containing the (x,y) coords of a point in the piece
    rot_mat_90=np.array([[0,-1],[1,0]])
    rot_mat_180=np.dot(rot_mat_90,rot_mat_90)
    rot_mat_270=np.dot(rot_mat_90,rot_mat_180)
    if num_rots==0:
        return piece
    elif num_rots==1:
        transformed_coords=[np.dot(rot_mat_90,c) for c in coords]
        return np.array(transformed_coords).tolist()
    elif num_rots==2:
        transformed_coords=[np.dot(rot_mat_180,c) for c in coords]
        return np.array(transformed_coords).tolist()
    elif num_rots==3:
        transformed_coords=[np.dot(rot_mat_270,c) for c in coords]
        return np.array(transformed_coords).tolist()
    elif num_rots>3:
        return rotate(piece,num_rots%4)

def x_y_rot_piece_to_indices(x,y,rot,piece):
    # returns the grid indices of a rotated piece centered on (x,y)
    rotated_piece=rotate(piece,rot)
    indices=[[coords[0]+x,coords[1]+y] for coords in rotated_piece]
    return np.array(indices)

def isvalid(N,configuration,pieces):
    # a configuration is a variable length tuple, each sub-tuple of which is (x,y,rot)
    space=np.zeros((N,N))
    configuration=np.array(configuration)
    centers_x=configuration[:,0]
    centers_y=configuration[:,1]
    centers_rot=configuration[:,2]
    all_indices=[x_y_rot_piece_to_indices(x,y,rot,piece) for x,y,rot,piece in zip(centers_x,centers_y,centers_rot,pieces)]
    all_indices=np.array(all_indices)
    # check to make sure none of them are outside of the grid first
    for indices in all_indices:
        if (indices>=N).any() or (indices<0).any():
            return False # that piece tried to go outside the grid
        else:
            # construct the candidate solution (which may have overlaps and be ultimately invalid)
            for i,j in indices:
                space[i,j]+=1
    if (space>=2).any():
        return False
    else:
        return True

@jit
def solve(N,pieces):
    """
    Attempts to solve the tetris puzzle for specified pieces on an NxN grid with a discrete Pi/4 rotation DOF.
    If the puzzle is solvable, returns a solution. If it is unsolvable, returns None.
    A configuration is specified by the (x,y) coordinates of each centerpiece and a corresponding list of rotation values.
    """
    start=time.time()
    i=0
    num_pieces=len(pieces)
    for configuration in itertools.combinations(itertools.product(range(N),range(N),[0,1,2,3]),num_pieces):
        i+=1
        print(configuration)
        if isvalid(N,configuration,pieces):
            print("Solution found!")
            print(configuration)
            print("Total iterations:",i)
            print("Total runtime: ",time.time()-start)
            return configuration
        else:
            continue
    print("No solution.")
    print("Total iterations:",i)
    print("Total runtime: ",time.time()-start)
    return None


pieces=[[[0,0],[0,-1]],
[[0,0],[0,-1],[0,-2],[1,2]],
[[0,0],[1,3],[1,-1]],
]
solve(4,pieces)

