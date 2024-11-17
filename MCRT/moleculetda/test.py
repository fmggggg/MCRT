from moleculetda.structure_to_vectorization import structure_to_pd
import matplotlib.pyplot as plt
from moleculetda.plotting import plot_pds
import numpy as np

filename = 'cifs/T2_Predicted_Structures/T2_1_1061.cif'

# return a dict containing persistence diagrams for different dimensions (1d - channels, 2d - voids)
arr_dgms = structure_to_pd(filename, supercell_size=100)

# plot out the 1d and 2d diagrams
dgm_0d = arr_dgms['dim0']
dgm_1d = arr_dgms['dim1']
dgm_2d = arr_dgms['dim2']
print(dgm_0d)
# plot_pds(dgm_1d, dgm_2d)

# initialize parameters for the "image" representation:
# spread: Gaussian spread of the kernel, pixels: size of representation (n, n),
# weighting_type: how to weigh the persistence diagram points
# Optional: specs can be provided to give bounds on the representation
from moleculetda.vectorize_pds import PersImage, pd_vectorization
from moleculetda.plotting import plot_pers_images

image_0d=pd_vectorization(dgm_0d, spread=0.15, weighting='identity', pixels=[50, 50])
image_1d=pd_vectorization(dgm_1d, spread=0.15, weighting='identity', pixels=[50, 50])
image_2d=pd_vectorization(dgm_2d, spread=0.15, weighting='identity', pixels=[50, 50])
print(type(image_0d))
print(image_0d)
# np.save('image_1d.npy', image_1d)
# images_loaded = np.load('image_1d.npy')
# print(type(images_loaded))
# print(images_loaded)
# # get both the 1d and 2d representations
# images = []
# for dim in [1, 2]:
#     dgm = arr_dgms[f"dim{dim}"]
#     images.append(pd_vectorization(dgm, spread=0.15, weighting='identity', pixels=[50, 50],specs={
#                 "maxB": 8,
#                 "maxP": 8,
#                 "minBD": 0
#             }))

# plot_pers_images(images, arr_dgms)