import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
from ase.data import covalent_radii
from MCRT.assets.colors import cpk_colors
from MCRT.visualize.utils import plot_cube
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.rcParams["font.family"] = "Times New Roman"

def draw_colorbar(fig, ax, cmap, minatt, maxatt, **cbar_kwargs):
    norm = Normalize(0.0, 1.0)
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(
        smap, ax=ax, fraction=cbar_kwargs["fraction"], shrink=cbar_kwargs["shrink"]
    )
    cbar.ax.tick_params(labelsize=cbar_kwargs["fontsize"])
    ticks_loc = np.linspace(0, 1, cbar_kwargs["num_ticks"])
    ticks_label = np.round(
        np.linspace(minatt, maxatt, cbar_kwargs["num_ticks"]),
        decimals=cbar_kwargs["decimals"],
    )
    cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    cbar.ax.set_yticklabels(ticks_label)

    cbar.ax.set_ylabel(
        "Attention score",
        rotation=270,
        labelpad=cbar_kwargs["labelpad"],
        fontdict={"size": cbar_kwargs["labelsize"]},
    )


def draw_line(ax, pos1, pos2, **kwargs):
    """
    Draw line from position 1 to position 2
    :param ax: <matplotlib.axes> figure axis
    :param pos1: <np.array> starting point position
    :param pos2: <np.array> end point position
    :param kwargs: matplotlib plot3D kwargs
    :return:
    """
    ax.plot3D(*zip(pos1, pos2), **kwargs)


def draw_cell(ax, lattice, s_point=None, **kwargs):
    """
    Draw unit-p_lattice p_lattice using matplotlib
    :param ax: <matplotlib.axes> figure axis
    :param lattice: <np.array> p_lattice vectors (3 X 3 matrix)
    :param s_point: <np.array> start point of p_lattice
    :param kwargs: matplotlib plot3D kwargs
    """
    vec1, vec2, vec3 = lattice
    if s_point is None:
        s_point = np.zeros(3)

    opp_vec = vec1 + vec2 + vec3 + s_point

    for v1, v2 in combinations([vec1, vec2, vec3], 2):
        draw_line(ax, s_point, s_point + v1, **kwargs)
        draw_line(ax, s_point, s_point + v2, **kwargs)
        draw_line(ax, s_point + v1, s_point + v1 + v2, **kwargs)
        draw_line(ax, s_point + v2, s_point + v1 + v2, **kwargs)
        draw_line(ax, s_point + v1 + v2, opp_vec, **kwargs)


def draw_atoms(ax, atoms, atomic_scale):
    """
    Draw p_atoms using matplotlib
    :param ax: <matplotlib.axes> figure axis
    :param atoms: <ase.p_atoms> Target p_atoms for drawing
    :param atomic_scale: <float> scaling factor for draw_atoms.
    """
    coords = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    atomic_sizes = np.array([covalent_radii[i] for i in atomic_numbers])
    atomic_colors = np.array([cpk_colors[i] for i in atomic_numbers])
    ax.scatter(
        xs=coords[:, 0],
        ys=coords[:, 1],
        zs=coords[:, 2],
        c=atomic_colors,
        s=atomic_sizes * atomic_scale,
        marker="o",
        edgecolor="black",
        linewidths=0.8,
        alpha=1.0,
    )


def draw_heatmap_graph(ax, atoms, colors, atomic_scale, alpha):
    coords = atoms.get_positions()
    for i, coord in enumerate(coords):
        ax.scatter(
            xs=coord[0],
            ys=coord[1],
            zs=coord[2],
            color=colors[i],
            s=atomic_scale,
            marker="o",
            linewidth=0,
            alpha=alpha,
        )



def tick_labels(max_birth, max_persistence, pixel_size):
    """Convert image units to units of the persistence diagram.

    Args:
        max_birth: Maximum birth time for the x-axis.
        max_persistence: Maximum persistence time for the y-axis.
        pixel_size: Pixel size resolution for the image (int).
    """

    ticks = np.linspace(0, pixel_size, 6)

    ticklabels_x = [(max_birth / pixel_size) * i for i in ticks]
    ticklabels_y = [(max_persistence / pixel_size) * i for i in ticks]

    ticklabels_x = [round(elem, 2) for elem in ticklabels_x]
    ticklabels_y = [round(elem, 2) for elem in ticklabels_y]

    return ticklabels_x, ticklabels_y


def filter_top_n_heatmap(heatmap, top_n):
    """Keep only the top N elements in the heatmap and set others to zero."""
    flattened_heatmap = heatmap.flatten()
    top_indices = np.argpartition(flattened_heatmap, -top_n)[-top_n:]
    filtered_heatmap = np.zeros_like(flattened_heatmap)
    filtered_heatmap[top_indices] = flattened_heatmap[top_indices]
    return filtered_heatmap.reshape(heatmap.shape)

def get_highest_attention_patch_center(heatmap, max_birth, max_persistence):
    """Get the center coordinates of the highest attention patch."""
    max_index = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    patch_size = 5  # Each patch is 5x5 pixels
    center_x = (max_birth / 10) * (max_index[1] + 0.5)
    center_y = (max_persistence / 10) * (10-(max_index[0] + 0.5))
    return center_x, center_y

def create_blue_colormap():
    """Create a blue colormap with transparency for zero values."""
    colors = [(0, 0, 0, 0), (0, 0, 1, 1)]  # RGBA for transparent and blue
    return LinearSegmentedColormap.from_list('blue_cmap', colors, N=256)

def create_pink_colormap():
    """Create a pink colormap with transparency for zero values."""
    colors = [(1, 0.9, 0.9, 0.2), (1, 0.75, 0.8, 1)]  # RGBA for transparent and pink
    return LinearSegmentedColormap.from_list('pink_cmap', colors, N=256)

def create_orange_red_colormap():
    """Create an orange-red colormap with transparency for zero values."""
    colors = color_steps = [(1, 0.9, 0.9, 0)] + [(1, 0.8 - i * (1/30), 0.8 - i * (1/30), 1) for i in range(30)]  # RGBA for transparent, orange, and red
    return LinearSegmentedColormap.from_list('orange_red_cmap', colors, N=256)

def plot_image(image, max_birth, max_persistence, heatmap, top_n=10):
    """Take a persistence image and plot it with a heatmap overlay.

    Args:
        image: The persistence image to be plotted.
        max_birth: Maximum birth time for the x-axis.
        max_persistence: Maximum persistence time for the y-axis.
        heatmap: 1D array representing the heatmap image.
        top_n: Number of top elements to keep in the heatmap.
    """

    # Take the logarithm of the image to enhance contrast
    image_log = np.log1p(image)

    fig, ax = plt.subplots(figsize=(8, 6))

    pixel_size = image.shape[0]  
    ticklabels_x, ticklabels_y = tick_labels(max_birth, max_persistence, pixel_size)
    ticks = np.linspace(0, pixel_size, 6)

    # Display the original image
    im = ax.imshow(image_log, cmap=plt.cm.viridis_r, origin='upper', extent=[0, pixel_size, 0, pixel_size])

    # Prepare the heatmap
    heatmap_reshaped = heatmap[:100].reshape((10, 10))  # Adjust the heatmap size to 10x10
    heatmap_reshaped[heatmap_reshaped < 0.001] = 0  # Set values below 0.001 to zero

    # Get the highest attention patch center
    center_x, center_y = get_highest_attention_patch_center(heatmap_reshaped, max_birth, max_persistence)
    print(f"Highest Attention Patch Center - Birth: {center_x}, Persistence: {center_y}")

    # Filter to keep only top N values
    heatmap_filtered = filter_top_n_heatmap(heatmap_reshaped, top_n)

    # Upsample the filtered heatmap to match the image size (50x50)
    # heatmap_resized = np.kron(heatmap_filtered, np.ones((5, 5)))  # Use np.kron to upsample
    heatmap_resized = np.clip(np.kron(heatmap_filtered, np.ones((5, 5))), 0, 0.03)
    # Create an orange-red colormap
    orange_red_cmap = create_orange_red_colormap()

    # Overlay the heatmap
    # hm = ax.imshow(heatmap_resized, cmap=orange_red_cmap, alpha=0.5, extent=[0, pixel_size, 0, pixel_size])
    hm = ax.imshow(heatmap_resized, cmap=orange_red_cmap, alpha=0.4, extent=[0, pixel_size, 0, pixel_size], vmin=0, vmax=0.03)
    # Set tick labels
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels_x)
    ax.set_yticklabels(ticklabels_y)
    ax.set_xlabel("Birth",fontsize=16)
    ax.set_ylabel("Persistence",fontsize=16)
    # ax.set_title("Persistence Image")

    # Add colorbars
    divider = make_axes_locatable(ax)
    cax_im = divider.append_axes("right", size="5%", pad=0.1)
    cax_hm = divider.append_axes("right", size="5%", pad=0.46)  # Adjust pad to move cax_hm to the left

    cbar_image = plt.colorbar(im, cax=cax_im)
    cbar_image.set_label('Image Intensity')

    cbar_heatmap = plt.colorbar(hm, cax=cax_hm)
    cbar_heatmap.set_label('Attention Score')

    plt.tight_layout()
    return fig,ax
    # plt.show()
