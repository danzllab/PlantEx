# %%
# +
import glob
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from matplotlib import pyplot as plt
from scipy import optimize
from skimage import filters
from skimage.feature import peak_local_max
from sklearn.metrics import pairwise_distances

from matplotlib.patches import Circle, Ellipse
import seaborn as sns

from tqdm.auto import tqdm


# %% [markdown]
# Helper function and classes


# %%
def read_imagej_voxel_size_zyx(file_path):
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    """

    def _xy_voxel_size(tags, key):
        assert key in ["XResolution", "YResolution"]
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.0

    with tifffile.TiffFile(file_path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get("spacing", 1.0)
        else:
            # default voxel size
            z = 1.0

        tags = tiff.pages[0].tags
        # parse X, Y resolution
        y = _xy_voxel_size(tags, "YResolution")
        x = _xy_voxel_size(tags, "XResolution")
        # return voxel size
        return {"z": z, "y": y, "x": x}  # np.array([z, y, x])


def crop_at_3d(im, z, y, x, radius=5):
    dims = [int(z), int(y), int(x)]
    for i, d in enumerate(dims):
        assert (d - radius > -1) and d + radius < im.shape[i], "crop outside of array"

    return im[
        dims[0] - radius : dims[0] + radius + 1,
        dims[1] - radius : dims[1] + radius + 1,
        dims[2] - radius : dims[2] + radius + 1,
    ]


def crop_at_2d(im, y, x, radius=3):
    dims = [int(y), int(x)]
    for i, d in enumerate(dims):
        assert (d - radius > -1) and d + radius < im.shape[i], "crop outside of array"

    return im[
        dims[0] - radius : dims[0] + radius + 1, dims[1] - radius : dims[1] + radius + 1
    ]


def gaussian_diagonal_3d(height, yc, xc, zc, yw, xw, zw):
    return lambda y, x, z: height * np.exp(
        -(((xc - x) / xw) ** 2 + ((yc - y) / yw) ** 2 + ((zc - z) / zw) ** 2) / 2
    )


def gaussian_spherical_3d(height, yc, xc, zc, w):
    return lambda y, x, z: height * np.exp(
        -(((xc - x) / w) ** 2 + ((yc - y) / w) ** 2 + ((zc - z) / w) ** 2) / 2
    )


def gaussian_diagonal_2d(height, yc, xc, yw, xw):
    return lambda y, x: height * np.exp(
        -(((xc - x) / xw) ** 2 + ((yc - y) / yw) ** 2) / 2
    )


def gaussian_spherical_2d(height, yc, xc, w):
    return lambda y, x: height * np.exp(
        -(((xc - x) / w) ** 2 + ((yc - y) / w) ** 2) / 2
    )


def fit_gaussian_diagonal_3d(data):
    data = data - data.min()
    inti_params = (data.max(),) + tuple(np.array(data.shape) / 2) + (1,) * data.ndim
    errorfunction = lambda p: np.ravel(
        gaussian_diagonal_3d(*p)(*np.indices(data.shape)) - data
    )
    fit_params = optimize.least_squares(errorfunction, inti_params)
    okay = False

    if (
        fit_params.success
        and ((fit_params.x[1:4] > 1) & (fit_params.x[1:4] < data.shape[0] - 1)).all()
    ):
        okay = True

    return okay, fit_params.x


def fit_gaussian_spherical_3d(data):
    data = data - data.min()
    init_params = (data.max(),) + tuple(np.array(data.shape) / 2) + (1,)
    errorfunction = lambda p: np.ravel(
        gaussian_spherical_3d(*p)(*np.indices(data.shape)) - data
    )
    fit_params = optimize.least_squares(
        errorfunction,
        init_params,
    )
    okay = False

    if (
        fit_params.success
        and ((fit_params.x[1:4] > 1) & (fit_params.x[1:4] < data.shape[0] - 1)).all()
    ):
        okay = True

    return okay, fit_params.x


def fit_gaussian_diagonal_2d(data):
    data = data - data.min()
    init_params = (data.max(),) + tuple(np.array(data.shape) / 2) + (1,) * data.ndim
    errorfunction = lambda p: np.ravel(
        gaussian_diagonal_2d(*p)(*np.indices(data.shape)) - data
    )
    fit_params = optimize.least_squares(errorfunction, init_params)

    okay = False

    if (
        fit_params.success
        and ((fit_params.x[1:3] > 1) & (fit_params.x[1:3] < data.shape[0] - 1)).all()
    ):
        okay = True

    return okay, fit_params.x


def fit_gaussian_spherical_2d(data):
    data = data - data.min()

    init_params = (data.max(),) + tuple(np.array(data.shape) / 2) + (1,)
    errorfunction = lambda p: np.ravel(
        gaussian_spherical_2d(*p)(*np.indices(data.shape)) - data
    )
    fit_params = optimize.least_squares(
        errorfunction,
        init_params,
    )

    okay = False

    if (
        fit_params.success
        and ((fit_params.x[1:3] > 1) & (fit_params.x[1:3] < data.shape[0] - 1)).all()
    ):
        okay = True

    return okay, fit_params.x


def isolated_points_(ctab, min_dist=7, coords=None):
    if not coords:
        coords = ["dim0", "dim1", "dim2"]
    pw = pairwise_distances(ctab[coords].values)
    pw[np.eye(pw.shape[0]) > 0] = pw.max()

    isolated_points_idx = (pw > min_dist).all(1)

    return ctab[isolated_points_idx].copy()


def isolated_points(ctab, min_dist=7, coords=None):
    if not coords:
        coords = ["dim0", "dim1", "dim2"]
    pw = pairwise_distances(ctab[coords].values)
    pw[np.eye(pw.shape[0]) > 0] = pw.max()

    isolated_points_idx = pw.min(0) > min_dist

    return ctab.iloc[isolated_points_idx].copy()


def normalize_intensity(im, percentiles):
    assert im.dtype in [np.uint8, np.uint16], "only uint8 and uint16"

    max_value = np.iinfo(im.dtype).max

    low, high = np.percentile(im.ravel(), percentiles)
    print(
        " - normalize_intensity():\n\tminimum:",
        low,
        "\n\tmaximum:",
        high,
        "\n\tdtype",
        im.dtype,
    )

    im_norm = ((im.astype(np.float32) - low) / (high - low)).clip(0, 1) * (max_value)

    return im_norm


def read_tif(fn):
    img = tifffile.imread(fn)

    if np.all(img >= 2**15):
        print(
            " - read_tif(): dtype misinterpretation, subtracting 2^15, in order to restore minimum 0"
        )
        img -= 2**15

    return img


def find_peaks(
    img_norm, dims, sigma_gaussian=0.5, threshold_rel=0.33, peak_min_distance=7
):
    img_filt = filters.gaussian(img_norm, sigma_gaussian, preserve_range=True)

    peaks = peak_local_max(
        img_filt,
        min_distance=2,
        threshold_rel=threshold_rel,
        exclude_border=True,
    )

    peaks_tab = pd.DataFrame(peaks, columns=list(dims))

    # filter for isolated points with distance 10px
    peaks_tab_iso = isolated_points(
        peaks_tab, min_dist=peak_min_distance, coords=list(dims)
    )

    print(
        f" - find_peaks(): Found {len(peaks_tab)} peaks total and {len(peaks_tab_iso)} isolated peaks"
    )

    return img_filt, peaks_tab_iso


def crop_fitting_roi(img, peaks_tab, crop_radius=5):
    crops = []
    for i, row in peaks_tab.iterrows():
        try:
            if img.ndim == 2:
                crop = crop_at_2d(img, row["y"], row["x"], radius=crop_radius)
            elif img.ndim == 3:
                crop = crop_at_3d(img, row["z"], row["y"], row["x"], radius=crop_radius)
            else:
                raise RuntimeError(f"img.ndim {img.ndim} not supported")

            crops.append(crop)
        except AssertionError:
            pass  # happens if xy is too close to border

    return np.stack(crops)


class GaussianFit(object):
    def __init__(
        self,
        dims,
        covariance_type,
    ):
        self.ndim = len(dims)
        self.dims = dims
        self.covariance_type = covariance_type

        if self.ndim == 2:
            if covariance_type == "diagonal":
                self.fit_func = fit_gaussian_diagonal_2d
                self.func = gaussian_diagonal_2d
                self.fit_output_index = [-2, -1]
                self.fit_output_names = [f"{d}_std" for d in self.dims]

            elif covariance_type == "spherical":
                self.fit_func = fit_gaussian_spherical_2d
                self.func = gaussian_spherical_2d
                self.fit_output_index = [-1]
                self.fit_output_names = [f"{dims}_std"]
            else:
                raise RuntimeError("Gaussian type must be spherical or diagonal")

        elif self.ndim == 3:
            if covariance_type == "diagonal":
                self.fit_func = fit_gaussian_diagonal_3d
                self.func = gaussian_diagonal_3d
                self.fit_output_index = [-3, -2, -1]
                self.fit_output_names = [f"{d}_std" for d in self.dims]

            elif covariance_type == "spherical":
                self.fit_func = fit_gaussian_spherical_3d
                self.func = gaussian_spherical_3d
                self.fit_output_index = [-1]
                self.fit_output_names = [f"{dims}_std"]
            else:
                raise RuntimeError("Gaussian type must be spherical or diagonal")
        else:
            raise RuntimeError("Only 2d and 3d data supported")

    def fit(self, data):
        okay, param = self.fit_func(data)
        if okay:
            values = param[self.fit_output_index]
            keys = self.fit_output_names
            return dict([(k, v) for k, v in zip(keys, values)]), param

        return None, None


# %% [markdown]
# main run function


# %%
def run(
    fn,
    norm_percentiles=(0, 99.9),
    sigma=0.5,
    threshold_rel=0.33,
    peak_min_distance=7,
    crop_radius=5,
    covariance_type="diagonal",
    show=True,
):
    img = read_tif(fn)

    fn_base, _ = os.path.splitext(fn)

    if img.ndim == 2:
        dims = "yx"
    elif img.ndim == 3:
        dims = "zyx"
    else:
        raise RuntimeError("Only 2D and 3D data supported...")

    print(f" - data is {img.ndim}D with order {dims}")

    pixel_sizes_dct = read_imagej_voxel_size_zyx(fn)
    print(" - pixel sizes:")
    for k, v in pixel_sizes_dct.items():
        print(f"\t{k}: {v} micron")

    fitter = GaussianFit(dims, covariance_type)

    img_norm = normalize_intensity(img, norm_percentiles)

    img_filt, peaks_tab = find_peaks(
        img_norm,
        dims,
        sigma_gaussian=sigma,
        threshold_rel=threshold_rel,
        peak_min_distance=peak_min_distance,
    )

    if len(peaks_tab) == 0:
        print(" - No peaks found for given parameters, exiting...")
        return

    if show:
        img_show = img_norm

        if img.ndim == 3:
            img_show = img_show.max(0)

        f, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img_show, "gray")
        ax.plot(peaks_tab.x, peaks_tab.y, "ro", markerfacecolor="none")
        ax.set_title(f"Coarse peaks\n{os.path.basename(fn_base)}")

    result_tab = []
    crop_and_fits = []
    for i, row in tqdm(peaks_tab.iterrows(), total=len(peaks_tab)):
        center = {}
        for d in dims:
            center[d] = row[d]

        try:
            if img.ndim == 2:
                crop = crop_at_2d(img, **center, radius=crop_radius)
            elif img.ndim == 3:
                crop = crop_at_3d(img, **center, radius=crop_radius)
            else:
                raise RuntimeError(f"img.ndim {img.ndim} not supported")
        except AssertionError:
            continue

        res, param = fitter.fit(crop)
        if res is not None:
            for k, v in res.items():
                row[k] = v

            result_tab.append(row)

            crop_fit = fitter.func(*param)(*np.indices(crop.shape))

            axis = 0
            if img.ndim == 3:
                axis = 1
            crop_and_fits.append(np.stack([crop, crop_fit], axis=axis))

    
    if len(crop_and_fits) == 0:
        print(" - No Gaussians could be fitted to peaks, exiting...")
        return

    crop_and_fits = np.stack(crop_and_fits)

    tifffile.imwrite(
        fn_base + "_crops.tif",
        crop_and_fits.astype(np.float32),
        imagej=True,
        resolution=(1.0 / pixel_sizes_dct["x"], 1.0 / pixel_sizes_dct["y"]),
        metadata={
            "spacing": pixel_sizes_dct["z"],
            "unit": "mircon",
            "finterval": 1,
        },
    )

    result_tab = pd.DataFrame(result_tab).reset_index()
    print(f" - Gaussian fit converged for {len(result_tab)}/{len(peaks_tab)}")

    for col in result_tab.columns:
        if col.endswith("std"):
            d, _ = col.split("_")
            result_tab[col] = result_tab[col] * pixel_sizes_dct[col[0]]
            result_tab[d + "_fwhm"] = result_tab[col] * 2.355

    # napari
    for k, d in enumerate(dims):
        result_tab[f"axis-{k}"] = result_tab[d]

    # save
    result_tab.to_csv(f"{fn_base}_gaussian_fits.csv")

    if show:
        f, ax = plt.subplots()
        fwhm_cols = [c for c in result_tab.columns if c.endswith("_fwhm")]
        sns.boxenplot(data=result_tab[fwhm_cols], ax=ax)
        ax.set_ylabel("FWHM (microns)")
        ax.set_title(os.path.basename(fn_base))

    return result_tab


# %% [markdown]
# ## the argparser for it


# %%
def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Gaussian FWHM/STD for 2D/3D images")
    parser.add_argument(
        "input_tifs",
        type=str,
        help="Input tif file(s)",
        nargs="+",
    )

    parser.add_argument(
        "-np",
        "--norm_percentiles",
        type=float,
        nargs=2,
        help="Images are normalized to percentiles. Provide tuple for minimum and maximum (default: 0, 99.9)",
        default=(0, 99.9),
    )

    parser.add_argument(
        "-s",
        "--sigma",
        type=float,
        help="Images are smoothed with a Gaussian before peak detection. Gaussian sigma (in px)",
        default=0.5,
    )

    parser.add_argument(
        "-t",
        "--threshold_rel",
        type=float,
        help="Peak detection threshold relative to normalized maximum. See skimage.feature.peak_local_max for details.",
        default=0.33,
    )

    parser.add_argument(
        "-p",
        "--peak_min_distance",
        type=float,
        help="Only peaks with a minimum peak distance of peak_min_distance (in px) are considered for fitting",
        default=7,
    )

    parser.add_argument(
        "-cr",
        "--crop_radius",
        type=int,
        help="Around each peak the images is croped with a crop radius (in px)",
        default=5,
    )

    parser.add_argument(
        "-ct",
        "--covariance_type",
        type=str,
        help="Covariance type of the Gaussian fit: spherical (single sigma) or diagonal (one sigma per dimension)",
        default="diagonal",
    )

    parser.add_argument(
        "-sh",
        "--show",
        type=bool,
        help="Show FWHM plots at the end of the run",
        default=True,
    )

    return parser.parse_args()


def main():
    args = get_args()

    print("\nGaussian FWHM/STD for 2D/3D images")
    print("Parameters: ", "#" * 68)
    for arg in vars(args):
        print(f" {arg:28s}", getattr(args, arg))
    print("#" * 80)

    options = vars(args)
    input_tifs = options.pop("input_tifs")

    for fn in input_tifs:
        run(os.path.abspath(fn), **options)

    if options["show"]:
        plt.show()


if __name__ == "__main__":
    main()


# %% [markdown]
# stuff (to ignore)

# %%

# crops = crop_fitting_roi(img, peaks_tab, crop_radius=crop_radius)

# result_fits = []
# for ci in crop_img in crops:
#     okay, para = fit_func(crop)
#     if okay:
#         result_fits.append(para[-1])
#         continue

#         print(para[-1] * 2.355 * 0.03)

#         f, ax = plt.subplots(1, 2)
#         ax[0].imshow(crop)
#         ax[0].add_patch(Circle([para[2], para[1]], para[3], fill=False, color="r"))
#         print(para)

#         f = gaussian2d_iso(*para)
#         fit = f(*np.indices(crop.shape))
#         ax[1].imshow(fit)
#     else:
#         print("out")

# # if ((para[1:3] > 1) & (para[1:3] < crop.shape[0]-1)).all():
# #     cres.append(para)

#


# fn = "T:/BIF_StaffSci/Christoph/danzlgrp/Vitali/VIS/Data for 2D vessicle+golgi+pan/tifs/20231120_seedling3T_vha1-A594+sec21rb-STAR635P+NHS-A488_post-E...ackRescue100nm_STED_LS_xySTEDnoRescue_30nm_0201.msr - 640 {22}_vesicles.tif"
# fn = "../data/sample_2D_STED2.tif"
# tab = run(
#     fn,
#     covariance_type="diagonal",
#     sigma=0.5,
#     threshold_rel=0.33,
#     norm_percentiles=(0, 99.9),
#     dims="yx",
# )

# # %matplotlib widget
# fn = "../data/sample_volume_01_ps.tif"
# tab = run(
#     fn,
#     covariance_type="diagonal",
#     sigma=0.7,
#     threshold_rel=0.1,
#     norm_percentiles=(0, 99.99),
# )
