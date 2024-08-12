# Gaussian Fit FWHM

## Install
```bash
pip install git+https://git.ista.ac.at/csommer/gaussian_fit_fwhm.git
```

## Usage 
Installing will create a command line script `gaussian_fit_fwhm`:

```
usage: gaussian_fit_fwhm [-h] [-np NORM_PERCENTILES NORM_PERCENTILES] [-s SIGMA] [-t THRESHOLD_REL] [-p PEAK_MIN_DISTANCE] [-cr CROP_RADIUS] [-ct COVARIANCE_TYPE] [-d DIMS] [-sh SHOW] input_tifs [input_tifs ...]

Gaussian FWHM/STD for 2D/3D images

positional arguments:
  input_tifs            Input tif file(s)

optional arguments:
  -h, --help            show this help message and exit
  -np NORM_PERCENTILES NORM_PERCENTILES, --norm_percentiles NORM_PERCENTILES NORM_PERCENTILES
                        Images are normalized to percentiles. Provide tuple for minimum and maximum (default: 0, 99.9)
  -s SIGMA, --sigma SIGMA
                        Images are smoothed with a Gaussian before peak detection. Gaussian sigma (in px)
  -t THRESHOLD_REL, --threshold_rel THRESHOLD_REL
                        Peak detection threshold relative to normalized maximum. See skimage.feature.peak_local_max for details.
  -p PEAK_MIN_DISTANCE, --peak_min_distance PEAK_MIN_DISTANCE
                        Only peaks with a minimum peak distance of peak_min_distance (in px) are considered for fitting
  -cr CROP_RADIUS, --crop_radius CROP_RADIUS
                        Around each peak the images is croped with a crop radius (in px)
  -ct COVARIANCE_TYPE, --covariance_type COVARIANCE_TYPE
                        Covariance type of the Gaussian fit: spherical (single sigma) or diagonal (one sigma per dimension)
  -sh SHOW, --show SHOW
                        Show FWHM plots at the end of the run
```

## Output
For an input tif file called `my_input.tif` the script will generate:

```
├── my_input.tif
├── my_input_crops.tif
└── my_input_gaussian_fits.csv
```

