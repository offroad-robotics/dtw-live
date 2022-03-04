# Datasets module

## Description

The `datasets` module contains methods for collecting and manipulating trial data. Since most of our data is recorded 

There are a couple unique characteristics of the datasets used that necessitate a set of functions for properly handling raw input streams. Firstly, it is important to set up experiments such that **each input stream shares all gesture types required**, to allow for easy splitting of training and testing data. This also makes randomization easier for both experiments and extracting training sample data.

## Dataset formats

| File Format | Supported |
|-------------|-----------|
| json        | Yes       |
| pickle      | No        |
| csv         | No        |
| hdf5        | No        |

## Time series array format

The data shape used by functions in other `dtw_live` modules is:

`np.ndarray` of shape `(timesteps, ndim>=1)`

Loaded datasets are typically a list of the above to facilitate samples with differing numbers of `timesteps` (generally samples must have the same `ndim`).

## Usage

```python
from dtw_live.datasets import *

# load json formatted data found in the ./datasets directory.
paths = glob_files("./datasets", ftypes=[".json"])
data = load_data(*paths)

# filter data by header field (sensors) and label type (classes).
data_filt = filter_data(data, 
                        header_fields=["acc", "gyr.y"],
                        label_types=["wave_hand", "swipe_left"])

# get filtered header and target names for the loaded data, 
# i.e. ["acc.x", "acc.y", "acc.z", "gyr.y"] and 
#      ["swipe_gesture", "wave gesture"]
_, _, feature_names, target_names = data_filt

# get samples from training streams
X_train, y_train = get_samples(data_filt)

# or, get streams
X_test, y_test = get_streams(data_filt)
```
