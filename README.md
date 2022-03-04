# Real-Time Dynamic Time Warping Library

Dynamic Time Warping (DTW) package for real-time recognition. For further information, see [documentation](docs/).

## Installing

### From repo

This method may take a while due to GitLab latency

```bash
git clone git@code.engineering.queensu.ca:13jvt/dtw_live.git
pip install git+ssh://git@code.engineering.queensu.ca/13jvt/dtw_live.git
```

### From source

```bash
python setup.py build
python setup.py install
```

If you just want to build the `dtwlib.so` shared library, a MakeFile is included in `dtw_live/dtw_c`.

## Testing

Helpful for testing our C library, lots of issues can arise when passing n-dimensional numpy array pointers to our `cost_matrix` functions. **Update:** Lots of type-checking has been put in place to prevent known issues.

To run all tests (requires `pytest >= 6.2.4`), run:

```bash
py.test
```