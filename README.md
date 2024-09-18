# Real-Time Dynamic Time Warping Library

Dynamic Time Warping (DTW) package for real-time recognition. For further information, see [documentation](docs/).

## Installing

### From source

```bash
python setup.py build
python setup.py install
```

If you want to build the `dtwlib.so` shared library, a MakeFile is included in `dtw_live/dtw_c`.

## Testing

Helpful for testing our C library, lots of issues can arise when passing n-dimensional numpy array pointers to our `cost_matrix` functions. **Update:** Lots of type-checking has been put in place to prevent known issues.

To run all tests (requires `pytest >= 6.2.4`), run:

```bash
py.test
```
