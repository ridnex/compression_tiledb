# SZ3 Compression with Relative Error Bound

This project demonstrates how to compress a 3D scientific dataset using the SZ3
error-bounded lossy compressor and store both original and compressed data in TileDB.

---

## Requirements

- Python 3.8 or higher
- numpy
- tiledb
- pysz
- os
- shutil

Install lib using:

```bash
pip install numpy tiledb pysz
```

## Inputs
```
INPUT_FILE = "Redsea_t2_4k_gan.dat"
ARRAY_D_NAME = "arrayD" 
ARRAY_G_NAME = "arrayG" 
SHAPE = (4000, 855, 1215)
DTYPE = np.float32
EPSILON = 1e-2 
```




## Different Algorithms


```
config = szConfig()
config.errorBoundMode = szErrorBoundMode.REL
config.relErrorBound = EPSILON

# --- ADD THIS LINE TO CHANGE THE ALGORITHM ---
# Default is INTERP_LORENZO (best quality/ratio trade-off)
# To try other algorithms, uncomment one of the following:

# config.cmprAlgo = szAlgorithm.INTERP        # Interpolation only
# config.cmprAlgo = szAlgorithm.LORENZO_REG   # Lorenzo/regression
```

## run Python code:  compress.py 


## Jupyter file code_res.ipynb and SZmain.ipynb contains all printed results in report 
