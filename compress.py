import numpy as np
import tiledb
import os
import shutil
from pysz import sz, szConfig, szErrorBoundMode

INPUT_FILE = "Redsea_t2_4k_gan.dat"
ARRAY_D_NAME = "arrayD" 
ARRAY_G_NAME = "arrayG" 

SHAPE = (4000, 855, 1215)
DTYPE = np.float32
EPSILON = 1e-2 

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        data_d = np.fromfile(INPUT_FILE, dtype=DTYPE).reshape(SHAPE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please download the dataset[cite: 90].")
        return

    d_max = data_d.max()
    d_min = data_d.min()
    v_range = d_max - d_min
    print(f"Data Range: {v_range} (Min: {d_min}, Max: {d_max})")

    print("Storing original data D in TileDB...")
    if os.path.exists(ARRAY_D_NAME):
        shutil.rmtree(ARRAY_D_NAME)
    
    dom_d = tiledb.Domain(
        tiledb.Dim(name="time", domain=(0, SHAPE[0]-1), tile=1, dtype=np.int32),
        tiledb.Dim(name="x", domain=(0, SHAPE[1]-1), tile=855, dtype=np.int32),
        tiledb.Dim(name="y", domain=(0, SHAPE[2]-1), tile=1215, dtype=np.int32)
    )
    schema_d = tiledb.ArraySchema(domain=dom_d, sparse=False, attrs=[tiledb.Attr(name="temp", dtype=DTYPE)])
    tiledb.DenseArray.create(ARRAY_D_NAME, schema_d)
    
    with tiledb.DenseArray(ARRAY_D_NAME, mode='w') as A:
        A[:] = data_d

    print(f"Compressing with SZ3 (Relative Error = {EPSILON})...")
    
    config = szConfig()
    config.errorBoundMode = szErrorBoundMode.REL
    config.relErrorBound = EPSILON 
    
    compressed_bytes, ratio_sz_internal = sz.compress(data_d, config)
    
    print(f"Internal SZ Ratio: {ratio_sz_internal:.2f}")

    print("Storing compressed data G in TileDB...")
    if os.path.exists(ARRAY_G_NAME):
        shutil.rmtree(ARRAY_G_NAME)

    comp_size = compressed_bytes.size
    dom_g = tiledb.Domain(tiledb.Dim(name="index", domain=(0, comp_size-1), tile=comp_size, dtype=np.int32))
    schema_g = tiledb.ArraySchema(domain=dom_g, sparse=False, attrs=[tiledb.Attr(name="bytes", dtype=np.uint8)])
    tiledb.DenseArray.create(ARRAY_G_NAME, schema_g)
    
    with tiledb.DenseArray(ARRAY_G_NAME, mode='w') as A:
        A[:] = compressed_bytes

    size_D_folder = get_folder_size(ARRAY_D_NAME)
    size_G_folder = get_folder_size(ARRAY_G_NAME)
    
    rho = size_D_folder / size_G_folder
    print("-" * 30)
    print(f"Size of Array D (disk): {size_D_folder / (1024**3):.2f} GB")
    print(f"Size of Array G (disk): {size_G_folder / (1024**3):.2f} GB")
    print(f"Final Compression Ratio (rho): {rho:.4f}")
    print("-" * 30)

    print("Verifying Decompression and Error Bounds...")
    
    with tiledb.DenseArray(ARRAY_G_NAME, mode='r') as A:
        read_bytes = A[:]['bytes']
    
    decompressed_data, dec_config = sz.decompress(read_bytes, DTYPE,  SHAPE)
    
    diff = np.abs(data_d - decompressed_data)
    max_pointwise_diff = diff.max()
    
    actual_max_rel_error = max_pointwise_diff / v_range
    
    print(f"Max Absolute Error: {max_pointwise_diff}")
    print(f"Max Relative Error (calc): {actual_max_rel_error}")
    print(f"Target Epsilon: {EPSILON}")
    
    if actual_max_rel_error <= EPSILON + 1e-9: # small buffer for float precision
        print("SUCCESS: Error bound satisfied Eq. 2!")
    else:
        print("WARNING: Error bound NOT satisfied.")
        
if __name__ == "__main__":
    main()