# webdataset-resharder
Efficiently process webdatasets

## Selecting samples in the FixedPools track

Before training, you will need to select the subset of samples you wish to use. Given a set of chosen samples, we create new shards with only those samples, which the train
ing code then consumes.

Each sample in our pool has a unique identifier, which is present in the metadata parquets, and in the `json` files inside the `.tar` shards.

The format describing the subset of samples should be a numpy array of dtype `numpy.dtype("u8,u8")` (i.e. a [structured array](https://numpy.org/doc/stable/user/basics.rec.html) of pairs of unsigned 64-bit integers), with shape `(subset_size,)`, containing a list of `uid`s (128-bit hashes from the parquet files) in *lexicographic sorted order*, saved to disk in either [`npy` format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) or [memory-mapped format](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).

For instance, if you have a list of uids `uids = ['139e4a9b22a614771f06c700a8ebe150', '6e356964a967af455c8016b75d691203']`, you can store them by running the following python code:

```
processed_uids = np.array([(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids], np.dtype("u8,u8"))
processed_uids.sort()
np.save(out_filename, processed_uids)
```

After creating a subset, you may invoke the resharder to build the subset shards in `$output_dir` like so:

```
python resharder.py -i $download_dir -o $output_dir -s $subset_file
```

If desired, the resharder can be run in parallel on multiple nodes. The easiest way to do so is to split the input directory into smaller subfolders with fewer shards, and run separate resharder jobs for each of them, each with to separate output directories.
