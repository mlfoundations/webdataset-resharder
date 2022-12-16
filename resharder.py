#!/usr/bin/env python3

import time
import re
import multiprocessing as mp
import shutil
import json
import os
import contextlib
import argparse
import bisect

from pathlib import Path
from cloudpathlib import CloudPath
from dataclasses import dataclass
from typing import List, Optional, Callable

import numpy as np
import tqdm
import simdjson
import webdataset as wds

# Monkey-patch webdataset to support S3 via aws s3

def gopen_aws(url, mode="rb", bufsize=8192):
    """Open a URL with `aws s3`.
    :param url: url (usually, s3:// etc.)
    :param mode: file mode
    :param bufsize: buffer size
    """
    # TODO not sure about ignore_status
    if mode[0] == "r":
        cmd = f"aws s3 cp '{url}' -"
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141, 23],
        )
    elif mode[0] == "w":
        cmd = f"aws s3 cp - '{url}'"
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141, 26],
        )
    else:
        raise ValueError(f"{mode}: unknown mode")

wds.gopen_schemes.setdefault('s3', gopen_aws)


class ShardWriter:
    """Like TarWriter but splits into multiple shards."""

    def __init__(
        self,
        pattern: str,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Optional[Callable] = None,
        start_shard: int = 0,
        **kw,
    ):
        """Create a ShardWriter.
        :param pattern: output file pattern
        :param maxcount: maximum number of records per shard (Default value = 100000)
        :param maxsize: maximum size of each shard (Default value = 3e9)
        :param kw: other options passed to TarWriter
        """
        self.verbose = 1
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.post = post

        self.tarstream = None
        self.shard = start_shard
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None
        self.stream = None
        self.next_stream()

    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()
        self.fname = self.pattern % self.shard
        # if self.verbose:
        #     print(
        #         "# writing",
        #         self.fname,
        #         self.count,
        #         "%.1f GB" % (self.size / 1e9),
        #         self.total,
        #     )

        self.shard += 1

        self.stream = None
        self.tarstream = wds.TarWriter(self.fname, **self.kw)

        self.count = 0
        self.size = 0

    def write(self, obj):
        """Write a sample.
        :param obj: sample to be written
        """
        if self.tarstream is None or self.count >= self.maxcount or self.size >= self.maxsize:
            self.next_stream()
        size = self.tarstream.write(obj)
        self.count += 1
        self.total += 1
        self.size += size

    def finish(self):
        """Finish all writing (use close instead)."""
        if self.tarstream is not None:
            self.tarstream.close()
            assert self.fname is not None
            if callable(self.post):
                self.post(self.fname)
            self.tarstream = None
        if self.stream is not None:
            self.stream.close()

    def close(self):
        """Close the stream."""
        self.finish()
        del self.tarstream
        del self.shard
        del self.count
        del self.size

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, *args, **kw):
        """Exit context."""
        self.close()



@dataclass
class Shard:
    shard_id: int
    data_start: int
    size: int


@dataclass
class WorkerTask:
    worker_id: int
    shards: List[Shard]


u16 = np.dtype("u8,u8")


def ceildiv(a, b):
    return -(-a // b)


def path_or_cloudpath(s):
    if re.match(r"^\w+://", s):
        return CloudPath(s)
    return Path(s)

def make_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=path_or_cloudpath,
        required=True,
        help="input directory containing a webdataset",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, required=True, help="output directory"
    )
    parser.add_argument(
        "-s",
        "--subset-file",
        type=path_or_cloudpath,
        required=True,
        help="subset file, either a NumPy or memmap array of 128 bit hashes",
    )
    parser.add_argument(
        "-n",
        "--num-shards",
        type=int,
        help="number of shards to process (beware of off-by-ones)",
    )
    parser.add_argument(
        "--first-shard",
        type=int,
        default=0,
        help="index of first shard to process",
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        default=mp.cpu_count(),
        type=int,
        help="number of workers to use",
    )
    parser.add_argument(
        "--shard-size",
        default=10000,
        type=int,
        help="maximum number of examples per output shard",
    )
    parser.add_argument(
        "--shard-format",
        default="{:05d}.tar",
        type=str,
        help="format for each shard in str.format syntax",
    )
    parser.add_argument(
        "--shard-stats-format",
        default="{:05d}_stats.json",
        type=str,
        help="format for each shard stats file in str.format syntax",
    )
    parser.add_argument(
        "--shard-table",
        default="sizes.json",
        type=path_or_cloudpath,
        help="JSON file recording input shard sizes relative to INPUT_DIR",
    )
    parser.add_argument(
        "--shuffle-bufsize", default=100000, type=int, help="buffer size for shuffling"
    )
    return parser


parser = make_argparser()


def guess_num_shards(
    *,
    input_dir: Path,
    first_shard: int = parser.get_default("first_shard"),
    shard_format: str = parser.get_default("shard_format"),
    shard_stats_format: str = parser.get_default("shard_stats_format"),
    **_,
):
    n = 1

    def test_size(i):
        shard = input_dir / shard_format.format(first_shard + i)
        shard_stats = input_dir / shard_stats_format.format(first_shard + i)
        return shard.exists() and shard_stats.exists()

    if not test_size(0):
        raise RuntimeError("Did not find any shards")

    for _ in range(40):
        if not test_size(n):
            break
        n *= 2
    else:
        raise RuntimeError(f"Found too many shards (at least {n})")

    n = n // 2 + bisect.bisect_right(
        range(n // 2, n), False, key=lambda i: not test_size(i)
    )

    return n


def load_shard_metadata(
    *,
    input_dir: Path,
    num_shards: int = parser.get_default("num_shards"),
    first_shard: int = parser.get_default("first_shard"),
    shard_format: str = parser.get_default("shard_format"),
    shard_stats_format: str = parser.get_default("shard_stats_format"),
    shard_table: Path = parser.get_default("shard_table"),
    **_,
):
    shards = []
    offset = 0
    parser = simdjson.Parser()

    table = {}
    shard_table_path = input_dir / shard_table
    if shard_table_path.exists():
        print(f"loading shard table {shard_table_path}")
        with open(shard_table_path, "rb") as f:
            table = json.load(f)
            print(f"shard table has size {len(table)}")

    if not num_shards and not table:
        num_shards = guess_num_shards(
            input_dir=input_dir,
            first_shard=first_shard,
            shard_format=shard_format,
            shard_stats_format=shard_stats_format,
        )

    if not num_shards:
        num_shards = len(table) - first_shard

    shard_ids = range(first_shard, first_shard + num_shards)

    for shard_id in tqdm.tqdm(shard_ids):
        size_path = input_dir / shard_stats_format.format(shard_id)
        shard_name = shard_format.format(shard_id)
        shard_path = input_dir / shard_name

        if shard_name in table:
            size = table[shard_name]
            shards.append(Shard(shard_id, offset, size))
            offset += size

        elif size_path.exists() and shard_path.exists():
            with size_path.open("r") as f:
                size = int(simdjson.Parser().parse(f.read()).get("successes"))
            shards.append(Shard(shard_id, offset, size))
            offset += size

    total_data = shards[-1].data_start + shards[-1].size
    print(f"found a total of {len(shards)} shards with {total_data} examples")

    return shards, total_data


def load_subset(*, subset_file: Path, **_):
    # Detect the NumPy format magic string
    if open(subset_file, "rb").read(6) == b"\x93NUMPY":
        subset = np.load(subset_file, mmap_mode="r")
        assert subset.dtype == u16

    else:
        subset = np.memmap(subset_file, u16, mode="r+")

    # print(f"selecting a subset of {len(subset)} examples")
    return subset


def plan_tasks(shards: List[Shard], /, **args):
    num_workers = args["num_workers"]
    worker_tasks = []
    total_data = shards[-1].data_start + shards[-1].size

    # evenly distribute data to workers
    data_starts = [shard.data_start for shard in shards]
    shard_chunks = [
        np.searchsorted(data_starts, i, side="right")
        for i in range(0, total_data, -(-total_data // num_workers))
    ]
    shard_chunks.append(len(shards))

    for worker_id, (shard_start, shard_end) in enumerate(
        zip(shard_chunks, shard_chunks[1:])
    ):
        if shard_start == shard_end:
            continue
        first_shard, last_shard = shards[shard_start], shards[shard_end - 1]

        first_index = first_shard.data_start
        last_index = last_shard.data_start + last_shard.size - 1

        print(
            f"worker {worker_id:03d} will process shards {shard_start} to {shard_end-1}"
        )
        worker_tasks.append(
            WorkerTask(
                worker_id,
                shards[shard_start:shard_end],
            )
        )

    return worker_tasks


def copy_worker(
    state,
    lock,
    task: WorkerTask,
    *,
    input_dir: Path,
    output_dir: Path,
    subset_file: Path,
    shard_format: str = parser.get_default("shard_format"),
    shard_size: int = parser.get_default("shard_size"),
    shuffle_bufsize: int = parser.get_default("shuffle_bufsize"),
    **_,
):
    # print(task.worker_id, task.shards[0], task.shards[-1])

    subset = load_subset(subset_file=subset_file)
    ds = wds.WebDataset(
        [str(input_dir / shard_format.format(shard.shard_id)) for shard in task.shards]
    )
    # disgusting hack to prevent ShardWriter from printing
    # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    sw = ShardWriter(
        str(output_dir / f"shard_{task.worker_id:04d}_%07d.tar"),
        maxcount=shard_size,
    )

    sw.verbose = False

    total_data = (
        task.shards[-1].data_start + task.shards[-1].size - task.shards[0].data_start
    )
    with lock:
        bar = tqdm.tqdm(
            desc=f"worker {task.worker_id:03d}",
            total=total_data,
            position=task.worker_id,
            leave=False,
        )

    processed_count, output_count = 0, 0

    def subset_iter():
        nonlocal processed_count
        nonlocal output_count
        parser = simdjson.Parser()

        for i, d in enumerate(ds):
            key_str = parser.parse(d["json"]).get("uid")
            key_u16 = np.array([(int(key_str[:16], 16), int(key_str[16:32], 16))], u16)[
                0
            ]

            a = np.searchsorted(subset, key_u16, "left")
            b = np.searchsorted(subset, key_u16, "right")
            count = b - a

            for j in range(count):
                yield {**d, "__key__": f"{key_str}-{j}"}
                output_count += 1

            processed_count += 1

            if i % 1000 == 0 and i > 0:
                with lock:
                    bar.update(1000)

    it = subset_iter()
    if shuffle_bufsize > 0:
        it = wds.filters._shuffle(it, shuffle_bufsize, shuffle_bufsize)

    for d in it:
        sw.write(d)

    with lock:
        bar.update(total_data - bar.n)
        bar.close()
        state["processed_count"] += processed_count
        state["output_count"] += output_count


def do_tasks(worker_tasks, args):
    manager = mp.Manager()

    state = manager.dict()
    state["processed_count"] = 0
    state["output_count"] = 0

    lock = manager.Lock()

    processes = [
        mp.Process(target=copy_worker, args=(state, lock, task), kwargs=args)
        for task in worker_tasks
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    return state


def rmtree_contents(path: Path):
    for path in path.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def postprocess_output(*, output_dir, shard_format, **_):
    print("postprocessing output shards")
    for i, shard in enumerate(sorted(output_dir.iterdir())):
        shard.rename(output_dir / shard_format.format(i))


def main(args):
    shards, total_data = load_shard_metadata(**vars(args))
    if len(shards) < args.num_workers:
        args.num_workers = len(shards)

    subset = load_subset(**vars(args))
    print(f"selecting a subset of {len(subset)} examples")

    worker_tasks = plan_tasks(shards, **vars(args))

    rmtree_contents(args.output_dir)

    start_time = time.perf_counter()
    state = do_tasks(worker_tasks, vars(args))
    elapsed_time = time.perf_counter() - start_time

    print()

    # postprocess_output(**vars(args))

    print(
        f"processed {total_data} images in {elapsed_time:.3f}s ({total_data/elapsed_time:.2f} images/sec)"
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
