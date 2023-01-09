#!/usr/bin/env python3

import argparse
import bisect
import copy
import logging
import multiprocessing as mp
import os
import queue
import re
import shutil
import tempfile
import threading
import time

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Callable, Union, Dict
from multiprocessing.managers import NamespaceProxy, AcquirerProxy

import cv2
import numpy as np
import pandas as pd
import simdjson
import tqdm
import webdataset as wds

from cloudpathlib import CloudPath
from img2dataset.blurrer import BoundingBoxBlurrer


Pipe = wds.writer.gopen.Pipe
Pathy = Union[Path, CloudPath]


class ColoredConsoleHandler(logging.Handler):
    # TODO: Abstract ANSI color escapes
    def __init__(self, sub_handler=None):
        super().__init__()
        self.sub_handler = (
            logging.StreamHandler() if sub_handler is None else sub_handler
        )

    def emit(self, record):
        # Need to make a actual copy of the record
        # to prevent altering the message for other loggers
        myrecord = copy.copy(record)
        levelno = myrecord.levelno

        # NOTSET and anything else
        color = "\x1b[0m"  # normal
        tag = "NOTSET"

        if levelno >= logging.FATAL:
            color = "\x1b[31m"  # red
            tag = "FATAL"
        elif levelno >= logging.ERROR:
            color = "\x1b[31m"  # red
            tag = "ERROR"
        elif levelno >= logging.WARNING:
            color = "\x1b[33m"  # yellow
            tag = "WARN"
        elif levelno >= logging.INFO:
            color = "\x1b[32m"  # green
            tag = "INFO"
        elif levelno >= logging.DEBUG:
            color = "\x1b[35m"  # pink
            tag = "DEBUG"

        myrecord.msg = f"{color}[{tag}]\x1b[0m {myrecord.msg}"
        self.sub_handler.emit(myrecord)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class MultiProcessingHandler(logging.Handler):
    def __init__(self, name, queue):
        super().__init__()
        self.name = name
        self.queue = queue

    def _format_record(self, record):
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            self.format(record)
            record.exc_info = None

        record.msg = f"[{self.name}] {record.msg}"

        return record

    def emit(self, record):
        record = self._format_record(record)
        self.queue.put_nowait(record)


def setup_process_logging(log_queue, worker_id):
    logger = logging.getLogger("resharder")
    for handler in logger.handlers:
        logger.removeHandler(handler)

    logger.addHandler(MultiProcessingHandler(f"worker {worker_id:03d}", log_queue))
    return logger


logger = logging.getLogger("resharder")
logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
logger.addHandler(ColoredConsoleHandler(log_handler))
log_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))


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


wds.gopen_schemes.setdefault("s3", gopen_aws)


class ShardWriter:
    """Like TarWriter but splits into multiple shards."""

    def __init__(
        self,
        namer: Callable,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Optional[Callable] = None,
        start_shard: int = 0,
        logger: Optional[logging.Logger] = None,
        **kw,
    ):
        """Create a ShardWriter.
        :param namer: function mapping shard number to output file name
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
        self.namer = namer
        self.logger = logger
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None
        self.stream = None

    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()
        self.fname = self.namer(self.shard)

        self.shard += 1

        self.stream = None
        self.tarstream = wds.TarWriter(self.fname, **self.kw)

        self.count = 0
        self.size = 0

    def write(self, obj):
        """Write a sample.
        :param obj: sample to be written
        """
        if (
            self.tarstream is None
            or self.count >= self.maxcount
            or self.size >= self.maxsize
        ):
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
                self.post(fname=self.fname, count=self.count, size=self.size)
            self.tarstream = None

            self.logger.debug(
                f"wrote {self.fname} {self.size / 1e9:.1f} GB, {self.count}/{self.total}"
            )

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
    parquets: Optional[List[str]]


u16 = np.dtype("u8,u8")


def ceildiv(a, b):
    return -(-a // b)


def path_or_cloudpath(s: str) -> Pathy:
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
        "-o",
        "--output-dir",
        type=path_or_cloudpath,
        required=True,
        help="output directory",
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
        default="{:08d}.tar",
        type=str,
        help="format for each input shard in str.format syntax",
    )
    parser.add_argument(
        "--output-shard-format",
        default="{:08d}.tar",
        type=str,
        help="format for each output shard in str.format syntax",
    )
    parser.add_argument(
        "--shard-stats-format",
        default="{:08d}_stats.json",
        type=str,
        help="format for each input shard stats file in str.format syntax",
    )
    parser.add_argument(
        "--output-shard-stats-format",
        default="{:08d}_stats.json",
        type=str,
        help="format for each output shard stats file in str.format syntax",
    )
    parser.add_argument(
        "--shard-table",
        default="sizes.json",
        type=path_or_cloudpath,
        help="JSON file recording input shard sizes relative to INPUT_DIR",
    )
    parser.add_argument(
        "--write-shard-table",
        action="store_true",
        help="write shard table to output_dir if it does not exist",
    )
    parser.add_argument(
        "--shuffle-bufsize", default=100000, type=int, help="buffer size for shuffling"
    )
    parser.add_argument(
        "--is-master",
        action="store_true",
        default=True,
        help="for multi-node processing, indicate whether the current script is the master",
    )
    parser.add_argument(
        "--blur-metadata-map",
        type=path_or_cloudpath,
        default=None,
        help="Map file from shards to parquets for blurring.",
    )
    parser.add_argument(
        "--apply-blur",
        action="store_true",
        help="Apply blurring to images and re-encode them",
    )
    parser.add_argument(
        "--inject-blur-metadata",
        action="store_true",
        help="Add blur bounding boxes to the json field of the output examples",
    )
    parser.add_argument(
        "--reencode-webp-quality",
        type=str,
        default=100,
        help="Quality for re-encoding images if necessary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not make any changes to the output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="append_const",
        const=1,
        help="decrease the logging level",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="append_const",
        const=1,
        help="increase the logging level",
    )
    return parser


parser = make_argparser()


def guess_num_shards(
    *,
    input_dir: Pathy,
    first_shard: int = parser.get_default("first_shard"),
    shard_format: str = parser.get_default("shard_format"),
    **_,
):
    n = 1

    def test_size(i):
        shard = input_dir / shard_format.format(first_shard + i - 1)
        return shard.exists()

    for _ in range(40):
        if not test_size(n):
            break
        n *= 2
    else:
        raise RuntimeError(f"Found too many shards (at least {n})")

    if n == 1:
        raise RuntimeError("Did not find any shards")

    n = (
        n // 2
        + bisect.bisect_right(range(n // 2, n), False, key=lambda i: not test_size(i))
        - 1
    )

    return n


def load_shard_size(args):
    shard_id, input_dir, shard_format, shard_stats_format = args
    size_path = input_dir / shard_stats_format.format(shard_id)
    shard_name = shard_format.format(shard_id)
    shard_path = input_dir / shard_name
    size = None
    if size_path.exists() and shard_path.exists():
        with size_path.open("r") as f:
            size = int(simdjson.Parser().parse(f.read()).get("successes"))
    return shard_name, size


def load_shard_metadata(
    *,
    input_dir: Pathy,
    num_shards: int = parser.get_default("num_shards"),
    first_shard: int = parser.get_default("first_shard"),
    shard_format: str = parser.get_default("shard_format"),
    shard_stats_format: str = parser.get_default("shard_stats_format"),
    shard_table: Pathy = parser.get_default("shard_table"),
    write_shard_table: bool = parser.get_default("write_shard_table"),
    num_workers: int = parser.get_default("num_workers"),
    **_,
):
    shards = []
    offset = 0
    parser = simdjson.Parser()

    table = {}
    shard_table_path = input_dir / shard_table
    if shard_table_path.exists():
        logger.info(f"loading shard table {shard_table_path}")
        with open(shard_table_path, "rb") as f:
            try:
                table = simdjson.load(f)
            except ValueError as e:
                logger.error(f"shard table parsing error: {e.args[0]}")
            logger.info(f"shard table has size {len(table)}")

    if not num_shards and not table:
        num_shards = guess_num_shards(
            input_dir=input_dir,
            first_shard=first_shard,
            shard_format=shard_format,
        )
        logger.info(f"binary search found {num_shards} potential shards")

    if not num_shards:
        num_shards = len(table) - first_shard

    shard_ids = range(first_shard, first_shard + num_shards)
    pool = mp.Pool(num_workers)
    size_iter = pool.imap(
        load_shard_size,
        (
            (
                shard_id,
                input_dir,
                shard_format,
                shard_stats_format,
            )
            for shard_id in tqdm.tqdm(shard_ids)
        ),
        chunksize=16,
    )

    for shard_name, size in size_iter:
        if size is not None:
            table[shard_name] = size

    for shard_id in shard_ids:
        size_path = input_dir / shard_stats_format.format(shard_id)
        shard_name = shard_format.format(shard_id)
        shard_path = input_dir / shard_name

        if shard_name in table:
            size = table[shard_name]
            shards.append(Shard(shard_id, offset, size))
            offset += size
        else:
            logger.warning(f"missing shard {shard_name}")

    total_data = shards[-1].data_start + shards[-1].size
    logger.info(f"found a total of {len(shards)} shards with {total_data} examples")

    if write_shard_table and not shard_table_path.exists():
        logger.info("writing shard table")
        with shard_table_path.open("w") as f:
            simdjson.dump(table, f)

    return shards, total_data


def load_subset(*, subset_file: Path, **_):
    assert not isinstance(subset_file, CloudPath)

    with open(subset_file, "rb") as f:
        # Detect the NumPy format magic string
        if f.read(6) == b"\x93NUMPY":
            subset = np.load(subset_file, mmap_mode="r")
            assert subset.dtype == u16

        else:
            subset = np.memmap(subset_file, u16, mode="r+")

    return subset


def load_parquet_metadata(
    shards: List[Shard],
    /,
    blur_metadata_map: Optional[Pathy] = parser.get_default("blur_metadata_map"),
    shard_format: str = parser.get_default("shard_format"),
    **_,
):
    if blur_metadata_map is None:
        return None

    with blur_metadata_map.open("r") as f:
        parquets = simdjson.load(f)

    parquet_table = {}

    # invert the parquet → shard multi-map
    for pq in parquets.values():
        for shard in pq["shards"]:
            parquet_table[path_or_cloudpath(shard).name] = pq["parquet"]

    parquet_list = []
    for shard in shards:
        shard_name = shard_format.format(shard.shard_id)
        parquet_list.append(parquet_table.get(shard_name))
        if parquet_list[-1] is None:
            logger.warning(f"could not find parquet for shard {shard_name}")

    return parquet_list


def plan_tasks(shards: List[Shard], parquets: Optional[List[str]] = None, /, **args):
    num_workers = args["num_workers"]
    worker_tasks = []
    total_data = shards[-1].data_start + shards[-1].size

    # evenly distribute data to workers
    data_starts = [shard.data_start for shard in shards]
    shard_chunks = [
        np.searchsorted(data_starts, i, side="left")
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

        worker_parquets = (
            parquets[shard_start:shard_end] if parquets is not None else None
        )

        logger.debug(
            f"worker {worker_id:03d} will process shards {shard_start} to {shard_end-1}"
        )
        worker_tasks.append(
            WorkerTask(worker_id, shards[shard_start:shard_end], worker_parquets)
        )

    return worker_tasks


def blur_image(
    blurrer,
    jpg,
    blur_bboxes,
    reencode_webp_quality=parser.get_default("reencode_webp_quality"),
):
    img_buf = np.frombuffer(jpg, np.uint8)
    decoded = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
    blurred = blurrer(decoded, blur_bboxes)
    encoded = cv2.imencode(
        ".webp",
        blurred,
        params=[int(cv2.IMWRITE_WEBP_QUALITY), reencode_webp_quality],
    )[1].tobytes()
    return encoded


def copy_worker(
    task: WorkerTask,
    state: NamespaceProxy,
    lock: AcquirerProxy,
    log_queue,
    *,
    input_dir: Pathy,
    output_dir: Pathy,
    subset_file: Path,
    shard_format: str = parser.get_default("shard_format"),
    output_shard_format: str = parser.get_default("output_shard_format"),
    output_shard_stats_format: str = parser.get_default("output_shard_stats_format"),
    shard_size: int = parser.get_default("shard_size"),
    shuffle_bufsize: int = parser.get_default("shuffle_bufsize"),
    reencode_webp_quality: int = parser.get_default("reencode_webp_quality"),
    apply_blur: bool = parser.get_default("apply_blur"),
    inject_blur_metadata: bool = parser.get_default("inject_blur_metadata"),
    dry_run: bool = parser.get_default("dry_run"),
    **_,
):
    logger = setup_process_logging(log_queue, task.worker_id)

    subset = load_subset(subset_file=subset_file)
    ds = wds.WebDataset(
        [str(input_dir / shard_format.format(shard.shard_id)) for shard in task.shards],
        handler=lambda exn: logger.error(repr(exn)),
    )

    # create shard_name → parquet_name mapping
    assert task.parquets is None or len(task.shards) == len(task.parquets)
    parquet_table = (
        {
            shard_format.format(shard.shard_id): parquet
            for shard, parquet in zip(task.shards, task.parquets)
        }
        if task.parquets is not None
        else {}
    )

    @lru_cache(1)
    def load_parquet(fname):
        try:
            logger.debug(f"loading parquet {fname}")
            with path_or_cloudpath(fname).open("rb") as f:
                return pd.read_parquet(f).set_index("uid")["face_bboxes"]
        except FileNotFoundError:
            return None

    def load_blur_bboxes(url, uid):
        fname = parquet_table.get(path_or_cloudpath(url).name)
        if fname is not None:
            parquets = load_parquet(fname)
            if parquets is None:
                logger.error(f"failed to find parquet for {url}")
            if parquets is not None:
                return parquets.get(uid)

    output_shard_index = None

    def output_shard_namer(_shard):
        nonlocal output_shard_index
        with lock:
            output_shard_index = state.output_shard_count
            state.output_shard_count += 1

        return str(output_dir / output_shard_format.format(output_shard_index))

    def output_shard_size_writer(count, **_):
        with (output_dir / output_shard_stats_format.format(output_shard_index)).open(
            "w"
        ) as f:
            simdjson.dump({"successes": count}, f)

    sw = ShardWriter(
        output_shard_namer,
        maxcount=shard_size,
        logger=logger,
        post=output_shard_size_writer,
    )

    sw.verbose = False

    total_data = (
        task.shards[-1].data_start + task.shards[-1].size - task.shards[0].data_start
    )

    processed_count, output_count, blur_count = 0, 0, 0

    def subset_iter():
        nonlocal processed_count, output_count, blur_count
        parser = simdjson.Parser()
        blurrer = BoundingBoxBlurrer()

        for i, d in enumerate(ds):
            json_parsed = parser.parse(d["json"])
            key_str = json_parsed.get("uid")
            # TODO: is this really the best way to get a u16 scalar?
            key_u16 = np.array([(int(key_str[:16], 16), int(key_str[16:32], 16))], u16)[
                0
            ]

            a = np.searchsorted(subset, key_u16, "left")
            b = np.searchsorted(subset, key_u16, "right")
            count = b - a

            if task.parquets and count > 0:
                blur_bboxes = load_blur_bboxes(d["__url__"], key_str)
                if blur_bboxes is None:
                    logger.error(
                        f"{task.worker_id:04d} failed to find blur bboxes for {d['__url__']}, {key_str}"
                    )

                elif len(blur_bboxes) > 0:
                    if apply_blur:
                        d["webp"] = blur_image(
                            blurrer, d["jpg"], blur_bboxes, reencode_webp_quality
                        )
                        del d["jpg"]  # Remove jpg version of image
                        blur_count += 1

                    if inject_blur_metadata:
                        json = json_parsed.as_dict()
                        json["face_bboxes"] = list(map(list, blur_bboxes))
                        d["json"] = simdjson.dumps(json).encode()

            for j in range(count):
                if not dry_run:
                    yield {**d, "__key__": f"{key_str}-{j}"}

                output_count += 1

            processed_count += 1

            if processed_count % 1000 == 0:
                log_queue.put_nowait(1000)

            del json_parsed

        log_queue.put_nowait(processed_count % 1000)

    it = subset_iter()
    if shuffle_bufsize > 0:
        it = wds.filters._shuffle(it, shuffle_bufsize, shuffle_bufsize)

    for d in it:
        sw.write(d)

    sw.close()

    if processed_count != total_data:
        logger.error(f"expected {total_data} samples but found {processed_count}")

    with lock:
        state.processed_count += processed_count
        state.output_count += output_count
        state.blur_count += blur_count


def logging_handler(total_data, log_queue):
    bar = tqdm.tqdm(total=total_data)

    # this feels a bit ad-hoc
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(log_handler.formatter)
    handler = ColoredConsoleHandler(tqdm_handler)

    while True:
        try:
            message = log_queue.get(timeout=1)
            if message is None:
                break

            if isinstance(message, int):
                bar.update(message)

            if isinstance(message, logging.LogRecord):
                handler.emit(message)

        except queue.Empty:
            pass

        except:
            from sys import stderr
            from traceback import print_exc

            print_exc(file=stderr)
            raise

    bar.close()


def do_tasks(worker_tasks, args):
    manager = mp.Manager()

    state = manager.Namespace()
    state.processed_count = 0
    state.output_count = 0
    state.blur_count = 0
    state.output_shard_count = 0

    lock = manager.Lock()
    log_queue = manager.Queue()

    # not very elegant
    last_shard = worker_tasks[-1].shards[-1]
    total_data = last_shard.data_start + last_shard.size

    logging_thread = threading.Thread(
        target=logging_handler, args=(total_data, log_queue)
    )
    logging_thread.daemon = True

    processes = [
        mp.Process(target=copy_worker, args=(task, state, lock, log_queue), kwargs=args)
        for task in worker_tasks
    ]

    logging_thread.start()
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # send the sentinel value to the thread to tell it to exit
    log_queue.put_nowait(None)
    logging_thread.join()

    return state


def rmtree_contents(path: Pathy):
    for path in path.iterdir():
        if path.is_file():
            path.unlink()


def postprocess_output(*, output_dir, shard_format, **_):
    logger.info("postprocessing output shards")
    for i, shard in enumerate(sorted(output_dir.iterdir())):
        shard.rename(output_dir / shard_format.format(i))


def set_loglevel(logger, /, verbose, quiet, **_):
    verbose = 0 if verbose is None else sum(verbose)
    quiet = 0 if quiet is None else sum(quiet)
    log_levels = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]
    logger.setLevel(log_levels[max(min(1 - verbose + quiet, len(log_levels)), 0)])


def main(args):
    set_loglevel(logger, **vars(args))

    logger.info("loading shard metadata")
    shards, total_data = load_shard_metadata(**vars(args))
    if len(shards) < args.num_workers:
        args.num_workers = len(shards)

    logger.info("deleting files from output directory")
    rmtree_contents(args.output_dir)

    if args.is_master and not args.dry_run:
        logger.info("copying the subset file to the output directory")
        output_filename = args.output_dir / "sample_ids.npy"
        if isinstance(args.subset_file, CloudPath):
            args.subset_file.copy(output_filename)
        else:
            shutil.copyfile(args.subset_file, output_filename)

    if args.apply_blur and not args.blur_metadata_map:
        logger.fatal("need to pass --blur-metadata-map to use --apply-blur")

    if args.inject_blur_metadata and not args.blur_metadata_map:
        logger.fatal("need to pass --blur-metadata-map to use --inject-blur-metadata")

    # If blur is needed, retrieve json with metadata parquet locations.
    if args.blur_metadata_map is not None:
        parquets = load_parquet_metadata(shards, **vars(args))
        logger.info("loading parquet files")
    else:
        parquets = None

    with tempfile.NamedTemporaryFile("wb") as f:
        if isinstance(args.subset_file, CloudPath):
            logger.info("copying remote subset file to local machine")
            with args.subset_file.open("rb") as sf:
                f.write(sf.read())
            args.subset_file = Path(f.name)

        subset = load_subset(**vars(args))
        logger.info(f"selecting a subset of {len(subset)} examples")

        worker_tasks = plan_tasks(shards, parquets, **vars(args))

        logger.info("starting workers...")
        start_time = time.perf_counter()
        state = do_tasks(worker_tasks, vars(args))
        elapsed_time = time.perf_counter() - start_time

        logger.info(
            f"processed {state.processed_count} images in {elapsed_time:.3f}s ({total_data/elapsed_time:.2f} images/sec)"
        )
        if state.processed_count != total_data:
            logger.error(
                f"expected {total_data} samples but found {state.processed_count}"
            )

        logger.info(f"output {state.output_count} images")
        if state.output_count != len(subset):
            logger.warning(
                f"{len(subset) - state.output_count} images in the subset were not found in the input!"
            )

        logger.info(f"wrote {state.output_shard_count} output shards")

        if state.blur_count > 0:
            logger.info(f"applied blur to {state.blur_count} images")

        if not args.dry_run:
            with (args.output_dir / "meta.json").open("w") as f:
                simdjson.dump(
                    {
                        **{k: str(v) for k, v in vars(args).items()},
                        **vars(state._getvalue()),
                        "cwd": str(Path.cwd()),
                    },
                    f,
                )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
