"""Read Engram table rows from the checkpoint file into host memory.

The weight loader hands a table to the layer as a tensor mapped from the safetensors file.
Copying out of that mapping reads the file one page fault at a time, from one thread:
each fault waits for a readahead window of the page cache, so on a network filesystem the
load is bound by latency rather than bandwidth. The rows are a plain byte range of the
file, so they can be read straight into the destination with large sequential preads from
several threads instead, and the bytes are the ones the mapping would have shown.
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import re
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# start-end perms offset major:minor inode path
_MAPS_LINE = re.compile(
    r"^([0-9a-f]+)-([0-9a-f]+) \S+ ([0-9a-f]+) ([0-9a-f]+):([0-9a-f]+) (\d+) *(.*)$"
)


def mapped_file_range(addr: int, nbytes: int) -> Optional[tuple[str, int]]:
    """(path, file offset) of the bytes [addr, addr + nbytes) of this process when they
    are one contiguous run of a single regular file's mapping, else None: anonymous
    memory, several mappings, or a file that was deleted or replaced since it was mapped
    (the path must still name the mapped inode)."""
    if nbytes <= 0:
        return None
    end = addr + nbytes
    identity = None
    first_offset = 0
    cursor = addr  # the next address that still needs a mapping
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                m = _MAPS_LINE.match(line)
                if m is None:
                    continue
                start, stop = int(m.group(1), 16), int(m.group(2), 16)
                if stop <= cursor:
                    continue
                if start > cursor:
                    return None  # a hole
                this = m.group(4, 5, 6, 7)
                offset = int(m.group(3), 16)
                if identity is None:
                    identity = this
                    first_offset = offset + (cursor - start)
                elif this != identity or offset != first_offset + (start - addr):
                    return None
                cursor = stop
                if cursor >= end:
                    break
    except OSError:
        return None
    if identity is None or cursor < end:
        return None
    dev_major, dev_minor, inode, path = identity
    if not path.startswith("/") or path.endswith(" (deleted)") or inode == "0":
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    if (
        not stat.S_ISREG(st.st_mode)
        or st.st_ino != int(inode)
        or (os.major(st.st_dev), os.minor(st.st_dev))
        != (int(dev_major, 16), int(dev_minor, 16))
        or first_offset + nbytes > st.st_size
    ):
        return None
    return path, first_offset


def read_file_range(
    path: str,
    offset: int,
    dst: memoryview,
    *,
    threads: int,
    block_bytes: int,
) -> None:
    """Fill ``dst`` (a writable byte view) with ``len(dst)`` bytes of ``path`` from
    ``offset``. The range is cut into at most ``threads`` contiguous stripes of whole
    blocks; each stripe is read front to back by one thread on its own descriptor, with
    preads of ``block_bytes``, so the kernel sees sequential streams. Raises OSError if
    the file ends early or a read fails, after every thread has stopped."""
    if threads < 1 or block_bytes < 1:
        raise ValueError(
            f"threads and block_bytes must be positive, got {threads} and {block_bytes}"
        )
    nbytes = len(dst)
    blocks = -(-nbytes // block_bytes)
    stripes = []
    base, extra = divmod(blocks, min(threads, max(blocks, 1)))
    lo = 0
    while lo < nbytes:
        hi = min(lo + (base + (len(stripes) < extra)) * block_bytes, nbytes)
        stripes.append((lo, hi))
        lo = hi
    failed = threading.Event()

    def read_stripe(lo: int, hi: int) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            pos = lo
            while pos < hi and not failed.is_set():
                got = os.preadv(
                    fd, [dst[pos : min(pos + block_bytes, hi)]], offset + pos
                )
                if got == 0:
                    raise OSError(
                        errno.EIO, f"unexpected end of file at offset {offset + pos}"
                    )
                pos += got
        except BaseException:
            failed.set()
            raise
        finally:
            os.close(fd)

    if len(stripes) <= 1:
        for stripe in stripes:
            read_stripe(*stripe)
        return
    with ThreadPoolExecutor(len(stripes)) as pool:
        for done in [pool.submit(read_stripe, lo, hi) for lo, hi in stripes]:
            done.result()


def _byte_view(t: torch.Tensor) -> memoryview:
    nbytes = t.numel() * t.element_size()
    return memoryview((ctypes.c_char * nbytes).from_address(t.data_ptr())).cast("B")


def read_checkpoint_rows(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """Fill the host tensor ``dst`` with the bytes of ``src``, a slice of a tensor the
    loader mapped from a checkpoint file, by reading that file range in parallel
    (SGLANG_ENGRAM_LOAD_THREADS, SGLANG_ENGRAM_LOAD_BLOCK_MB). Returns False, with ``dst``
    untouched or only partly written, when the caller must copy ``src`` instead: the read
    is off, ``src`` is not backed by a regular file (read into memory, streamed, converted
    in dtype or non-contiguous), or the read failed."""
    threads = envs.SGLANG_ENGRAM_LOAD_THREADS.get()
    if (
        threads < 1
        or src.numel() == 0
        or src.dtype != dst.dtype
        or src.shape != dst.shape
        or src.device.type != "cpu"
        or dst.device.type != "cpu"
        or not src.is_contiguous()
        or not dst.is_contiguous()
    ):
        return False
    nbytes = src.numel() * src.element_size()
    backing = mapped_file_range(src.data_ptr(), nbytes)
    if backing is None:
        return False
    path, offset = backing
    started = time.perf_counter()
    try:
        read_file_range(
            path,
            offset,
            _byte_view(dst),
            threads=threads,
            block_bytes=max(1, envs.SGLANG_ENGRAM_LOAD_BLOCK_MB.get()) << 20,
        )
    except OSError as e:
        logger.warning(
            "engram host table: reading %d bytes of the checkpoint in parallel failed "
            "(%s); copying them out of the mapping instead",
            nbytes,
            e,
        )
        return False
    seconds = time.perf_counter() - started
    logger.info(
        "engram host table: read %.2f GiB from the checkpoint with %d threads in "
        "%.1f s (%.0f MB/s)",
        nbytes / 2**30,
        threads,
        seconds,
        nbytes / 1e6 / max(seconds, 1e-9),
    )
    return True
