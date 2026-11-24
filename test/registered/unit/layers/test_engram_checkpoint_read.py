"""Engram host-table loading: reading rows from the checkpoint file in parallel and
the switch for emptying the checkpoint page cache. CPU only, on small fake checkpoints."""

import errno
import json
import mmap
import os
import tempfile
import threading
import types
import unittest
from unittest import mock

import safetensors
import safetensors.torch
import torch

from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.srt.arg_groups.engram_config import EngramReplication
from sglang.srt.environ import envs
from sglang.srt.layers import engram
from sglang.srt.layers import engram_checkpoint_read as ckpt_read
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

ROWS, ROW_BYTES, SCALES = 40, 96, 3
SENTINEL = 0xAB


def _random_bytes(n: int, seed: int) -> bytes:
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 256, (n,), dtype=torch.uint8, generator=gen).numpy().tobytes()
    )


def _raw(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def _write_checkpoint(directory: str) -> tuple[str, dict]:
    """A safetensors file with the three dtypes an Engram table is stored in."""
    path = os.path.join(directory, "model-00001-of-00001.safetensors")
    tensors = {
        "mxfp4": torch.frombuffer(
            bytearray(_random_bytes(ROWS * ROW_BYTES // 2, 1)), dtype=torch.uint8
        ).reshape(ROWS, ROW_BYTES // 2),
        "fp8": torch.frombuffer(
            bytearray(_random_bytes(ROWS * ROW_BYTES, 2)), dtype=torch.uint8
        )
        .view(torch.float8_e4m3fn)
        .reshape(ROWS, ROW_BYTES),
        "scale": torch.frombuffer(
            bytearray(_random_bytes(ROWS * SCALES, 3)), dtype=torch.uint8
        )
        .view(torch.float8_e8m0fnu)
        .reshape(ROWS, SCALES),
    }
    safetensors.torch.save_file(tensors, path)
    return path, tensors


def _data_offset(path: str, name: str) -> int:
    """Where the tensor's bytes start in the file, from the safetensors header."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return 8 + header_len + header[name]["data_offsets"][0]


def _preadv_spy(record: list):
    real = os.preadv

    def spy(fd, buffers, offset, *args):
        record.append((threading.get_ident(), fd, offset, sum(len(b) for b in buffers)))
        return real(fd, buffers, offset, *args)

    return spy


@unittest.skipUnless(os.path.exists("/proc/self/maps"), "needs /proc")
class TestMappedFileRange(CustomTestCase):
    def test_a_loaded_tensor_maps_to_its_range_of_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            path, expected = _write_checkpoint(d)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, ref in expected.items():
                    loaded = f.get_tensor(name)
                    nbytes = loaded.numel() * loaded.element_size()
                    self.assertEqual(
                        ckpt_read.mapped_file_range(loaded.data_ptr(), nbytes),
                        (os.path.realpath(path), _data_offset(path, name)),
                    )
                    # A row slice starts where its first row does.
                    rows = loaded[5:9]
                    self.assertEqual(
                        ckpt_read.mapped_file_range(
                            rows.data_ptr(), rows.numel() * rows.element_size()
                        ),
                        (
                            os.path.realpath(path),
                            _data_offset(path, name) + 5 * nbytes // ROWS,
                        ),
                    )

    def test_memory_that_is_not_a_file_mapping_has_no_range(self):
        anon = torch.zeros(1 << 16, dtype=torch.uint8)
        self.assertIsNone(ckpt_read.mapped_file_range(anon.data_ptr(), anon.numel()))
        self.assertIsNone(ckpt_read.mapped_file_range(anon.data_ptr(), 0))
        with tempfile.TemporaryDirectory() as d:
            _, expected = _write_checkpoint(d)
            blob = safetensors.torch.save({"fp8": expected["fp8"]})
            in_memory = safetensors.torch.load(blob)["fp8"]
            self.assertIsNone(
                ckpt_read.mapped_file_range(
                    in_memory.data_ptr(), in_memory.numel() * in_memory.element_size()
                )
            )

    def test_a_range_running_off_its_mapping_has_no_range(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = _write_checkpoint(d)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                loaded = f.get_tensor("fp8")
                self.assertIsNone(
                    ckpt_read.mapped_file_range(loaded.data_ptr(), 1 << 40)
                )

    def test_a_path_that_does_not_name_the_mapped_file_has_no_range(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = _write_checkpoint(d)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                loaded = f.get_tensor("fp8")
                nbytes = loaded.numel() * loaded.element_size()
                real = os.stat(path)
                for label, changed in {
                    "inode": {"st_ino": real.st_ino + 1},
                    "device": {"st_dev": real.st_dev + 1},
                    "size": {"st_size": real.st_size // 2},
                    "type": {"st_mode": 0o040755},
                }.items():
                    fields = {
                        "st_mode": real.st_mode,
                        "st_ino": real.st_ino,
                        "st_dev": real.st_dev,
                        "st_size": real.st_size,
                    }
                    fake = types.SimpleNamespace(**{**fields, **changed})
                    with (
                        self.subTest(label),
                        mock.patch.object(os, "stat", return_value=fake),
                    ):
                        self.assertIsNone(
                            ckpt_read.mapped_file_range(loaded.data_ptr(), nbytes)
                        )

    def _map_private(self, path: str):
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
        return mm, torch.frombuffer(mm, dtype=torch.uint8)

    def test_a_file_that_was_deleted_or_replaced_has_no_range(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "table.bin")
            with open(path, "wb") as f:
                f.write(_random_bytes(1 << 16, 4))
            mm, view = self._map_private(path)
            try:
                self.assertEqual(
                    ckpt_read.mapped_file_range(view.data_ptr(), view.numel()),
                    (os.path.realpath(path), 0),
                )
                other = os.path.join(d, "other.bin")
                with open(other, "wb") as f:
                    f.write(_random_bytes(1 << 16, 5))
                os.replace(other, path)  # the path now names another inode
                self.assertIsNone(
                    ckpt_read.mapped_file_range(view.data_ptr(), view.numel())
                )
                os.unlink(path)  # and now none
                self.assertIsNone(
                    ckpt_read.mapped_file_range(view.data_ptr(), view.numel())
                )
            finally:
                del view
                mm.close()


class TestReadFileRange(CustomTestCase):
    DATA = _random_bytes(300_007, 6)

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "blob.bin")
        with open(self.path, "wb") as f:
            f.write(self.DATA)

    def _read(self, offset, nbytes, *, threads, block_bytes):
        dst = bytearray([SENTINEL]) * nbytes
        ckpt_read.read_file_range(
            self.path,
            offset,
            memoryview(dst),
            threads=threads,
            block_bytes=block_bytes,
        )
        return bytes(dst)

    def test_bytes_match_the_file_for_any_threads_blocks_and_range(self):
        for offset, nbytes in [
            (0, 0),
            (0, 1),
            (5, 4096),
            (12_345, 250_000),
            (0, 300_007),
        ]:
            for threads in (1, 2, 3, 8, 64):
                for block_bytes in (1 << 20, 100_000, 4096, 7):
                    if block_bytes == 7 and nbytes > 5000:
                        continue
                    with self.subTest(
                        offset=offset,
                        nbytes=nbytes,
                        threads=threads,
                        block_bytes=block_bytes,
                    ):
                        self.assertEqual(
                            self._read(
                                offset, nbytes, threads=threads, block_bytes=block_bytes
                            ),
                            self.DATA[offset : offset + nbytes],
                        )

    def test_each_stripe_is_one_thread_reading_blocks_in_order(self):
        for threads, block_bytes in [
            (4, 10_000),
            (8, 40_000),
            (64, 100_000),
            (3, 1 << 20),
        ]:
            nbytes = 250_000
            expected = min(threads, -(-nbytes // block_bytes))
            calls = []
            start = threading.Barrier(expected, timeout=10)
            seen = set()
            spy = _preadv_spy(calls)

            def preadv(fd, buffers, offset, *args):
                if threading.get_ident() not in seen:
                    seen.add(threading.get_ident())
                    start.wait()  # every stripe must run on a thread of its own
                return spy(fd, buffers, offset, *args)

            with self.subTest(threads=threads, block_bytes=block_bytes):
                with mock.patch.object(os, "preadv", preadv):
                    out = self._read(
                        1000, nbytes, threads=threads, block_bytes=block_bytes
                    )
                self.assertEqual(out, self.DATA[1000 : 1000 + nbytes])
                self.assertEqual(len({c[0] for c in calls}), expected)
                self.assertEqual(
                    len({c[1] for c in calls}), expected
                )  # own descriptors
                self.assertLessEqual(max(c[3] for c in calls), block_bytes)
                self.assertEqual(sum(c[3] for c in calls), nbytes)
                for tid in {c[0] for c in calls}:
                    mine = [c for c in calls if c[0] == tid]
                    for before, after in zip(mine, mine[1:]):
                        self.assertEqual(after[2], before[2] + before[3])

    def test_short_reads_are_continued(self):
        real = os.preadv

        def short(fd, buffers, offset, *args):
            return real(fd, [buffers[0][:1000]], offset, *args)

        with mock.patch.object(os, "preadv", short):
            out = self._read(0, 50_000, threads=2, block_bytes=20_000)
        self.assertEqual(out, self.DATA[:50_000])

    def test_rejects_non_positive_threads_and_block_sizes(self):
        for threads, block_bytes in [(0, 4096), (-1, 4096), (4, 0), (4, -8)]:
            with (
                self.subTest(threads=threads, block_bytes=block_bytes),
                self.assertRaises(ValueError),
            ):
                self._read(0, 100, threads=threads, block_bytes=block_bytes)

    def test_a_file_that_ends_early_raises_and_closes_every_descriptor(self):
        opened, closed = [], []
        real_open, real_close = os.open, os.close

        def tracked_open(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            opened.append(fd)
            return fd

        def tracked_close(fd):
            closed.append(fd)
            return real_close(fd)

        threads_before = threading.active_count()
        with (
            mock.patch.object(os, "open", tracked_open),
            mock.patch.object(os, "close", tracked_close),
            self.assertRaises(OSError) as raised,
        ):
            self._read(len(self.DATA) - 10_000, 100_000, threads=4, block_bytes=8192)
        self.assertEqual(raised.exception.errno, errno.EIO)
        self.assertEqual(sorted(opened), sorted(closed))
        self.assertGreater(len(opened), 0)
        self.assertEqual(threading.active_count(), threads_before)

    def test_a_failing_read_stops_the_other_threads(self):
        real = os.preadv
        calls = []
        failure = threading.Event()

        def flaky(fd, buffers, offset, *args):
            calls.append(offset)
            if offset >= 225_000:  # the last stripe of four fails on its first block
                failure.set()
                raise OSError(errno.EIO, "injected")
            # The others hold their first block until it has failed.
            self.assertTrue(failure.wait(timeout=10))
            return real(fd, buffers, offset, *args)

        threads_before = threading.active_count()
        with mock.patch.object(os, "preadv", flaky), self.assertRaises(OSError):
            self._read(0, 300_000, threads=4, block_bytes=1000)
        self.assertEqual(threading.active_count(), threads_before)
        # Each stripe has 75 blocks; the three healthy ones stop after the block in hand.
        self.assertEqual(len(calls), 4)


@unittest.skipUnless(os.path.exists("/proc/self/maps"), "needs /proc")
class TestReadCheckpointRows(CustomTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path, self.expected = _write_checkpoint(self.dir.name)
        self.file = safetensors.safe_open(self.path, framework="pt", device="cpu")
        self.addCleanup(self.file.__exit__, None, None, None)
        self.file.__enter__()

    def _loaded(self, name):
        return self.file.get_tensor(name)

    def test_rows_are_byte_identical_to_copying_out_of_the_mapping(self):
        for name in ("mxfp4", "fp8", "scale"):
            src = self._loaded(name)[7:31]
            via_copy = torch.empty_like(src)
            via_copy.copy_(src)
            dst = (
                torch.empty_like(src).view(torch.uint8).fill_(SENTINEL).view(src.dtype)
            )
            with self.subTest(name=name):
                with envs.SGLANG_ENGRAM_LOAD_THREADS.override(4):
                    self.assertTrue(ckpt_read.read_checkpoint_rows(dst, src))
                self.assertEqual(_raw(dst), _raw(via_copy))
                self.assertEqual(_raw(dst), _raw(self.expected[name][7:31]))

    def test_threads_and_block_size_come_from_the_environment(self):
        src = self._loaded("fp8")
        dst = torch.empty_like(src)
        with (
            envs.SGLANG_ENGRAM_LOAD_THREADS.override(3),
            envs.SGLANG_ENGRAM_LOAD_BLOCK_MB.override(2),
            mock.patch.object(
                ckpt_read, "read_file_range", wraps=ckpt_read.read_file_range
            ) as read,
        ):
            self.assertTrue(ckpt_read.read_checkpoint_rows(dst, src))
        self.assertEqual(read.call_args.kwargs["threads"], 3)
        self.assertEqual(read.call_args.kwargs["block_bytes"], 2 << 20)
        self.assertEqual(_raw(dst), _raw(self.expected["fp8"]))

    def test_zero_threads_switches_the_parallel_read_off(self):
        src = self._loaded("fp8")
        dst = torch.empty_like(src).view(torch.uint8).fill_(SENTINEL).view(src.dtype)
        with (
            envs.SGLANG_ENGRAM_LOAD_THREADS.override(0),
            mock.patch.object(ckpt_read, "read_file_range") as read,
        ):
            self.assertFalse(ckpt_read.read_checkpoint_rows(dst, src))
        read.assert_not_called()
        self.assertEqual(set(_raw(dst)), {SENTINEL})

    def test_sources_that_are_not_plain_file_ranges_are_left_to_the_caller(self):
        src = self._loaded("fp8")
        in_memory = src.clone()
        cases = {
            "anonymous memory": (torch.empty_like(src), in_memory),
            "other dtype": (torch.empty(src.shape, dtype=torch.uint8), src),
            "other shape": (torch.empty(ROWS - 1, ROW_BYTES, dtype=src.dtype), src),
            "strided source": (torch.empty(ROW_BYTES, ROWS, dtype=src.dtype), src.t()),
            "strided destination": (
                torch.empty(ROW_BYTES, ROWS, dtype=src.dtype).t(),
                src,
            ),
            "no rows": (torch.empty(0, ROW_BYTES, dtype=src.dtype), src[:0]),
        }
        for label, (dst, source) in cases.items():
            with (
                self.subTest(label),
                mock.patch.object(ckpt_read, "read_file_range") as read,
            ):
                self.assertFalse(ckpt_read.read_checkpoint_rows(dst, source))
                read.assert_not_called()

    def test_a_failed_read_is_reported_so_the_caller_can_copy(self):
        src = self._loaded("fp8")
        dst = torch.empty_like(src)
        with (
            mock.patch.object(os, "preadv", side_effect=OSError(errno.EIO, "boom")),
            self.assertLogs(ckpt_read.logger, level="WARNING") as logs,
        ):
            self.assertFalse(ckpt_read.read_checkpoint_rows(dst, src))
        self.assertIn("copying them out of the mapping", logs.output[0])


@unittest.skipUnless(os.path.exists("/proc/self/maps"), "needs /proc")
class TestEngramLoadRows(CustomTestCase):
    """EngramEmbedding._load_rows, called on a stand-in layer: the real class needs a
    parallel runtime."""

    ROW_START, N_ROWS = 6, 20

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path, self.expected = _write_checkpoint(self.dir.name)
        self.file = safetensors.safe_open(self.path, framework="pt", device="cpu")
        self.addCleanup(self.file.__exit__, None, None, None)
        self.file.__enter__()

    def _layer(self, *, shared: bool, host: bool = True):
        return types.SimpleNamespace(
            row_start=self.ROW_START,
            rows=self.N_ROWS,
            _shared=shared,
            host_table=types.SimpleNamespace(dirty=False) if host else None,
        )

    def _param(self, rows: int):
        data = torch.empty(rows, ROW_BYTES, dtype=torch.float8_e4m3fn)
        data.view(torch.uint8).fill_(SENTINEL)
        return torch.nn.Parameter(data, requires_grad=False)

    def _wanted(self):
        return _raw(self.expected["fp8"][self.ROW_START : self.ROW_START + self.N_ROWS])

    def test_per_rank_shard_is_read_in_parallel_and_marked_dirty(self):
        layer, param = self._layer(shared=False), self._param(self.N_ROWS)
        with mock.patch.object(
            engram, "read_checkpoint_rows", wraps=engram.read_checkpoint_rows
        ) as read:
            engram.EngramEmbedding._load_rows(layer, param, self.file.get_tensor("fp8"))
        self.assertTrue(read.call_count == 1)
        self.assertEqual(_raw(param.data), self._wanted())
        self.assertTrue(layer.host_table.dirty)

    def test_shared_table_writes_only_this_ranks_rows(self):
        layer, param = self._layer(shared=True), self._param(ROWS)
        engram.EngramEmbedding._load_rows(layer, param, self.file.get_tensor("fp8"))
        rows = param.data[self.ROW_START : self.ROW_START + self.N_ROWS]
        self.assertEqual(_raw(rows), self._wanted())
        untouched = torch.cat(
            [param.data[: self.ROW_START], param.data[self.ROW_START + self.N_ROWS :]]
        )
        self.assertEqual(set(_raw(untouched)), {SENTINEL})
        self.assertTrue(layer.host_table.dirty)

    def test_loading_is_identical_whichever_way_the_rows_are_read(self):
        loaded = self.file.get_tensor("fp8")
        results = {}
        for label, threads in (("parallel", 8), ("mapping", 0)):
            layer, param = self._layer(shared=False), self._param(self.N_ROWS)
            with envs.SGLANG_ENGRAM_LOAD_THREADS.override(threads):
                engram.EngramEmbedding._load_rows(layer, param, loaded)
            results[label] = _raw(param.data)
            self.assertTrue(layer.host_table.dirty)
        self.assertEqual(results["parallel"], results["mapping"])
        self.assertEqual(results["parallel"], self._wanted())

    def test_a_failed_parallel_read_falls_back_to_the_copy(self):
        layer, param = self._layer(shared=False), self._param(self.N_ROWS)
        with mock.patch.object(os, "preadv", side_effect=OSError(errno.EIO, "boom")):
            engram.EngramEmbedding._load_rows(layer, param, self.file.get_tensor("fp8"))
        self.assertEqual(_raw(param.data), self._wanted())
        self.assertTrue(layer.host_table.dirty)

    def test_tables_outside_host_memory_are_copied(self):
        layer, param = self._layer(shared=False, host=False), self._param(self.N_ROWS)
        with mock.patch.object(engram, "read_checkpoint_rows") as read:
            engram.EngramEmbedding._load_rows(layer, param, self.file.get_tensor("fp8"))
        read.assert_not_called()
        self.assertEqual(_raw(param.data), self._wanted())


@unittest.skipUnless(os.path.exists("/proc/self/maps"), "needs /proc")
class TestHostTableLayouts(CustomTestCase):
    """Rows loaded into the buffers _HostTable maps: an anonymous huge-page-advised shard
    per rank, or one memfd-backed table shared by the TP group."""

    DIM, ROW_START, N_ROWS = 4 * FP8_BLOCK_SIZE, 11, 17
    TOTAL = 48

    def _checkpoint(self, directory: str, mxfp4: bool) -> str:
        row_bytes = engram._table_row_bytes(self.DIM, mxfp4)
        n_scales = self.DIM // FP8_BLOCK_SIZE
        weight = torch.frombuffer(
            bytearray(_random_bytes(self.TOTAL * row_bytes, 7)), dtype=torch.uint8
        ).reshape(self.TOTAL, row_bytes)
        scale = (
            torch.frombuffer(
                bytearray(_random_bytes(self.TOTAL * n_scales, 8)), dtype=torch.uint8
            )
            .view(torch.float8_e8m0fnu)
            .reshape(self.TOTAL, n_scales)
        )
        path = os.path.join(directory, "model.safetensors")
        safetensors.torch.save_file(
            {
                "weight": weight if mxfp4 else weight.view(torch.float8_e4m3fn),
                "scale": scale,
            },
            path,
        )
        return path

    def _buffer(self, shared: bool, nbytes: int):
        if shared:
            fd = os.memfd_create("engram_test")
            self.addCleanup(os.close, fd)
            os.ftruncate(fd, nbytes)
            mm = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED)
        else:
            mm = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        mm.madvise(mmap.MADV_HUGEPAGE)
        raw = torch.frombuffer(mm, dtype=torch.uint8)
        raw.fill_(SENTINEL)
        self.addCleanup(mm.close)
        return raw

    def _load(self, path: str, *, shared: bool, mxfp4: bool, threads: int) -> bytes:
        n = self.TOTAL if shared else self.N_ROWS
        row_bytes = engram._table_row_bytes(self.DIM, mxfp4)
        nbytes = n * (row_bytes + self.DIM // FP8_BLOCK_SIZE)
        raw = self._buffer(shared, nbytes)
        weight, scale = engram._table_views(raw, n, self.DIM, mxfp4)
        layer = types.SimpleNamespace(
            row_start=self.ROW_START,
            rows=self.N_ROWS,
            _shared=shared,
            host_table=types.SimpleNamespace(dirty=False),
        )
        with (
            envs.SGLANG_ENGRAM_LOAD_THREADS.override(threads),
            safetensors.safe_open(path, framework="pt", device="cpu") as f,
        ):
            for name, param in (("weight", weight), ("scale", scale)):
                engram.EngramEmbedding._load_rows(
                    layer,
                    torch.nn.Parameter(param, requires_grad=False),
                    f.get_tensor(name),
                )
        self.assertTrue(layer.host_table.dirty)
        return raw.numpy().tobytes()

    def test_every_layout_loads_the_same_bytes_as_the_mapping_copy(self):
        for shared in (False, True):
            for mxfp4 in (False, True):
                with (
                    self.subTest(shared=shared, mxfp4=mxfp4),
                    tempfile.TemporaryDirectory() as d,
                ):
                    path = self._checkpoint(d, mxfp4)
                    parallel = self._load(path, shared=shared, mxfp4=mxfp4, threads=4)
                    copied = self._load(path, shared=shared, mxfp4=mxfp4, threads=0)
                    self.assertEqual(parallel, copied)
                    self.assertNotEqual(parallel, bytes([SENTINEL]) * len(parallel))
                    if shared:
                        # Rows outside this rank's range stay as they were.
                        row_bytes = engram._table_row_bytes(self.DIM, mxfp4)
                        before = self.ROW_START * row_bytes
                        self.assertEqual(set(parallel[:before]), {SENTINEL})


class TestDropCheckpointCacheSwitch(CustomTestCase):
    def setUp(self):
        patcher = mock.patch.object(engram, "_page_cache_dropped", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_drops_the_cache_once_per_process(self):
        with mock.patch.object(
            engram, "drop_checkpoint_page_cache", return_value=(3, 1 << 30)
        ) as drop:
            engram._drop_page_cache_once("first")
            engram._drop_page_cache_once("second")
        drop.assert_called_once_with()

    def test_switch_keeps_the_cache(self):
        with (
            envs.SGLANG_ENGRAM_DROP_CHECKPOINT_CACHE.override(False),
            mock.patch.object(engram, "drop_checkpoint_page_cache") as drop,
            self.assertLogs(engram.logger, level="INFO") as logs,
        ):
            engram._drop_page_cache_once("before pre-faulting")
            engram._drop_page_cache_once("again")
        drop.assert_not_called()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("keeping the page cache", logs.output[0])

    def test_drop_covers_every_safetensors_file_of_the_model(self):
        with tempfile.TemporaryDirectory() as d:
            for name, size in (
                ("a.safetensors", 10),
                ("b.safetensors", 20),
                ("c.json", 5),
            ):
                with open(os.path.join(d, name), "wb") as f:
                    f.write(b"x" * size)
            with (
                mock.patch.object(
                    engram,
                    "get_model",
                    return_value=types.SimpleNamespace(model_path=d),
                ),
                mock.patch.object(os, "posix_fadvise") as fadvise,
            ):
                self.assertEqual(engram.drop_checkpoint_page_cache(), (2, 30))
            self.assertEqual(fadvise.call_count, 2)
            for call in fadvise.call_args_list:
                self.assertEqual(call.args[1:], (0, 0, os.POSIX_FADV_DONTNEED))

    def _finish_load(self):
        table = engram._HostTable.__new__(engram._HostTable)
        table.replication = EngramReplication.NONE
        table.dirty = True
        table.bytes = torch.zeros(1, dtype=torch.uint8)
        with (
            mock.patch.object(
                engram, "_huge_pages_backing", side_effect=[(1000, 0), (1000, 1000)]
            ),
            mock.patch.object(engram, "drop_checkpoint_page_cache") as drop,
            mock.patch.object(engram._HostTable, "_collapse") as collapse,
        ):
            table.finish_load("layer 0")
        return drop, collapse

    def test_collapse_after_a_huge_page_shortfall_drops_the_cache_by_default(self):
        drop, collapse = self._finish_load()
        drop.assert_called_once_with()
        collapse.assert_called_once_with()

    def test_switch_skips_the_drop_but_still_collapses(self):
        with envs.SGLANG_ENGRAM_DROP_CHECKPOINT_CACHE.override(False):
            drop, collapse = self._finish_load()
        drop.assert_not_called()
        collapse.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
