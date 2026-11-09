"""Protocol and real pyserial loopback tests; never opens physical COM ports."""
import threading
import time
import unittest
from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
from unittest import mock

import serial

try:
    from .result_serial import (MAX_LINE_BYTES, ResultMessage, ResultSerialSender, ResultStreamParser,
                                encode_result, parse_result, same_serial_port, open_result_port)
except ImportError:
    from result_serial import (MAX_LINE_BYTES, ResultMessage, ResultSerialSender, ResultStreamParser,
                               encode_result, parse_result, same_serial_port, open_result_port)


def wait_for(predicate, timeout=1.5):
    deadline = time.perf_counter()+timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        threading.Event().wait(.005)
    raise AssertionError("Timed out waiting for background result writer")


class ProtocolTests(unittest.TestCase):
    def test_wire_format_all_targets_modes_and_zero(self):
        expected = b"RESULT,42,EEG,3,15.00\r\n"
        self.assertEqual(encode_result(ResultMessage(42, "EEG", 3, 15)), expected)
        self.assertEqual(parse_result(expected), ResultMessage(42, "EEG", 3, 15))
        for mode in ("EEG", "TEST", "SIM"):
            for target in range(5):
                message = ResultMessage(1, mode, target, 0. if not target else 8.57+target)
                self.assertEqual(parse_result(encode_result(message)), message)
        self.assertEqual(encode_result(ResultMessage(1, "TEST", 0, -0.)), b"RESULT,1,TEST,0,0.00\r\n")

    def test_invalid_frames_never_become_commands(self):
        for raw in (b"channel:1,2,3,4,5,6,7,8\n", b"RESULT,1,EEG,3,nan\n",
                    b"RESULT,1,EEG,3,0.00\n", b"RESULT,1,TEST,0,15.00\n",
                    b"RESULT,0,EEG,1,8.00\n", b"RESULT,1,UNKNOWN,1,8.00\n",
                    b"RESULT,1,EEG,5,8.00\n", b"RESULT,1,EEG,3,15.00,extra\n", b"\xff\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_result(raw)
        for message in (ResultMessage(1, "EEG", True, 12), ResultMessage(1, "EEG", 1, float("nan")),
                        ResultMessage(-1, "EEG", 1, 12), ResultMessage(1, "EEG", 1, .001)):
            with self.assertRaises(ValueError):
                encode_result(message)

    def test_fragmented_coalesced_corrupt_and_oversized_input_recovers(self):
        stream = ResultStreamParser()
        frame = encode_result(ResultMessage(1, "TEST", 4, 20))
        self.assertEqual(stream.feed(frame[:9]), [])
        self.assertEqual(stream.feed(frame[9:]+frame), [parse_result(frame)]*2)
        self.assertEqual(stream.feed(b"noise\n"), [])
        self.assertEqual(stream.feed(b"x"*(MAX_LINE_BYTES+10)), [])
        self.assertEqual(stream.feed(b"unfinished\n"+frame), [parse_result(frame)])
        self.assertEqual(stream.errors, 2)
        self.assertEqual(stream.pending, bytearray())

    def test_windows_port_aliases_cannot_bypass_conflict_guard(self):
        for port in ("COM6", " com6 ", r"\\.\COM6", "COM006"):
            self.assertTrue(same_serial_port(port, "COM6"))
        self.assertFalse(same_serial_port("COM6", "COM7"))
        self.assertFalse(same_serial_port("", ""))


class SenderTests(unittest.TestCase):
    def make_loopback(self, **kwargs):
        handles = []
        def factory(port, baudrate):
            handle = open_result_port(port, baudrate)
            handles.append(handle)
            return handle
        sender = ResultSerialSender("loop://", interval=.04, stale_after=.16, port_factory=factory, **kwargs)
        self.addCleanup(sender.close)
        sender.start()
        wait_for(lambda: sender.snapshot()["sent_count"] >= 1)
        self.assertEqual(handles[0].bytesize, serial.EIGHTBITS)
        self.assertEqual(handles[0].parity, serial.PARITY_NONE)
        self.assertEqual(handles[0].stopbits, serial.STOPBITS_ONE)
        return sender, handles[0]

    def read_until(self, handle, predicate):
        stream = ResultStreamParser()
        messages = []
        def read():
            messages.extend(stream.feed(handle.read(handle.in_waiting)))
            return any(predicate(message) for message in messages)
        wait_for(read)
        return messages

    def test_real_serial_loopback_positive_zero_and_final_zero(self):
        sender, handle = self.make_loopback()
        sender.publish("EEG", 3, 15.)
        received = self.read_until(handle, lambda message: message.target_id == 3)
        self.assertTrue(any(message.mode == "EEG" and message.frequency_hz == 15 for message in received))
        sender.publish("EEG")
        received = self.read_until(handle, lambda message: message.target_id == 0)
        self.assertTrue(all(message.frequency_hz == 0 for message in received if message.target_id == 0))
        self.assertTrue(sender.close())
        info = sender.snapshot()
        self.assertEqual(parse_result(info["last_line"].encode()).target_id, 0)
        self.assertFalse(info["connected"])
        self.assertFalse(info["error"])

    def test_stale_evidence_expires_even_when_writer_keeps_heartbeating(self):
        sender, handle = self.make_loopback()
        handle.reset_input_buffer()
        sender.publish("TEST", 4, 20.)
        received = self.read_until(handle, lambda message: message.target_id == 4)
        last_positive = max(message.sequence for message in received if message.target_id == 4)
        received = self.read_until(handle, lambda message: message.sequence > last_positive and message.target_id == 0)
        self.assertEqual(received[-1].target_id, 0)
        sender.publish("TEST", 4, 20.)  # A fresh UI heartbeat restores a held mouse choice.
        self.read_until(handle, lambda message: message.target_id == 4)

    def test_out_of_order_result_cannot_overwrite_newer_choice(self):
        sender, handle = self.make_loopback()
        older = time.perf_counter()
        sender.publish("TEST", 2, 12.)
        self.assertFalse(sender.publish("TEST", 1, 8.57, evidence_time=older))
        messages = self.read_until(handle, lambda message: message.target_id == 2)
        self.assertFalse(any(message.target_id == 1 for message in messages))

    def test_mode_changes_and_sequence_numbers_on_wire(self):
        sender, handle = self.make_loopback()
        sender.publish("SIM", 3, 18.)
        messages = self.read_until(handle, lambda message: message.mode == "SIM" and message.target_id == 3)
        sender.publish("TEST")
        messages += self.read_until(handle, lambda message: message.mode == "TEST" and message.target_id == 0)
        self.assertEqual([message.sequence for message in messages], sorted({message.sequence for message in messages}))

    def test_write_failure_is_reported_and_handle_closed(self):
        class BrokenPort:
            closed = False
            def write(self, line):
                raise serial.SerialException("USB adapter unplugged")
            def close(self):
                self.closed = True
        handle = BrokenPort()
        sender = ResultSerialSender("fake", port_factory=lambda *args: handle)
        sender.start()
        wait_for(lambda: not sender.is_running())
        self.assertIn("unplugged", sender.snapshot()["error"])
        self.assertEqual(sender.snapshot()["sent_count"], 0)
        self.assertTrue(handle.closed)

    def test_partial_write_is_failure(self):
        class PartialPort:
            def write(self, line):
                return len(line)-1
            def close(self):
                pass
        sender = ResultSerialSender("fake", port_factory=lambda *args: PartialPort())
        sender.start()
        wait_for(lambda: not sender.is_running())
        self.assertTrue(sender.snapshot()["error"])
        self.assertEqual(sender.snapshot()["sent_count"], 0)

    def test_slow_writer_does_not_queue_obsolete_positive_results_after_stop(self):
        entered, release = threading.Event(), threading.Event()
        class SlowPort:
            def __init__(self):
                self.lines = []
            def write(self, line):
                if not self.lines:
                    entered.set()
                    release.wait(1)
                self.lines.append(line)
                return len(line)
            def flush(self):
                pass
            def close(self):
                pass
        handle = SlowPort()
        sender = ResultSerialSender("fake", port_factory=lambda *args: handle)
        self.addCleanup(sender.close)
        sender.start()
        self.assertTrue(entered.wait(1))
        for target in (1, 2, 3, 4):
            sender.publish("TEST", target, 20.)
        sender.request_close()
        release.set()
        wait_for(lambda: not sender.is_running())
        self.assertTrue(all(parse_result(line).target_id == 0 for line in handle.lines))
        self.assertFalse(sender.publish("TEST", 4, 20.))


class ReceiverTests(unittest.TestCase):
    def test_receiver_cli_reads_real_loopback_and_saves_csv(self):
        try:
            from . import receive_results
        except ImportError:
            import receive_results
        handles = []
        def factory(port, baudrate):
            handle = open_result_port(port, baudrate, read_timeout=.01)
            handles.append(handle)
            return handle
        sender = ResultSerialSender("loop://", mode="TEST", interval=.025, stale_after=.2, port_factory=factory)
        self.addCleanup(sender.close)
        sender.start()
        wait_for(lambda: sender.snapshot()["sent_count"] >= 1)
        sender.publish("TEST", 3, 15.)
        deadline = time.perf_counter()+.13
        class Reader:
            def __getattr__(self, key):
                return getattr(handles[0], key)
            def read(self, count):
                if time.perf_counter() >= deadline:
                    raise KeyboardInterrupt
                return handles[0].read(count)
            def close(self):
                pass  # Sender owns this loopback endpoint.
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/"received.csv"
            output = io.StringIO()
            with mock.patch.object(receive_results, "open_result_port", return_value=Reader()), redirect_stdout(output):
                code = receive_results.main(["--port", "loop://", "--csv", str(destination)])
            self.assertEqual(code, 0)
            with destination.open(encoding="utf-8-sig") as received:
                rows = list(csv.DictReader(received))
            self.assertGreaterEqual(len(rows), 2)
            self.assertTrue(all(row["mode"] == "TEST" and row["target_id"] == "3" and row["frequency_hz"] == "15.00" for row in rows))
            self.assertIn("左肘", output.getvalue())


if __name__ == "__main__":
    unittest.main()
