"""Standalone receiving PC monitor. Copy this file and result_serial.py; install pyserial."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time

import serial
import serial.tools.list_ports

try:
    from .result_serial import ResultStreamParser, TARGET_LABELS, open_result_port
except ImportError:
    from result_serial import ResultStreamParser, TARGET_LABELS, open_result_port


def main(argv=None):
    parser = argparse.ArgumentParser(description="接收 SSVEP 分类结果，只监视与记录，不控制设备")
    parser.add_argument("--port", help="接收电脑上的串口号，例如 COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=1.5, help="超过此秒数没有有效帧则提示结果失效")
    parser.add_argument("--csv", type=Path, help="可选：保存收到的结果")
    parser.add_argument("--list-ports", action="store_true", help="列出本机串口")
    args = parser.parse_args(argv)
    if args.list_ports:
        for item in serial.tools.list_ports.comports():
            print(f"{item.device}: {item.description}")
        return 0
    if not args.port or args.baud <= 0 or args.timeout <= 0:
        parser.error("请设置 --port，并使用正数波特率与超时")
    device = log_file = None
    try:
        device = open_result_port(args.port, args.baud, read_timeout=.1)
        device.reset_input_buffer()
        writer = None
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            log_file = args.csv.open("w", newline="", encoding="utf-8-sig")
            writer = csv.writer(log_file)
            writer.writerow(["received_local", "sequence", "mode", "target_id", "frequency_hz"])
        print(f"已连接 {args.port} / {args.baud} / 8N1，等待结果；Ctrl+C 退出。", flush=True)
        stream = ResultStreamParser()
        last_valid = time.perf_counter()
        timed_out = False
        while True:
            chunk = device.read(min(max(device.in_waiting, 1), 4096))
            before = stream.errors
            for message in stream.feed(chunk):
                last_valid, timed_out = time.perf_counter(), False
                stamp = datetime.now().isoformat(timespec="milliseconds")
                frequency = f"{message.frequency_hz:.2f} Hz" if message.target_id else "—"
                print(f"{stamp}  #{message.sequence}  [{message.mode}]  {TARGET_LABELS[message.target_id]}  {frequency}", flush=True)
                if writer:
                    writer.writerow([stamp, message.sequence, message.mode, message.target_id, f"{message.frequency_hz:.2f}"])
                    log_file.flush()
            if stream.errors != before:
                print(f"已丢弃格式错误帧，累计 {stream.errors} 条", file=sys.stderr, flush=True)
            if not timed_out and time.perf_counter()-last_valid >= args.timeout:
                print("接收超时：当前结果已失效，等待新消息。", flush=True)
                timed_out = True
    except KeyboardInterrupt:
        return 0
    except (OSError, serial.SerialException, ValueError) as exc:
        print(f"接收失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if device is not None:
            device.close()
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    sys.exit(main())
