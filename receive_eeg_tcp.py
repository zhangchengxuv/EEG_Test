#!/usr/bin/env python3
"""
方法 A：用 TCP 客户端接收 ESP32（main_wifi）推送的数据行。
数据格式与 USB 串口版相同，每行形如：channel:v1,v2,...,v8

默认连接 192.168.137.122:3333；IP 变化时可用参数或改下方默认值。
"""

import argparse
import socket
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="连接 ESP32 的 TCP 服务，打印或保存 channel: 数据行"
    )
    parser.add_argument(
        "host",
        nargs="?",
        default="192.168.137.122",
        help="ESP32 在局域网中的 IP（串口监视器里“已连接 WiFi”后显示）",
    )
    parser.add_argument(
        "port",
        nargs="?",
        type=int,
        default=3333,
        help="TCP 端口（与 main_wifi.cpp 中 TCP_SERVER_PORT 一致）",
    )
    parser.add_argument(
        "-n",
        "--lines",
        type=int,
        default=0,
        metavar="N",
        help="只接收 N 行后退出；0 表示一直接收直到 Ctrl+C",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        metavar="FILE",
        help="同时追加写入该文件（UTF-8）；不填则只打印到终端",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        metavar="SEC",
        help="TCP 连接超时（秒），默认 30；仍超时请按错误提示做网络排查",
    )
    args = parser.parse_args()

    out_fp = None
    if args.output:
        out_fp = open(args.output, "a", encoding="utf-8")

    try:
        print(
            f"正在连接 {args.host}:{args.port}（超时 {args.timeout} 秒）...",
            file=sys.stderr,
        )
        with socket.create_connection(
            (args.host, args.port), timeout=args.timeout
        ) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("已连接，开始接收数据（Ctrl+C 结束）\n", file=sys.stderr)
            f = sock.makefile("r", encoding="utf-8", newline="")
            count = 0
            try:
                while True:
                    line = f.readline()
                    if not line:
                        print("连接已关闭", file=sys.stderr)
                        break
                    line = line.rstrip("\r\n")
                    print(line)
                    if out_fp:
                        out_fp.write(line + "\n")
                        out_fp.flush()
                    count += 1
                    if args.lines and count >= args.lines:
                        break
            except KeyboardInterrupt:
                print("\n已停止", file=sys.stderr)
    except OSError as e:
        print(f"连接失败: {e}", file=sys.stderr)
        _print_troubleshoot(args.host, args.port)
        sys.exit(1)
    finally:
        if out_fp:
            out_fp.close()


def _print_troubleshoot(host: str, port: int) -> None:
    """超时时给出针对常见场景（含 Windows 热点 192.168.137.x）的说明。"""
    lines = [
        "",
        "排查建议（任一条不成立都可能导致 “timed out”）：",
        f"  1) 在 **能跑本脚本的同一台电脑** 上先执行: ping {host}",
        "     若 ping 都不通，说明走不到这块板子（不是 Python 的问题）。",
        "  2) IP 是 192.168.137.x 时，多半是连了 **本机开的热点(移动热点/共享网络)**。",
        "     请尽量在 **开热点的那台 Windows 电脑** 上直接运行本脚本，不要用另一台机子连。",
        "  3) 若本脚本在 **WSL2** 里跑，到 Windows 网段常不通。请用 **Windows 自带终端 / CMD / PowerShell** 运行:",
        f"     python receive_eeg_tcp.py {host} {port}",
        "  4) 暂时关闭 **VPN/代理/全局流量软件**，再试。",
        "  5) 确认板子仍在线：串口监视器里应仍是同一 IP；必要时复位板子再看 IP。",
        f"  6) 端口应为固件里 TCP_SERVER_PORT（当前常用 {port}），且板子已打印「监听 TCP 端口」。",
        "",
    ]
    print("\n".join(lines), file=sys.stderr)


if __name__ == "__main__":
    main()
