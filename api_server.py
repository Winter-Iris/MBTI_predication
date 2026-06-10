"""
MBTI 预测系统 — 统一启动脚本
=============================
一键启动后端 (FastAPI + 模型) 和前端 (静态 HTML)，
自动打开浏览器。

Usage:
    python api_server.py                  # 默认 http://localhost:8000
    python api_server.py --port 3000      # 自定义端口
    python api_server.py --no-browser     # 不自动打开浏览器
"""

import argparse
import os
import socket
import sys
import time
import webbrowser
from pathlib import Path

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, OSError, socket.timeout):
            return False


def main():
    parser = argparse.ArgumentParser(description="MBTI 预测系统启动器")
    parser.add_argument("--host", default=HOST, help=f"绑定地址 (默认 {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"端口 (默认 {PORT})")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    os.chdir(str(Path(__file__).parent))
    url = f"http://{args.host}:{args.port}"

    # 检查端口
    if is_port_in_use(args.host, args.port):
        print(f"Port {args.port} in use — opening browser: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return

    print(f"Starting MBTI server at {url} ...")
    print(f"(Press Ctrl+C to stop)\n")

    # 非阻塞启动 uvicorn（在后台线程启动 HTTP 服务，主线程打开浏览器）
    config = uvicorn.Config(
        "src.app.api:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=args.reload,
    )
    server = uvicorn.Server(config)

    import threading
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # 等待服务器就绪
    print("Waiting for server...", end="", flush=True)
    for _ in range(60):
        if is_port_in_use(args.host, args.port):
            print(" ready")
            break
        time.sleep(0.5)
    else:
        print(" timeout")

    if not args.no_browser:
        webbrowser.open(url)

    # 保持主线程存活
    try:
        while t.is_alive():
            t.join(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
