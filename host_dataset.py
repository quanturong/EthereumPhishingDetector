"""
HTTP server host dataset_bundle qua LAN hoac ngrok.
Show local IPs + log download.

Cach chay:
  python host_dataset.py                    # port 8000, LAN only
  python host_dataset.py --port 9000
  python host_dataset.py --dir E:\\other_path
"""

import argparse
import http.server
import os
import socket
import socketserver
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def get_local_ips():
    """Return all non-loopback IPv4 addresses."""
    hostname = socket.gethostname()
    ips = set()
    try:
        for res in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = res[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    # Also probe via UDP trick (no actual connection)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(ips)


class LoggingHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with download logging + friendly listing."""

    def log_message(self, format, *args):
        ts = datetime.now().strftime("%H:%M:%S")
        client = self.client_address[0]
        msg = format % args
        # Only log GET requests with file paths, not directory listing
        if "GET " in msg:
            print(f"[{ts}] {client} - {msg}", flush=True)

    def do_GET(self):
        # Log download start for non-directory paths
        path = self.translate_path(self.path)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / 1e6
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {self.client_address[0]} DOWNLOADING "
                  f"{Path(path).name} ({size_mb:.1f} MB)", flush=True)
        return super().do_GET()


def print_banner(port: int, root: Path):
    total_size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e9
    n_files = sum(1 for _ in root.rglob("*") if _.is_file())
    ips = get_local_ips()

    print("=" * 70)
    print(" EthPhishGraph-2026 Dataset HTTP Server")
    print("=" * 70)
    print(f"Serving:  {root}")
    print(f"Size:     {total_size:.2f} GB ({n_files} files)")
    print(f"Port:     {port}")
    print()
    print("Access URLs:")
    print(f"  Local:   http://localhost:{port}/")
    for ip in ips:
        print(f"  LAN:     http://{ip}:{port}/")
    print()
    print("To share OUTSIDE your LAN, run one of:")
    print(f"  ngrok http {port}")
    print(f"  cloudflared tunnel --url http://localhost:{port}")
    print()
    print("Firewall: Windows may prompt to allow python.exe on public network.")
    print("Press Ctrl+C to stop server.")
    print("=" * 70)
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dir",  default=str(Path(__file__).parent / "dataset_bundle"),
                   help="Folder to serve")
    p.add_argument("--bind", default="0.0.0.0", help="Bind address (0.0.0.0 = all interfaces)")
    args = p.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists():
        print(f"ERROR: {root} not exists. Run prepare_dataset_bundle.py first.")
        sys.exit(1)

    os.chdir(root)
    print_banner(args.port, root)

    # Allow reuse of port to avoid TIME_WAIT issues
    class ReuseTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    try:
        with ReuseTCPServer((args.bind, args.port), LoggingHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[stopped] Server terminated by user.")
    except OSError as e:
        if e.errno == 10048:  # Windows: port in use
            print(f"ERROR: Port {args.port} already in use. Try --port {args.port+1}")
        else:
            raise


if __name__ == "__main__":
    main()
