"""python -m dashboard [--port 8000] [--host 127.0.0.1] [--no-browser] [--no-sync] [--sync-every 300]"""

import argparse
import logging
import os
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC multi-agent trading dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument("--no-sync", action="store_true", help="don't download new Binance data in the background")
    parser.add_argument("--sync-every", type=float, default=300, help="seconds between background data syncs")
    args = parser.parse_args()
    # Read by dashboard.app at import time, so set them before importing it.
    os.environ["BTC_SYNC"] = "0" if args.no_sync else "1"
    os.environ["BTC_SYNC_EVERY"] = str(args.sync_every)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"\n  Dashboard running at {url}  (Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("dashboard.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
