"""Windows-friendly loopback launcher; opens only after server readiness."""
import argparse
import json
import logging
import socket
import threading
import time
import urllib.request
import webbrowser
from .config import StudioConfig


def reserve_port(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(('127.0.0.1', port))
        sock.listen(128)
        sock.setblocking(False)
        return sock
    except OSError:
        sock.close()
        raise


def wait_ready(url: str, timeout: float = 30) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=min(1, max(.01, deadline-time.monotonic()))) as response:
                data = json.load(response)
                if data.get('service') == '8d-engine-studio' and data.get('status') == 'ready':
                    return True
        except (OSError, ValueError):
            pass
        time.sleep(min(.1, max(0, deadline-time.monotonic())))
    return False


def main():
    parser = argparse.ArgumentParser(description='Start 8D Engine Studio locally')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    config = StudioConfig.default()
    logging.basicConfig(filename=config.data_dir/'studio.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    try:
        sock = reserve_port(config.port)
    except OSError:
        print('Port 8765 is already in use. Close the other Studio window or application and try again.')
        return 1
    try:
        from .app import create_app
        import uvicorn
        app = create_app(config)
        url = f'http://127.0.0.1:{config.port}'
        def open_when_ready():
            if wait_ready(url+'/api/health'):
                print('Studio ready: '+url)
                if not args.no_browser:
                    webbrowser.open(url)
            else:
                print('Studio did not become ready. See studio.log in '+str(config.data_dir))
        threading.Thread(target=open_when_ready, daemon=True).start()
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=config.port, log_level='warning'))
        print('8D Engine Studio | Keep this window open. Press Ctrl+C to stop.')
        server.run(sockets=[sock])
        return 0
    finally:
        sock.close()

if __name__ == '__main__':
    raise SystemExit(main())
