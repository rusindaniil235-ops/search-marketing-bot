"""Small local web server for the chatbot; no external dependencies."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

from bot import available_topics, classify


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "SearchMarketingBot/1.0"

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        path = urlparse(self.path).path
        if path == "/":
            self._send_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
        elif path == "/health":
            self._send_json({"status": "ok"})
        elif path == "/api/intents":
            self._send_json({"topics": available_topics()})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        if urlparse(self.path).path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "invalid content length"}, HTTPStatus.BAD_REQUEST)
            return
        if length <= 0 or length > 8192:
            self._send_json({"error": "request body must be 1..8192 bytes"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = payload.get("text", "")
            if not isinstance(text, str):
                raise TypeError("text must be a string")
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(classify(text).to_dict())

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local chatbot web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

