from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from flask import Flask, Response, request
from waitress import serve


INDEX_BODY = (Path(__file__).resolve().parents[2] / "examples/http_router_sim/index.html").read_text()
app = Flask(__name__)
logging.getLogger("waitress.queue").setLevel(logging.ERROR)


@app.get("/")
def root() -> Response:
    return Response(INDEX_BODY, mimetype="text/html")


@app.get("/health")
def health() -> Response:
    return Response('{"status":"ok"}', mimetype="application/json")


@app.get("/users/<user_id>")
def user(user_id: str) -> Response:
    body = json.dumps({"id": user_id, "name": f"User {user_id}"}, separators=(",", ":"))
    return Response(body, mimetype="application/json")


@app.get("/users/<user_id>/posts/<post_id>")
def user_post(user_id: str, post_id: str) -> Response:
    body = json.dumps({"user_id": user_id, "post_id": post_id}, separators=(",", ":"))
    return Response(body, mimetype="application/json")


@app.post("/echo")
def echo() -> Response:
    body = json.dumps({"echo": request.get_json(force=True, silent=False)}, separators=(",", ":"))
    return Response(body, mimetype="application/json")


@app.get("/assets/<path:suffix>")
def assets(suffix: str) -> Response:
    return Response(f"serving asset: {suffix}\n", mimetype="text/plain")


@app.errorhandler(404)
def not_found(_: object) -> Response:
    return Response("not found\n", status=404, mimetype="text/plain")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    serve(app, host="127.0.0.1", port=args.port, threads=args.threads)


if __name__ == "__main__":
    main()
