from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
import uvicorn


INDEX_BODY = (Path(__file__).resolve().parents[2] / "examples/http_router_sim/index.html").read_text()
app = FastAPI()


@app.get("/")
def root() -> HTMLResponse:
    return HTMLResponse(INDEX_BODY)


@app.get("/health")
def health() -> Response:
    return Response(content='{"status":"ok"}', media_type="application/json")


@app.get("/users/{user_id}")
def user(user_id: str) -> Response:
    body = json.dumps({"id": user_id, "name": f"User {user_id}"}, separators=(",", ":"))
    return Response(content=body, media_type="application/json")


@app.get("/users/{user_id}/posts/{post_id}")
def user_post(user_id: str, post_id: str) -> Response:
    body = json.dumps({"user_id": user_id, "post_id": post_id}, separators=(",", ":"))
    return Response(content=body, media_type="application/json")


@app.post("/echo")
async def echo(request: Request) -> Response:
    body = json.dumps({"echo": await request.json()}, separators=(",", ":"))
    return Response(content=body, media_type="application/json")


@app.get("/assets/{suffix:path}")
def assets(suffix: str) -> PlainTextResponse:
    return PlainTextResponse(f"serving asset: {suffix}\n")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def missing(path: str) -> PlainTextResponse:
    return PlainTextResponse("not found\n", status_code=404)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18083)
    args = parser.parse_args()

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
