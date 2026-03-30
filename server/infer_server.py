from __future__ import annotations
import argparse
import threading
import time
import pickle
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
import uvicorn
from builders import build_model
from builders import build_optimizer
from config import load_config
import numpy as np
INFER_LOCK = threading.Lock()

def create_app(handler, endpoint: str = "/infer") -> FastAPI:
    app = FastAPI()
    @app.post(endpoint)
    async def infer(request: Request) -> Response:
        body = await request.body()
        if not body:
            return Response(status_code=400)
        try:
            data = pickle.loads(body)
        except Exception:
            return Response(status_code=400)
        start_time = time.time()
        with INFER_LOCK:
            output = handler(data)
        infer_time = time.time() - start_time
        if isinstance(output, dict):
            result = dict(output)
            result["infer_time"] = infer_time
        else:
            result = {"action_list": output, "infer_time": infer_time}
        return Response(content=pickle.dumps(result), media_type="application/octet-stream")
    return app

class InferPipeline:
    def __init__(self, cfg):
        self._model = build_model(cfg)
        self._optimizer = build_optimizer(cfg)

    def __call__(self, request: dict) -> dict:
        raw_actions = self._model.infer_actions(request)
        optimized_actions = self._optimizer.optimize(raw_actions)
        return {
            "action_list": optimized_actions,
            "raw_action_list": raw_actions,
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_cloth.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pipeline = InferPipeline(cfg)
    app = create_app(pipeline, endpoint=cfg.server.endpoint)
    print(f"[infer_server] listening on {cfg.server.host}:{cfg.server.port}")
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, access_log=False)

if __name__ == "__main__":
    main()
