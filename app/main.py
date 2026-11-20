import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from script.features import extract_feature_dict  # 学習時と同じ特徴量の作り方を使う

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "locklog_model.joblib"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# keyの確率がこの値以上なら「鍵の操作」と判定する（学習結果の表を見て調整する）
CONFIDENCE_THRESHOLD = float(os.environ.get("LOCKLOG_THRESHOLD", "0.5"))

MIN_SECONDS, MAX_SECONDS = 0.3, 6.0
MIN_PEAK = 0.003  # これより小さい音は、判定せずに断る（ほぼ無音）
MIN_RATE, MAX_RATE = 8000, 96000
MAX_BODY_BYTES = 2_000_000

state = {}


@asynccontextmanager
async def lifespan(app):
    if MODEL_PATH.is_file():
        state["bundle"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="LockLog API", lifespan=lifespan)


@app.get("/api/health")
def health():
    bundle = state.get("bundle")
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "classes": bundle["classes"] if bundle else [],
        "model_sample_rate": bundle["sample_rate"] if bundle else None,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/api/predict")
async def predict(request: Request, sample_rate: int):
    """モノラル・16bit PCM（リトルエンディアン）の生データを受け取り、判定結果を返す"""
    bundle = state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="モデルが読み込まれていません")
    if not MIN_RATE <= sample_rate <= MAX_RATE:
        raise HTTPException(status_code=422, detail="sample_rate が範囲外です")

    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="音声データが大きすぎます")
    if len(body) < 2 or len(body) % 2 != 0:
        raise HTTPException(status_code=400, detail="音声データの形式が正しくありません")

    samples = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    seconds = len(samples) / sample_rate
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise HTTPException(status_code=422, detail=f"音声の長さが範囲外です: {seconds:.2f}秒")

    if float(np.max(np.abs(samples))) < MIN_PEAK:
        raise HTTPException(status_code=422, detail="音が小さすぎます")

    target_rate = bundle["sample_rate"]
    if sample_rate != target_rate:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_rate)

    features = extract_feature_dict(samples, sr=target_rate)
    columns = bundle["feature_columns"]
    X = pd.DataFrame([[features[name] for name in columns]], columns=columns)
    probabilities = bundle["pipeline"].predict_proba(X)[0]

    classes = bundle["classes"]
    if "key" in classes:
        # 鍵の操作(key)か、それ以外か。keyの確率がしきい値以上なら「key」
        confidence = float(probabilities[classes.index("key")])
        decision = "key" if confidence >= CONFIDENCE_THRESHOLD else "other"
        label = decision
    else:
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        label = classes[best]
        decision = label if confidence >= CONFIDENCE_THRESHOLD else "unknown"
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "decision": decision,
        "threshold": CONFIDENCE_THRESHOLD,
        "probabilities": {name: round(float(p), 4) for name, p in zip(classes, probabilities)},
        "seconds": round(seconds, 2),
    }


# 画面（app/static/index.html）がある場合は、トップページとして配信する
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")