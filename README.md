lock,door,unlock,other
,28,
# データ追加ステップ
録音 → raw\... に置く
切り出し → split_clips.py（まず --dry-run）
学習 → extract_features.py → train_model.py

仮想サーバ起動
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000


確信度を調べるもの
Get-ChildItem .\dataset\lock\*.wav | Select-Object -First 10 | ForEach-Object {
  $_.Name
  .\.venv\Scripts\python.exe .\script\send_wav.py $_.FullName | Select-String '"decision"|"confidence"'
}

split_clips.py
.\.venv\Scripts\python.exe .\script\split_clips.py --sequence unlock,lock --dry-run

.\.venv\Scripts\python.exe .\script\split_clips.py --sequence door --dry-run