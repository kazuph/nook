FROM python:3.11-slim

WORKDIR /app

# uvをインストール
RUN pip install --no-cache-dir uv

# 依存関係ファイルをコピー
COPY pyproject.toml .

# アプリケーションコードをコピー
COPY . .

# uvを使用して依存関係をインストール
RUN uv venv .venv
RUN uv sync

# 環境変数の設定
ENV PATH="/app/.venv/bin:$PATH"

# デフォルトコマンド（必要に応じてオーバーライド）
CMD ["uv", "run", "python", "main.py"]
