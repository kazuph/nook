
<p align="center">
<img src="assets/logos/nook-logo-01.svg" alt="Nook" width="50%">
<br>
</p>
<p align="right"><small><i>Logo generated by Claude 3.7 Sonnet</i></small></p>

<p align="center">
  <b>ブログ記事・Reddit投稿・GitHub動向・論文を毎日自動収集・要約するWebアプリ</b>
  <br>
  <br>
  <img src="assets/screenshots/nook-demo.gif" alt="Nook Demo" width="700">
</p>

## フォーク版概要

これは[@IMG_5955](https://x.com/IMG_5955)さんが開発したNookからAWS依存部分を無くしてローカルで動かせるようにしたものです。自宅サーバでの運用を想定してます。

## フォーク版導入方法

とりま、Ubuntuで動かす前提での導入方法です。（WSL2でも可）

1. **リポジトリをクローン**
   ```bash
   git clone https://github.com/umiyuki/nook.git
   cd nook
   ```

2. **環境変数の設定**
   `.env` ファイルを作成し、以下の内容を設定
   ```
   GEMINI_API_KEY=your_gemini_api_key
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=your_reddit_user_agent
   OUTPUT_DIR=/home/yourdirectory/output
   ```

3. **依存関係のインストール**
   ```bash
   uv venv
   uv sync
   ```

4. **データ収集実行**
   ```bash
   uv run python main.py
   ```
   .envのOUTPUT_DIR以下に収集した情報がmdファイルで保存されます。

5. **ビューワのサーバ起動**
   ```bash
   uv run python nook/functions/viewer/viewer.py
   ```
   サーバ起動した状態でブラウザからhttp://localhost:8080 にアクセスすると閲覧できます。

6. **Docker Composeでサービスを実行**

   Docker Composeを使用して、データ収集とビューワーサービスを自動化します。
   以下のファイルは作成済みなので、実際にはコマンドのみ実行してください。
   
   まず、Dockerfileを作成します：
   ```bash
   cat > Dockerfile << 'EOF'
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
   EOF
   ```

   次に、docker-compose.ymlを作成します：
   ```bash
   cat > docker-compose.yml << 'EOF'
   version: '3'

   services:
     # ビューワーサービス（常時稼働）
     viewer:
       build:
         context: .
         dockerfile: Dockerfile
       volumes:
         - ${OUTPUT_DIR}:/app/output
       ports:
         - "8080:8080"
       command: uv run python nook/functions/viewer/viewer.py
       restart: always
       environment:
         - GEMINI_API_KEY=${GEMINI_API_KEY}
         - OUTPUT_DIR=/app/output
         - TZ=Asia/Tokyo
       labels:
         org.label-schema.name: "nook-viewer"

     # データ収集サービス（Ofeliaから実行される）
     collector:
       build:
         context: .
         dockerfile: Dockerfile
       volumes:
         - ${OUTPUT_DIR}:/app/output
       environment:
         - GEMINI_API_KEY=${GEMINI_API_KEY}
         - REDDIT_CLIENT_ID=${REDDIT_CLIENT_ID}
         - REDDIT_CLIENT_SECRET=${REDDIT_CLIENT_SECRET}
         - REDDIT_USER_AGENT=${REDDIT_USER_AGENT}
         - OUTPUT_DIR=/app/output
         - TZ=Asia/Tokyo
       command: uv run python main.py
       restart: "no"
       labels:
         org.label-schema.name: "nook-collector"
         ofelia.enabled: "true"
         # 日本時間の朝7時に実行
         ofelia.job-exec.datajob.schedule: "0 7 * * *"
         ofelia.job-exec.datajob.command: "uv run python main.py"

     # Ofeliaスケジューラー
     ofelia:
       image: mcuadros/ofelia:latest
       depends_on:
         - collector
         - viewer
       command: daemon --docker
       volumes:
         - /var/run/docker.sock:/var/run/docker.sock:ro
       environment:
         - TZ=Asia/Tokyo
       restart: always
       labels:
         org.label-schema.name: "nook-scheduler"
   EOF
   ```

   Docker Composeを起動します：
   ```bash
   docker-compose up -d
   ```

   これで以下の機能が自動的に実行されます：
   - ビューワーサービスが常時稼働（http://localhost:8080 でアクセス可能）
   - データ収集が毎日朝7時に自動実行
   - コンテナの再起動も自動的に行われる

   コンテナの状態を確認するには：
   ```bash
   docker-compose ps
   ```

   ログを確認するには：
   ```bash
   docker-compose logs -f
   ```

↓以下、オリジナルのREADMEです。

## 🌟 概要

Nookは、テック系の最新情報を自動的に収集し、要約するWebアプリです。

Reddit、Hacker News、GitHub Trending、技術ブログ、学術論文など多様な情報源から毎日自動的にコンテンツを収集し、LLMを使用して日本語で要約します。シンプルなWebインターフェースで閲覧でき、各トピックに対してチャット形式でフォローアップ質問も可能です。

1日平均12円の運用費*で情報収集にかかる時間を大幅に削減します。

## 🎬 外観

### PC

<p align="center">
  <img src="assets/screenshots/web-screenshot.webp" alt="Nook Demo" width="700">
</p>

### モバイル

<p align="center">
  <img src="assets/screenshots/mobile-screenshot.webp" width="40%">
</p>

## ✨ 特徴

### 🔄 多様な情報源からの自動収集・要約
- **Hacker News**: Hacker Newsから最新のテクノロジーニュースを収集
- **GitHub Trending**: GitHubで各言語ごとの人気リポジトリを取得
- **Reddit**: r/MachineLearningなどのサブレディットから人気投稿を収集して要約
- **RSS**: RSSフィードを監視し、更新があれば取得して要約
- **arXiv論文**: HuggingFaceでキュレーションされた最新の学術論文を自動収集し、HTMLページがあれば論文ごと読んで要約

### 💬 インタラクティブなチャット機能
- **フォローアップ質問**: 要約された内容について、さらに詳しく質問可能（グラウンディングあり）

### 🌐 シンプルなWebインターフェース
- **日付別整理**: 収集した情報を日付ごとに整理して表示
- **レスポンシブデザイン**: モバイルデバイスでも快適に閲覧可能

### 🔧 簡単なデプロイと拡張性
- **AWS CDK**: インフラをコードとして管理し、簡単にデプロイ
- **モジュラー設計**: 新しい情報源の追加が容易
- **カスタマイズ可能**: 設定ファイルの編集で情報源を変更可能

## 🚀 導入方法

### 前提条件

<details>
<summary>必要なツール（クリックして展開）</summary>

- AWS CLI と AWS CDK
  - CDKでのデプロイには強めの権限が必要です。
  - 先に`aws configure`でシークレットなどの認証情報を登録する必要があります。
- Python 3.11
- Dockerイメージのビルドが可能な環境
  - `tech_feed`のセットアップにローカルのDockerを使用するらしいです。
- 以下のAPIキー:
  - Google Gemini API キー（有償）
  - Reddit API キー（クライアントID、クライアントシークレット）

</details>

### クイックスタート

1. **リポジトリをクローン**
   ```bash
   git clone https://github.com/discus0434/nook.git
   cd nook
   ```

2. **環境変数の設定**
   `.env` ファイルを作成し、以下の内容を設定
   ```
   GEMINI_API_KEY=your_gemini_api_key
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=your_reddit_user_agent
   ```

3. **依存関係のインストール**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windowsの場合: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **AWS CDKを使ったデプロイ**
   ```bash
   make cdk-deploy  # `cdk deploy`ではない
   ```

5. **アプリにアクセス**
   デプロイ完了後は、情報収集・要約用のLambdaが毎朝9時に実行されるようになり、その実行後にコンソールに表示される `viewer` 関数のURLからWebインターフェースにアクセスすると、ニュースが閲覧可能になります。

## 🏗️ アーキテクチャ

```mermaid
graph TD
    %% スタイル定義
    classDef lambdaStyle fill:#FF9900,stroke:#FF9900,stroke-width:2px,color:white;
    classDef s3Style fill:#569A31,stroke:#569A31,stroke-width:2px,color:white;
    classDef eventStyle fill:#FF4F8B,stroke:#FF4F8B,stroke-width:2px,color:white;
    classDef apiStyle fill:#232F3E,stroke:#232F3E,stroke-width:2px,color:white;
    classDef userStyle fill:#3B48CC,stroke:#3B48CC,stroke-width:2px,color:white;

    %% ノード定義
    User[ユーザー]:::userStyle
    EventBridge[Amazon EventBridge<br>毎日定時実行]:::eventStyle
    S3[Amazon S3<br>データストレージ]:::s3Style
    GeminiAPI[Google Gemini API]:::apiStyle

    %% Lambda関数
    RedditExplorer[Lambda: reddit_explorer<br>Redditの人気投稿収集・要約]:::lambdaStyle
    HackerNews[Lambda: hacker_news<br>Hacker News記事収集]:::lambdaStyle
    GitHubTrending[Lambda: github_trending<br>GitHubトレンド収集]:::lambdaStyle
    TechFeed[Lambda: tech_feed<br>技術ブログRSS収集・要約]:::lambdaStyle
    PaperSummarizer[Lambda: paper_summarizer<br>arXiv論文収集・要約]:::lambdaStyle
    Viewer[Lambda: viewer<br>Webインターフェース・チャット機能]:::lambdaStyle

    %% 外部情報源
    Reddit[Reddit API]
    HNSource[Hacker News API]
    GitHub[GitHub API]
    RSSFeeds[技術ブログRSSフィード]
    ArXiv[arXiv API]

    %% 接続関係
    EventBridge --> RedditExplorer
    EventBridge --> HackerNews
    EventBridge --> GitHubTrending
    EventBridge --> TechFeed
    EventBridge --> PaperSummarizer

    RedditExplorer --> Reddit
    HackerNews --> HNSource
    GitHubTrending --> GitHub
    TechFeed --> RSSFeeds
    PaperSummarizer --> ArXiv

    RedditExplorer --> GeminiAPI
    TechFeed --> GeminiAPI
    PaperSummarizer --> GeminiAPI
    Viewer --> GeminiAPI

    RedditExplorer --> S3
    HackerNews --> S3
    GitHubTrending --> S3
    TechFeed --> S3
    PaperSummarizer --> S3

    S3 --> Viewer
    User --> Viewer
```

## 🛠️ カスタマイズ

### 情報源の追加・変更

各情報源は設定ファイルで管理されており、簡単にカスタマイズできます:

#### Reddit

`nook/lambda/reddit_explorer/subreddits.toml` を編集して、収集するサブレディットを変更できます。

```toml
[[subreddits]]
name = "MachineLearning"

# 新しいサブレディットを追加
[[subreddits]]
name = "LocalLLaMA"
```

#### GitHub Trending

`nook/lambda/github_trending/languages.toml` を編集して、追跡するプログラミング言語を変更できます。

```toml
[[languages]]
name = "python"

# 新しい言語を追加
[[languages]]
name = "javascript"
```

#### 技術ブログ

`nook/lambda/tech_feed/feed.toml` を編集して、RSSフィードを追加/変更できます。

```toml
[[feeds]]
key = "new_blog"
name = "My Favorite Tech Blog"
url = "https://example.com/feed.xml"
```

### UI設定

`nook/lambda/viewer/templates/index.html` を編集して、UIをカスタマイズすることもできます。

## 🔄 使用例

### 日々のテックニュース収集

朝一番にNookにアクセスすれば、昨日1日のうちに発生した以下の情報をだいたい把握できます：

- 🔥 Redditで話題になっている最新の技術トピック
- 📈 GitHub上で人気急上昇中のプロジェクト
- 📰 Hacker Newsの注目記事
- 📚 最新のAI研究論文と主要な発見
- 🌐 更新されたブログ記事

すべて日本語で要約されているため、短時間で効率的に情報をキャッチアップできます。

### フォローアップ質問

記事/リポジトリの詳細について追加で知りたいことがある場合は、チャットインターフェースを使ってフォローアップ質問が可能です。

<p align="center">
  <img src="assets/screenshots/chat-screenshot.webp"  width="400">
</p>

### API連携

Nookが収集した情報はS3へ日毎に保存されるので、他のアプリケーションやワークフローと連携することも可能です：

```python
import boto3
import json
from datetime import date

# S3からのデータ取得の例
s3 = boto3.client('s3')
date_str = date.today().strftime("%Y-%m-%d")
response = s3.get_object(
    Bucket="your-nook-bucket",
    Key=f"tech_feed/{date_str}.md"
)
content = response["Body"].read().decode("utf-8")
print(content)
```

## 📄 ライセンス

このプロジェクトはGNU Affero General Public License v3.0の下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 📬 お問い合わせ

- GitHub Issues: 質問、バグ報告、機能リクエスト
- X: [@IMG_5955](https://x.com/IMG_5955)

<p><small>[*]ただし、使用状況や設定によって費用が変動する可能性があり、実際のコストは保証できません。自己責任での使用をお願いします。</small></p>
