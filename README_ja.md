# ComfyUI FastH3 OmniRef Accel

[English](README.md) | 日本語

> **非公式**のノード集です。FastVideo / Hao AI Lab / Nuva Lab とは関係ありません。FastH3 OmniRef のモデル本体は同梱しません。

FastH3 OmniRef（MiniMax H3 の 8 step 蒸留 Ref2VA）を RTX 4070 12GB で速く回すための ComfyUI ノードとワークフローです。
8 step はそのままで、生成解像度・テキストエンコーダ・デコード周りの無駄を削ります。

| 5 秒・124 frames・音声付き、サーバー起動直後の 1 本目 | 時間 |
|---|---|
| 公式テンプレート（VSA10、1024x1792 を直接生成） | 459.6 秒 |
| **このパック（1024x1792 出力）** | **64.6〜66.8 秒（約 7 倍）** |

同じサーバーで 2 本目以降（プロンプト変更あり）は約 62 秒。RTX 4070 12GB / RAM 48GB で計測（[docs/VALIDATION.md](docs/VALIDATION.md)）。

## 申し送り事項（画質）

- **顔が小さく映る構図（横長の引き、全身、遠景）では画質が落ちます。** 公式の直接生成では潰れない小さな顔が崩れることがあります。
  前半 5 step を出力の 1/4 で生成するため、その段階で顔が潜在の数マスにしかならず、後半で直りません。
- 想定用途はバストアップ〜膝上くらいの縦長キャラ動画です。
- 解像度が違うため、同じ seed でも公式構成とは動き・構図が変わります。

## 仕組み

1. **FastH3 Accel Sizes** に最終解像度を入れると、生成サイズ（出力の 1/2、短辺は `min_short_side` 以上）、前半サイズ（その 1/2）、ESRGAN 後の縮小率、切り抜きサイズを計算します。
2. 前半 5 step を低解像度で生成し、LBH 3D 潜在アップスケーラーで 2 倍、後半 3 step を生成します（音声は 8 step 分そのまま続きます）。
3. H3 VAE でデコードし、Real-ESRGAN animevideov3（x4）→ 縮小 → 中央切り抜きで最終解像度ぴったりに出力します（縦横比は引き伸ばしません）。

| ノード | 内容 | 出力への影響 |
|---|---|---|
| FastH3 Accel Sizes (from output size) | 最終解像度から各サイズを計算。最終解像度が小さいときは生成サイズの短辺を `min_short_side`（既定 512）まで引き上げる | - |
| FastH3 Accel Text Encoder BF16 | ComfyUI は TE の LLM を FP32 固定で計算する。BF16 で計算させる | 条件ベクトルがわずかに変わる（同 seed で見た目同等） |
| FastH3 Accel Prefetch DiT Weights | mmap された DiT の重みを TE 計算中に別スレッドで先読み（1 本目の約 6 秒を隠す） | なし |
| FastH3 Accel Chunk FeedForward | FFN をトークン方向に 8 分割（512x896・8 step で約 10 秒短縮） | なし |
| FastH3 Accel Sparse Producer Reuse | 各サンプラー初回 step で疎アテンションが QKV/Gate を 2 回射影するのを 1 回にする | なし |
| FastH3 Accel LBH Upscale + Continue | LBH で映像潜在を拡大して再ノイズ化、音声は途中状態を引き継ぐ。LBH の読み込みを高速化 | なし |
| FastH3 Accel Decode + Upscale + NVENC Save | VAE の時間チャンクごとに ESRGAN → 縮小 → 切り抜き → NVENC を重ねて流す。VAE タイル重なり 32px | 重なり 32px で Y-PSNR 38〜41dB（目視差なし） |

「なし」は、同じ seed でビット単位の一致を確認したものです。

テキストエンコーダは nvfp4_awq ファイル（15.7GB）を使います。RTX 4070 では FP4 演算の効果はありませんが、
int8（26GB）だと DiT 21GB と合わせて 48GB RAM のキャッシュに収まらず、TE を毎回ディスクから読み直すためです。

## 必要なもの

- Windows、NVIDIA GPU（RTX 4070 12GB で検証）、RAM 48GB で検証
- ネイティブ MiniMax H3（Reference to Video）と BlockSparseAttention を持つ ComfyUI。検証: ComfyUI `1a82e6d0`（2026-10-21）、Python 3.13.13、PyTorch 2.14.0+cu130、comfy-kitchen 0.2.39、PyAV 19.0.2
- h264_nvenc が使える PyAV（ComfyUI 標準の PyAV で可）
- ComfyUI の Dynamic VRAM を有効のまま使用。`--disable-pinned-memory` で検証

## モデル

モデルは同梱も自動ダウンロードもしません。各配布元の条件を確認して配置してください。

| ファイル | 置き場所 | 配布元 |
|---|---|---|
| fasth3_omniref_8step_pruned_int8_convrot.safetensors | models/diffusion_models | [FastVideo/FastH3-Omni-Comfy](https://huggingface.co/FastVideo/FastH3-Omni-Comfy) |
| qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | models/text_encoders | 同上 |
| minimax_h3_video_vae_int8_convrot.safetensors | models/vae | 同上 |
| minimax_h3_audio_vae_fp32.safetensors | models/vae | 同上 |
| minimax_h3_latent_upscaler_3d_fp16.safetensors | models/latent_upscale_models | [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) |
| realesr-animevideov3.pth | models/upscale_models | [xinntao/Real-ESRGAN v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0) |

LBH の現行配布名 `minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors` も setup の確認で受け付けますが、検証に使った重みと同一かは未確認です。
検証に使ったファイルの SHA-256 は [docs/VALIDATION.md](docs/VALIDATION.md) にあります。

## インストール

ComfyUI を停止してから、このフォルダの `setup.bat` を実行します。`ComfyUI/custom_nodes` 内に置いた場合は ComfyUI と Python を自動検出します。

```bat
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --check
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --install-dependencies
```

- 第 2 引数は **その ComfyUI の Python** です。portable の `python_embeded`、ComfyUI 内／親フォルダの `.venv`・`venv` を探し、候補が 1 つでなければ指定が必要です。グローバル Python は拒否します。
- `--check` は読み取りのみです。必要なネイティブノード、h264_nvenc、モデル 6 種の配置を確認します。
- 引数なしの実行で `ComfyUI/custom_nodes/ComfyUI-FastH3-OmniRef-Accel` にコピーします。既に別内容のフォルダがある場合は上書きせず停止します。
- `--install-dependencies` は requirements.txt のうち足りないものだけを入れます（pip が無い uv venv では uv を使用）。
- 共有のモデルフォルダは ComfyUI 標準の `extra_model_paths.yaml`、setup だけで読む場合は `--extra-model-paths PATH` で指定します。
- setup 中に `nodes_replacements.py` の IMPORT FAILED 警告が出ますが、サーバーを起動しない確認のためで、生成には影響しません。

## 使い方

1. `workflows/fasth3_omniref_accel.json` を ComfyUI にドラッグします（API 用は `fasth3_omniref_accel.api.json`）。
2. **Output size**（FastH3 Accel Sizes）の `output_width` / `output_height` に最終解像度を入れます。
3. **Reference image** に参照画像を、**Reference to Video (prompt)** にプロンプトを設定して実行します。プロンプトは `<Picture 1>` で参照画像を指します（例: `workflows/example_prompt.txt`）。

| 最終解像度 | 前半 → 後半 | 備考 |
|---|---|---|
| 1024x1792 | 256x448 → 512x896 | 検証の基準 |
| 1792x1024 | 448x256 → 896x512 | 引きの構図では顔が崩れやすい |
| 720x1280 | 256x448 → 512x896 | 1/2 だと 360x640 で小さすぎるため短辺 512 に引き上げ、731x1280 から中央を切り抜き |

出力は ComfyUI の `output/video/FastH3_OmniRef_Accel_*.mp4`（H.264 NVENC CQ18、AAC）です。

## 注意

- 前半の低解像度 step 数は SplitSigmas の `step`（既定 5）。縦長のアップ構図では 6 でも問題ありませんでしたが（約 5 秒速い）、横長では音声劣化判定で NG が出たため既定は 5 です。
- 先読みノードは DiT のファイルページをプロセスのメモリとして数えさせるため、タスクマネージャー等の RAM 表示が約 21GB 増えます（中身は元々 OS のファイルキャッシュにあるもの）。
- 1 回の生成で 1 本の動画を保存します（バッチ 1 のみ）。

## ライセンス

コードは GPL-3.0（[LICENSE](LICENSE)）。同梱・移植したコードの出典は [NOTICE.md](NOTICE.md) を参照してください。
モデルの重みにはそれぞれの配布元の条件が適用されます（FastH3 OmniRef: MiniMax-H3 Community License、LBH・Real-ESRGAN は各配布ページの条件）。
