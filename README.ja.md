# Kawasemi Colours — 公開再現性パッケージ

**2026-10-18 訂正：** [画像単位の権利情報と方法の説明](docs/RIGHTS_CORRECTION_20261018.md)を更新しました。
第三者画像へのCC BYの誤適用を訂正し、解析対象・数値結果は保持しています。

Kawasemi Colours研究に対応する、公開可能なコード・設定・派生出力の
リポジトリです。第三者論文から取得した電子顕微鏡画像そのものは含めません。
`data/`内の`images/...`というパスはレコードの来歴を保持するための参照であり、
画像ファイルが同梱されていることを意味しません。

## 収録内容

- `model/` — 特徴抽出、合成構造、順方向光学計算、逆設計、grouped評価、
  予備的なMEEP/FDTDコード。
- `experiments/` — 図生成コードと再配布可能な出力。
- `simulator/` — 単体で動作する順方向・逆方向ブラウザツール。
- `data/` — 出典、ライセンス、provenance、fold membership、派生数値。
- `figures/` — ソフトウェア画面、計算プロット、生成構造だけから成る図。
- `supplementary/` — ライセンス、provenance、評価、FDTDの補足表。

## 科学的な位置づけ

コヒーレント散乱計算と合成構造はin-silicoモデルです。MEEP/FDTDは予備的な
full-Maxwell cross-checkであり、収束済みの定量validationや独立実測検証では
ありません。解像度収束、正の対照、独立実測スペクトルとの比較は未完です。

実画像を用いるgrouped解析は、適法に取得した原画像が別途必要なため、この公開
リポジトリだけでは再実行できません。一方、評価対象、派生メタデータ、固定fold、
手順、集計表は収録しており、解析範囲と限界を確認できます。

## 実行例

Python 3.10以上を推奨します。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python model/inverse.py "#2a78b8"
python experiments/explain_and_emergent.py
python experiments/proof_3d.py
python experiments/error_vs_wavelength.py
```

CNNを使う場合だけ、`requirements-ml.txt`を追加で導入してください。MEEP環境は
`environment-meep.yml`と`supplementary/supplementary_S5_fdtd_reproducibility.md`
に分けています。

スクリーンショット付きの操作手順、各パラメータの読み方、モデルの適用範囲は
[Webアプリ使い方ガイド](docs/USER_GUIDE.ja.md)を参照してください。

## 図とライセンス

収録図と除外理由は`figures/README.md`に記載しています。第三者micrographを含む
原稿Figures 1、2、5、10は公開版から除外しました。

コードはMIT License、独自の文書・派生図・表形式メタデータはCC BY 4.0です。
第三者著作物は再ライセンスせず、本リポジトリには収録しません。詳細は
`LICENSES.md`と`data/sources.csv`を参照してください。
