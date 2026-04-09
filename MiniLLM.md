# MiniLLM
トランスフォーマーベースの小型言語モデル（シンプル実装）

- 軽量なLLMアーキテクチャ
- マルチヘッド自己注意機構
- 残差接続 + レイヤー正規化
- GELU活性化関数
- 学習・推論コード付き

## モデル構成
- レイヤー数: 2
- 隠れ次元: 128
- アテンションヘッド: 2
- 総パラメータ: 約XX K

## 実行方法
```bash
pip install -r requirements.txt
python mini_llm.py