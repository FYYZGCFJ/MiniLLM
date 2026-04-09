"""
MiniLLM: トランスフォーマーを基盤とした超小型言語モデル
シンプルな実装で学習・デモンストレーションに最適
GitHub: あなたのユーザー名/MiniLLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ハイパーパラメータ設定
VOCAB_SIZE = 512    # 語彙サイズ
DIM_MODEL = 128     # モデルの埋め込み次元
NUM_HEADS = 2       # マルチヘッドアテンションのヘッド数
NUM_LAYERS = 2      # トランスフォーマーブロックの積み重ね数
SEQ_LEN = 64        # 最大シーケンス長

# 小型トランスフォーマーブロック
class MiniBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # マルチヘッド自己注意機構
        self.attn = nn.MultiheadAttention(DIM_MODEL, NUM_HEADS, batch_first=True)
        # レイヤー正規化
        self.norm1 = nn.LayerNorm(DIM_MODEL)
        self.norm2 = nn.LayerNorm(DIM_MODEL)
        # フィードフォワードネットワーク
        self.fc = nn.Sequential(
            nn.Linear(DIM_MODEL, 4 * DIM_MODEL),
            nn.GELU(),
            nn.Linear(4 * DIM_MODEL, DIM_MODEL)
        )

    def forward(self, x):
        # 自己注意計算 + 残差接続
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # フィードフォワード計算 + 残差接続
        fc_out = self.fc(x)
        x = self.norm2(x + fc_out)
        return x

# メインとなるMiniLLMモデル
class MiniLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # トークン埋め込み層
        self.emb = nn.Embedding(VOCAB_SIZE, DIM_MODEL)
        # 位置埋め込み（学習可能パラメータ）
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, DIM_MODEL))
        # トランスフォーマーブロックを積み重ねる
        self.layers = nn.ModuleList([MiniBlock() for _ in range(NUM_LAYERS)])
        # 最終レイヤー正規化
        self.norm = nn.LayerNorm(DIM_MODEL)
        # 出力線形層（次トークン予測）
        self.head = nn.Linear(DIM_MODEL, VOCAB_SIZE)

    def forward(self, x):
        B, L = x.shape
        # 埋め込み + 位置埋め込みを加算
        x = self.emb(x) + self.pos_emb[:, :L]
        # トランスフォーマー層を順伝播
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # 語彙ごとの確率分布を出力
        logits = self.head(x)
        return logits

# テキスト生成用関数
def generate(model, prompt, max_len=32):
    model.eval()
    with torch.no_grad():
        # デモ用：ランダムトークンで初期化
        x = torch.randint(0, VOCAB_SIZE, (1, len(prompt)))
        for _ in range(max_len):
            logits = model(x)
            next_token = logits.argmax(-1)[:, -1:]
            x = torch.cat([x, next_token], dim=1)
    return x

if __name__ == "__main__":
    model = MiniLLM()
    print("MiniLLM モデルのロードが完了しました！")
    print(f"総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # テスト生成
    test_out = generate(model, "南から風が吹く")
    print("生成完了、出力トークン形状:", test_out.shape)