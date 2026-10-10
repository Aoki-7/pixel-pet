#!/usr/bin/env python3
"""
NCA 训练脚本：从目标图片训练 Neural Cellular Automata。
模型结构与 JS 引擎 (src/nca.js) 完全一致：
  感知 48维 (identity+sobelX+sobelY) → FC1: 48→128(ReLU) → FC2: 128→16
  残差更新 state+=delta, delta_clamp=±1.0, state_clamp=±2.0
  70% 随机更新 + 3×3 maxpool(alpha)>0.1 活性掩码

输出: src/weights.json (可被 JS 引擎 loadWeights 加载)
"""
import os, sys, json, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 配置 ----
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID    = 40
CHANS   = 16
HIDDEN  = 128
STEPS   = 80          # 每次前向传播步数
ITERS   = 4000        # 训练轮数
LR      = 0.002
POOL    = 32           # 样本池大小
DEV     = "cuda" if torch.cuda.is_available() else "cpu"
IMG     = os.path.join(ROOT, "assets", "target.png")
OUT_W   = os.path.join(ROOT, "src", "weights.json")
OUT_D   = os.path.join(ROOT, "train", "previews")
os.makedirs(OUT_D, exist_ok=True)

# ---- 1. 像素化 ----
def load_target(path, size):
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    # 白底 → 透明
    if arr[:,:,3].min() == 255 and arr[:,:,3].max() == 255:
        white = (arr[:,:,:3] > 235).all(axis=2)
        arr[white, 3] = 0
    img = Image.fromarray(arr)
    # 居中裁剪为正方形
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # (H,W,4)
    return arr

# ---- 2. NCA 模型 ----
class NCA(nn.Module):
    def __init__(self, ch=16, hid=128):
        super().__init__()
        self.ch = ch
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32)/8
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32)/8
        ident   = torch.tensor([[0,0,0],[0,1,0],[0,0,0]],dtype=torch.float32)
        self.register_buffer("k_id", ident.view(1,1,3,3))
        self.register_buffer("k_sx", sobel_x.view(1,1,3,3))
        self.register_buffer("k_sy", sobel_y.view(1,1,3,3))
        self.fc1 = nn.Linear(ch*3, hid)
        self.fc2 = nn.Linear(hid, ch)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.zeros_(self.fc2.bias)

    def perceive(self, s):
        B,C,H,W = s.shape
        i  = F.conv2d(s, self.k_id.expand(C,1,3,3), padding=1, groups=C)
        sx = F.conv2d(s, self.k_sx.expand(C,1,3,3), padding=1, groups=C)
        sy = F.conv2d(s, self.k_sy.expand(C,1,3,3), padding=1, groups=C)
        return torch.cat([i, sx, sy], dim=1)   # (B, 3C, H, W)

    def step(self, s):
        p = self.perceive(s)                      # (B,3C,H,W)
        p = p.permute(0,2,3,1)                     # (B,H,W,3C)
        u = self.fc2(F.relu(self.fc1(p)))         # (B,H,W,C)
        u = u.permute(0,3,1,2)                     # (B,C,H,W)
        u = u.clamp(-1.0, 1.0)                    # delta clamp
        mask = (torch.rand(s.shape[0],1,s.shape[2],s.shape[3],device=s.device)<0.7).float()
        s = s + u * mask                           # 残差更新 (rate=1.0)
        alive = F.max_pool2d(s[:,3:4],3,stride=1,padding=1) > 0.1
        s = s * alive.float()
        s = s.clamp(-2.0, 2.0)                    # state clamp
        return s

    def forward(self, s, steps):
        for _ in range(steps):
            s = self.step(s)
        return s

    def seed(self, batch, size, dev):
        s = torch.zeros(batch, self.ch, size, size, device=dev)
        c = size // 2
        s[:, 3, c, c] = 1.0   # alpha=1
        return s

# ---- 3. 保存预览 ----
def save_img(state, path):
    rgba = state[:4].detach().clamp(0,1).permute(1,2,0).cpu().numpy()
    rgba = (rgba*255).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)

# ---- 4. 训练 ----
def train(target_arr):
    print(f"Device: {DEV}  |  目标 {GRID}×{GRID}")
    transparency = (target_arr[:,:,3] < 0.1).sum() / (GRID*GRID)
    print(f"透明像素: {transparency:.1%}  有色像素: {1-transparency:.1%}")

    model = NCA(CHANS, HIDDEN).to(DEV)
    print(f"参数数: {sum(p.numel() for p in model.parameters())}")

    # 目标: (1,4,H,W)
    tgt = torch.tensor(target_arr[:,:,:4].transpose(2,0,1), device=DEV).unsqueeze(0)
    alpha_w = (tgt[:,3:4] > 0.1).float()  # 有色区域权重

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS)

    # Pool
    pool = [model.seed(1, GRID, DEV) for _ in range(POOL)]
    seed = model.seed(1, GRID, DEV)

    t0 = time.time()
    for it in range(ITERS):
        opt.zero_grad()
        # 从 pool 采样或从种子开始
        idx = np.random.randint(POOL)
        if it < 100 or np.random.random() < 0.1:
            state = seed.clone()
        else:
            state = pool[idx].clone()

        steps = np.random.randint(1, STEPS + 1)
        out = model(state, steps)

        # 损失：只算 RGBA，透明区域权重低
        loss = ((out[:, :4] - tgt) ** 2 * alpha_w).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        # 更新 pool
        pool[idx] = out.detach()

        if it % 200 == 0:
            elapsed = time.time() - t0
            print(f"[{it:4d}/{ITERS}] loss={loss.item():.6f}  lr={sch.get_last_lr()[0]:.6f}  {elapsed:.0f}s")

        if it % 500 == 0 or it == ITERS - 1:
            save_img(out[0], os.path.join(OUT_D, f"step_{it:05d}.png"))

    # 最终预览：从种子跑完整步数
    with torch.no_grad():
        final = model(seed.clone(), STEPS)
        save_img(final[0], os.path.join(OUT_D, "final.png"))

    # 导出权重
    w = {
        "w1": model.fc1.weight.detach().cpu().numpy().tolist(),
        "b1": model.fc1.bias.detach().cpu().numpy().tolist(),
        "w2": model.fc2.weight.detach().cpu().numpy().tolist(),
        "b2": model.fc2.bias.detach().cpu().numpy().tolist(),
    }
    with open(OUT_W, "w") as f:
        json.dump(w, f)
    print(f"\n训练完成，用时 {time.time()-t0:.1f}s")
    print(f"权重已导出: {OUT_W}")
    save_img(torch.tensor(target_arr[:,:,:4].transpose(2,0,1),device=DEV), os.path.join(OUT_D, "target.png"))

if __name__ == "__main__":
    target = load_target(IMG, GRID)
    save_img(torch.tensor(target[:,:,:4].transpose(2,0,1),device=DEV), os.path.join(OUT_D, "target.png"))
    print("目标图片已像素化，保存到 train/previews/target.png")
    train(target)
