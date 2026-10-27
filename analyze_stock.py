import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = "data_bond"
OUT_DIR  = "analysis_bond"
os.makedirs(OUT_DIR, exist_ok=True)

ROLL_WINDOW = 750     # 滚动回归窗口（约3年）
MIN_PERIODS = 250
Z_SMOOTH    = 20


def _fmt(v, fmt="{:.2f}", na="—"):
    try:
        if v is None: return na
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv): return na
        return fmt.format(fv)
    except Exception:
        return na


def _add_box(ax, text, loc="upper left", fontsize=8.0):
    xy = {"upper left": (0.015, 0.975), "lower left": (0.015, 0.025),
          "upper right": (0.985, 0.975), "lower right": (0.985, 0.025)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, fontsize=fontsize,
            ha=ha, va=va, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#888888", alpha=0.90, lw=0.8))


# ================= 1. 读数据 =================
def load_all():
    def _read(name):
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少文件: {path}")
        df = pd.read_csv(path)
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        return df.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)

    rate = _read("中美国债收益率")
    cpi  = _read("CPI")
    pmi  = _read("PMI")
    fdr  = _read("FDR007")

    # 主日历：中美国债收益率（2005年起，约5000个交易日）
    base = rate[["日期", "中国国债收益率10年", "中国国债收益率30年",
                 "美国国债收益率10年", "中美利差_10Y"]].copy()
    base = base.rename(columns={"中国国债收益率10年": "Y10",
                                 "中国国债收益率30年": "Y30"})
    base = base.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)

    # 月频 CPI/PMI -> 日频：按月份末映射，前向填充
    def monthly_to_daily(mdf, col):
        m = mdf.copy()
        m["month"] = m["日期"].dt.to_period("M")
        m = m.drop_duplicates("month", keep="last").set_index("month")[col]
        base_month = base["日期"].dt.to_period("M")
        return base_month.map(m).ffill().values

    base["CPI同比"] = monthly_to_daily(cpi, "CPI同比")
    base["制造业PMI"] = monthly_to_daily(pmi, "制造业PMI")

    # FDR007：日频，2021年起有值，早期为 NaN
    fdr_idx = fdr.set_index("日期")["FDR007"]
    base["FDR007"] = fdr_idx.reindex(base["日期"], method="ffill").values

    base = base.dropna(subset=["Y10", "Y30"]).reset_index(drop=True)
    return base


# ================= 2. 滚动 OLS 残差 =================
def rolling_residual(y, X, window=ROLL_WINDOW, min_periods=MIN_PERIODS):
    """滚动 OLS：y ~ const + X，返回拟合值、残差、系数
    只用非缺失样本拟合，缺失行跳过"""
    n = len(y)
    fitted = np.full(n, np.nan)
    coefs = np.full((n, X.shape[1] + 1), np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        if i - start + 1 < min_periods:
            continue
        seg_y = y[start:i+1]
        seg_X = X[start:i+1]
        mask = ~np.isnan(seg_y) & ~np.isnan(seg_X).any(axis=1)
        if mask.sum() < min_periods:
            continue
        yy = seg_y[mask]
        XX = np.column_stack([np.ones(mask.sum()), seg_X[mask]])
        try:
            beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
            # 用当前行的 X 预测（当前行 X 不能缺失）
            if not np.isnan(X[i]).any():
                fitted[i] = np.r_[1.0, X[i]] @ beta
                coefs[i] = beta
        except Exception:
            pass
    resid = y - fitted
    return fitted, resid, coefs


# ================= 3. L1：贵不贵 =================
def layer1_valuation(df):
    out = {}
    for col in ["Y30", "Y10"]:
        s = df[col].dropna()
        cur = s.iloc[-1]
        out[col] = {
            "当前": cur,
            "分位_全": (s <= cur).mean() * 100,
            "分位_10y": (s[s.index >= max(0, len(s)-2500)] <= cur).mean() * 100,
            "分位_5y":  (s[s.index >= max(0, len(s)-1250)] <= cur).mean() * 100,
            "分位_3y":  (s[s.index >= max(0, len(s)-750)]  <= cur).mean() * 100,
        }
    def grade(p):
        if pd.isna(p): return "未知"
        if p > 90: return "极便宜"
        if p > 70: return "便宜"
        if p >= 30: return "中性"
        if p >= 10: return "贵"
        return "极贵"
    out["档位_30Y"] = grade(out["Y30"]["分位_全"])
    out["档位_10Y"] = grade(out["Y10"]["分位_全"])
    return out


# ================= 4. L2：便宜是真的吗 =================
def layer2_attribution(df):
    """OLS 只用 CPI + PMI（2005年起完整样本）；FDR007 单独作为资金面状态"""
    X_cols = ["CPI同比", "制造业PMI"]
    result = {}

    for col in ["Y10", "Y30"]:
        sub = df[["日期", col] + X_cols].dropna().reset_index(drop=True)
        if len(sub) < MIN_PERIODS:
            result[col] = {"残差Z": np.nan, "信号": "样本不足"}
            continue
        fitted, resid, coefs = rolling_residual(sub[col].values, sub[X_cols].values)
        resid_s = pd.Series(resid)
        r_mean = resid_s.rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean()
        r_std  = resid_s.rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).std()
        z = (resid_s - r_mean) / r_std
        z_s = z.rolling(Z_SMOOTH).mean()
        cur_z = z_s.dropna().iloc[-1] if z_s.notna().any() else np.nan

        result[col] = {
            "残差Z": cur_z,
            "系数_CPI": coefs[-1, 1] if not np.isnan(coefs[-1, 1]) else np.nan,
            "系数_PMI": coefs[-1, 2] if not np.isnan(coefs[-1, 2]) else np.nan,
            "残差序列": resid_s,
            "残差Z序列": z_s,
            "拟合序列": pd.Series(fitted, index=sub["日期"].values if False else None),
        }
        # 拟合序列按日期对齐回 base
        result[col]["拟合_按日期"] = pd.Series(fitted, index=sub["日期"]).reindex(df["日期"]).values

    # 资金面状态：FDR007 近20日均 vs 近60日均
    fdr = df["FDR007"].dropna()
    if len(fdr) >= 60:
        fdr_trend = fdr.iloc[-20:].mean() - fdr.iloc[-60:].mean()
        fdr_cur = fdr.iloc[-1]
        fdr_ok = True
    else:
        fdr_trend, fdr_cur, fdr_ok = np.nan, np.nan, False

    # 归因矩阵：残差 × 资金面（资金面缺失时只看残差）
    z10 = result["Y10"].get("残差Z", np.nan)
    if pd.isna(z10):
        verdict, strength = "样本不足", 0.0
    else:
        resid_pos = z10 > 0.5
        resid_neg = z10 < -0.5
        if not fdr_ok:
            if resid_pos:   verdict, strength = "残差为正（便宜），资金面数据缺失，仅凭基本面判断", 0.5
            elif resid_neg: verdict, strength = "残差为负（贵），资金面数据缺失，仅凭基本面判断", -0.5
            else:           verdict, strength = "残差接近零（合理），资金面数据缺失", 0.0
        else:
            loose = fdr_trend <= 0
            tight = fdr_trend > 0
            if resid_pos and loose:
                verdict, strength = "真便宜，资金面也支持（强信号）", 1.0
            elif resid_pos and tight:
                verdict, strength = "便宜但资金面收紧（弱信号，等）", 0.5
            elif resid_neg and loose:
                verdict, strength = "偏贵，但资金面宽松（观察，不追）", -0.3
            elif resid_neg and tight:
                verdict, strength = "偏贵且资金面收紧（回避）", -1.0
            else:
                verdict, strength = "偏离不显著（中性）", 0.0

    result["FDR007"] = {"当前": fdr_cur, "趋势": fdr_trend, "可用": fdr_ok}
    result["判定"] = verdict
    result["强度"] = strength
    return result


# ================= 5. L3：买长的还是短的 =================
def layer3_structure(df):
    spread = (df["Y30"] - df["Y10"]).dropna()
    if len(spread) == 0:
        return {"久期倾向": "数据不足"}
    cur = spread.iloc[-1]
    pct = (spread <= cur).mean() * 100
    if pct > 70:   tilt = "拉久期（优先30Y）"
    elif pct < 30: tilt = "缩短久期（10Y或更短）"
    else:          tilt = "中性（10Y为主）"
    return {"当前利差": cur, "分位": pct, "久期倾向": tilt, "序列": spread}


# ================= 6. L4：能买多少 =================
def layer4_constraint(df):
    s = df["中美利差_10Y"].dropna()
    if len(s) == 0:
        return {"仓位系数": 1.0, "说明": "中美利差为空"}
    cur = s.iloc[-1]
    pct = (s <= cur).mean() * 100
    if pct < 5:    coef, note = 0.5, "极端倒挂，仓位砍半"
    elif pct < 20: coef, note = 0.8, "历史低位，仓位打八折"
    else:          coef, note = 1.0, "正常区间，不额外约束"
    return {"当前": cur, "分位": pct, "仓位系数": coef, "说明": note, "序列": s}


# ================= 7. 决策合成 =================
def decision(l1, l2, l3, l4):
    strength = l2["强度"]
    if strength <= -0.8:
        return {"综合": "回避", "方向": "偏空", "久期": "—", "仓位系数": 0.0,
                "说明": "L2判定回避，L1估值不生效"}

    grade = l1["档位_30Y"]
    cheap = grade in {"极便宜", "便宜"}
    expensive = grade in {"极贵", "贵"}

    if strength > 0 and cheap:
        final = "偏多（强）" if strength >= 0.8 else "偏多（中）"
        direction = "偏多"
    elif strength > 0 and expensive:
        final, direction = "观察（估值贵但残差偏便宜）", "中性"
    elif strength < 0 and expensive:
        final, direction = "偏空（中）", "偏空"
    elif strength < 0 and cheap:
        final, direction = "观察（估值便宜但残差偏贵）", "中性"
    else:
        final, direction = "中性", "中性"

    dur = l3["久期倾向"] if direction == "偏多" else "—"
    pos = l4["仓位系数"] if direction == "偏多" else 1.0
    return {"综合": final, "方向": direction, "久期": dur, "仓位系数": pos, "说明": ""}


# ================= 8. 绘图 =================
def plot_all(df, l1, l2, l3, l4, dec, out_png):
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), gridspec_kw={"hspace": 0.40})

    # 图1：收益率 + 拟合中枢
    ax = axes[0]
    ax.plot(df["日期"], df["Y30"], color="tab:red", lw=1.2, label="30Y 收益率")
    ax.plot(df["日期"], df["Y10"], color="tab:blue", lw=1.2, label="10Y 收益率")
    fit10 = l2["Y10"].get("拟合_按日期")
    if fit10 is not None and len(fit10) == len(df):
        ax.plot(df["日期"], fit10, color="tab:blue", lw=1.0, ls="--",
                alpha=0.7, label="10Y 模型拟合值（CPI+PMI）")
    ax.set_title("图1｜国债收益率与合理中枢（L1 估值）", fontsize=13, fontweight="bold")
    ax.set_ylabel("收益率 (%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    t1 = (
        "【计算】收益率历史分位，分位越高=价格越便宜；虚线=按CPI+PMI算出的合理值\n"
        f"【解读】30Y={_fmt(l1['Y30']['当前'])}%（全样本 {_fmt(l1['Y30']['分位_全'], '{:.0f}')}%，"
        f"10年 {_fmt(l1['Y30']['分位_10y'], '{:.0f}')}%，5年 {_fmt(l1['Y30']['分位_5y'], '{:.0f}')}%）\n"
        f"   10Y={_fmt(l1['Y10']['当前'])}%（全样本 {_fmt(l1['Y10']['分位_全'], '{:.0f}')}%）\n"
        f"   → L1 结论：30Y {l1['档位_30Y']}，10Y {l1['档位_10Y']}"
    )
    _add_box(ax, t1)

    # 图2：残差 Z + FDR007
    ax = axes[1]
    z10 = l2["Y10"].get("残差Z序列")
    if z10 is not None and len(z10):
        ax.plot(df["日期"].iloc[:len(z10)], z10.values, color="tab:blue", lw=1.2,
                label="10Y 残差 Z（剔除合理因素后的便宜/贵）")
        ax.axhline(0, color="black", lw=0.7)
        ax.axhline(1.5, color="tab:red", lw=0.9, ls="--", label="+1.5σ")
        ax.axhline(-1.5, color="tab:green", lw=0.9, ls="--", label="−1.5σ")
    ax.set_ylabel("残差 Z", color="tab:blue")
    ax2 = ax.twinx()
    if df["FDR007"].notna().any():
        ax2.plot(df["日期"], df["FDR007"], color="tab:orange", lw=0.9, alpha=0.7, label="FDR007")
        ax2.set_ylabel("FDR007 (%)", color="tab:orange")
    ax.set_title("图2｜剔除合理因素后的便宜/贵 + 资金面温度（L2 归因）",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    z10v = l2["Y10"].get("残差Z", np.nan)
    fdr_info = l2["FDR007"]
    if fdr_info["可用"]:
        fdr_txt = f"FDR007 当前={_fmt(fdr_info['当前'])}%，趋势={_fmt(fdr_info['趋势'], '{:+.3f}')}（20日均−60日均）"
    else:
        fdr_txt = "FDR007 数据不足（2021年前无值），资金面状态未知"
    t2 = (
        "【计算】10Y收益率 ~ CPI + PMI 滚动OLS，残差Z=剔除合理因素后的偏离；FDR007=资金面温度\n"
        f"【解读】10Y 残差 Z={_fmt(z10v)}（正=便宜，负=贵）；{fdr_txt}\n"
        f"   回归系数：CPI={_fmt(l2['Y10'].get('系数_CPI', np.nan))}，"
        f"PMI={_fmt(l2['Y10'].get('系数_PMI', np.nan))}\n"
        f"   → L2 结论：{l2['判定']}"
    )
    _add_box(ax, t2, loc="upper right")

    # 图3：期限利差
    ax = axes[2]
    s = l3.get("序列")
    if s is not None and len(s):
        ax.plot(df["日期"].iloc[:len(s)], s.values, color="tab:purple", lw=1.2, label="30Y−10Y 利差")
        ax.axhline(s.mean(), color="grey", lw=0.8, ls=":", label="历史均值")
    ax.set_title("图3｜期限利差：买长的还是买短的（L3）", fontsize=13, fontweight="bold")
    ax.set_ylabel("利差 (BP)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    t3 = (
        "【计算】30Y−10Y；分位>70%=曲线陡（长端补偿厚），<30%=曲线平（长端已充分定价）\n"
        f"【解读】当前利差={_fmt(l3.get('当前利差', np.nan), '{:.0f}')}BP，"
        f"历史分位={_fmt(l3.get('分位', np.nan), '{:.0f}')}%\n"
        f"   → L3 结论：{l3.get('久期倾向', '—')}"
    )
    _add_box(ax, t3)

    # 图4：中美利差
    ax = axes[3]
    s4 = l4.get("序列")
    if s4 is not None and len(s4):
        ax.plot(df["日期"].iloc[:len(s4)], s4.values, color="tab:brown", lw=1.2, label="中美10Y利差")
        ax.axhline(0, color="black", lw=0.7)
    ax.set_title("图4｜中美利差：外部环境允许买多少（L4）", fontsize=13, fontweight="bold")
    ax.set_ylabel("利差 (BP)")
    ax.set_xlabel("日期")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    t4 = (
        "【计算】中国10Y − 美国10Y；分位越低=倒挂越深=外部约束越紧\n"
        f"【解读】当前利差={_fmt(l4.get('当前', np.nan), '{:.0f}')}BP，"
        f"历史分位={_fmt(l4.get('分位', np.nan), '{:.0f}')}%\n"
        f"   → L4 结论：仓位系数 {_fmt(l4.get('仓位系数', 1.0), '{:.1f}')}（{l4.get('说明', '')}）\n"
        f"   【综合】{dec['综合']}｜久期：{dec.get('久期', '—')}｜仓位：{dec.get('仓位系数', 1.0)}"
    )
    _add_box(ax, t4)

    fig.suptitle(f"债券四层决策分析（{df['日期'].min().date()} ~ {df['日期'].max().date()}，"
                 f"{len(df)} 个交易日）", fontsize=15, fontweight="bold", y=0.995)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================= 9. 主流程 =================
if __name__ == "__main__":
    print("=" * 72)
    print("债券四层决策分析")
    print("  第1层：贵不贵？  → 第2层：便宜是真的吗？")
    print("  → 第3层：买长的还是短的？  → 第4层：能买多少？")
    print("=" * 72)

    df = load_all()
    print(f"\n数据对齐完成：{len(df)} 个交易日，"
          f"{df['日期'].min().date()} ~ {df['日期'].max().date()}")
    print(f"  收益率样本：{len(df)} 天（2005年起）")
    print(f"  CPI/PMI：月度数据前向填充至日频")
    print(f"  FDR007 有效：{df['FDR007'].notna().sum()} 天（2021年起）")

    l1 = layer1_valuation(df)
    l2 = layer2_attribution(df)
    l3 = layer3_structure(df)
    l4 = layer4_constraint(df)
    dec = decision(l1, l2, l3, l4)

    out_png = os.path.join(OUT_DIR, "bond_decision_panel.png")
    plot_all(df, l1, l2, l3, l4, dec, out_png)

    print("\n" + "═" * 62)
    print("  决策面板")
    print("═" * 62)
    print("【第1层】现在贵不贵？")
    print(f"  30Y {_fmt(l1['Y30']['当前'])}%  全样本分位 {_fmt(l1['Y30']['分位_全'], '{:.0f}')}%  "
          f"5年 {_fmt(l1['Y30']['分位_5y'], '{:.0f}')}%  → {l1['档位_30Y']}")
    print(f"  10Y {_fmt(l1['Y10']['当前'])}%  全样本分位 {_fmt(l1['Y10']['分位_全'], '{:.0f}')}%  "
          f"→ {l1['档位_10Y']}")

    print("\n【第2层】这个便宜是真的吗？")
    print(f"  10Y 残差 Z={_fmt(l2['Y10'].get('残差Z', np.nan))}（正=剔除合理因素后偏便宜）")
    print(f"  30Y 残差 Z={_fmt(l2['Y30'].get('残差Z', np.nan))}")
    if l2["FDR007"]["可用"]:
        print(f"  FDR007 当前={_fmt(l2['FDR007']['当前'])}%，"
              f"趋势={_fmt(l2['FDR007']['趋势'], '{:+.3f}')}")
    else:
        print(f"  FDR007 数据不足，资金面状态未知")
    print(f"  → {l2['判定']}")

    print("\n【第3层】买长的还是买短的？")
    print(f"  30Y−10Y 利差={_fmt(l3.get('当前利差', np.nan), '{:.0f}')}BP  "
          f"分位={_fmt(l3.get('分位', np.nan), '{:.0f')}%  → {l3.get('久期倾向', '—')}")

    print("\n【第4层】能买多少？")
    print(f"  中美利差={_fmt(l4.get('当前', np.nan), '{:.0f')}BP  "
          f"分位={_fmt(l4.get('分位', np.nan), '{:.0f')}%  → {l4.get('说明', '')}")

    print("─" * 62)
    print(f"综合：{dec['综合']}")
    print(f"方向：{dec['方向']}  久期：{dec.get('久期', '—')}  "
          f"仓位系数：{dec.get('仓位系数', 1.0)}")
    print("═" * 62)
    print(f"\n诊断图 -> {out_png}")