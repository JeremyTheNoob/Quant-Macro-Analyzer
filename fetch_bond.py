import os
import time
import akshare as ak
import pandas as pd

OUTPUT_DIR = "data_bond"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RATE_START_DATE = "20050101"
FDR007_START = "20210101"


def save_csv(df, filename, label):
    if df is None or df.empty:
        print(f"  ❌ {label}: 空数据，未保存")
        return False
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  ✅ {label}: {len(df)} 行 -> {filepath}")
    return True


def safe_to_datetime(series, fmt=None):
    if fmt:
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


# ================= 1. 国债收益率 =================
def fetch_bond_yield_sina(symbol):
    df = ak.bond_gb_zh_sina(symbol=symbol)
    if df is None or df.empty:
        raise ValueError(f"bond_gb_zh_sina 返回空: {symbol}")

    rename_map = {
        "date": "日期", "open": "开盘", "high": "最高",
        "low": "最低", "close": "收益率", "volume": "成交量",
    }
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=actual_rename)

    if "日期" not in df.columns or "收益率" not in df.columns:
        raise ValueError(f"国债收益率缺少核心列，实际列: {list(df.columns)}")

    df["日期"] = safe_to_datetime(df["日期"])
    df = df.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    df["收益率"] = pd.to_numeric(df["收益率"], errors="coerce")
    return df


# ================= 2. CPI：显式格式修正 =================
def fetch_cpi():
    df = ak.macro_china_cpi()
    if df is None or df.empty:
        raise ValueError("macro_china_cpi 返回空")

    if "月份" not in df.columns or "全国-同比增长" not in df.columns:
        raise ValueError(f"CPI 缺少核心列，实际列: {list(df.columns)}")

    out = df[["月份", "全国-同比增长"]].rename(
        columns={"月份": "日期", "全国-同比增长": "CPI同比"}
    )
    out["日期"] = safe_to_datetime(out["日期"], fmt="%Y年%m月份")
    out = out.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    out["CPI同比"] = pd.to_numeric(out["CPI同比"], errors="coerce")
    return out


# ================= 3. PMI =================
def fetch_pmi():
    df = ak.macro_china_pmi()
    if df is None or df.empty:
        raise ValueError("macro_china_pmi 返回空")

    if "月份" not in df.columns or "制造业-指数" not in df.columns:
        raise ValueError(f"PMI 缺少核心列，实际列: {list(df.columns)}")

    out = df[["月份", "制造业-指数"]].rename(
        columns={"月份": "日期", "制造业-指数": "制造业PMI"}
    )
    s = out["日期"].astype(str)
    s = s.str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False)
    s = s.str.replace("月", "-01", regex=False)
    out["日期"] = safe_to_datetime(s)
    out = out.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    out["制造业PMI"] = pd.to_numeric(out["制造业PMI"], errors="coerce")
    return out


# ================= 4. FDR007：按月循环拉取 =================
def fetch_fdr007(start_date=FDR007_START, end_date=None):
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    month_edges = pd.date_range(start=start.replace(day=1), end=end, freq="MS")
    if len(month_edges) > 0 and month_edges[-1] > end:
        month_edges = month_edges[:-1]
    month_edges = list(month_edges) + [end]

    frames = []
    for i in range(len(month_edges) - 1):
        seg_start = month_edges[i]
        seg_end = min(month_edges[i + 1] - pd.Timedelta(days=1), end)
        if seg_start > end:
            break
        try:
            df = ak.repo_rate_hist(
                start_date=seg_start.strftime("%Y%m%d"),
                end_date=seg_end.strftime("%Y%m%d"),
            )
            if df is not None and not df.empty and "date" in df.columns:
                frames.append(df)
                print(f"  FDR007 {seg_start.date()} ~ {seg_end.date()}: {len(df)} 行")
        except Exception as e:
            print(f"  FDR007 {seg_start.date()} ~ {seg_end.date()} 失败: {e}")
        time.sleep(0.5)

    if not frames:
        raise ValueError("FDR007 所有月份均获取失败")

    full = pd.concat(frames, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"], errors="coerce")
    full = full.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if "FDR007" not in full.columns:
        raise ValueError(f"FDR007 列不存在，实际列: {list(full.columns)}")

    out = full[["date", "FDR007"]].rename(columns={"date": "日期"})
    out["FDR007"] = pd.to_numeric(out["FDR007"], errors="coerce")
    out = out.dropna(subset=["日期", "FDR007"]).reset_index(drop=True)
    return out


# ================= 5. 中美国债收益率 + 利差 =================
def fetch_bond_rate_cn_us():
    df = ak.bond_zh_us_rate(start_date=RATE_START_DATE)
    if df is None or df.empty:
        raise ValueError("bond_zh_us_rate 返回空")

    required = ["日期", "中国国债收益率10年", "中国国债收益率30年", "美国国债收益率10年"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少核心列: {missing}，实际列: {list(df.columns)}")

    out = df[required].copy()
    out["日期"] = safe_to_datetime(out["日期"])
    for c in ["中国国债收益率10年", "中国国债收益率30年", "美国国债收益率10年"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    out["期限利差_30Y_10Y"] = out["中国国债收益率30年"] - out["中国国债收益率10年"]
    out["中美利差_10Y"] = out["中国国债收益率10年"] - out["美国国债收益率10年"]
    return out


# ================= 主流程 =================
if __name__ == "__main__":
    print("=" * 70)
    print("债券补充数据获取")
    print("=" * 70)

    summary = []

    tasks = [
        ("30年国债收益率", lambda: fetch_bond_yield_sina("中国30年期国债"), "30年国债收益率.csv", None),
        ("10年国债收益率", lambda: fetch_bond_yield_sina("中国10年期国债"), "10年国债收益率.csv", None),
        ("CPI", fetch_cpi, "CPI.csv", None),
        ("PMI", fetch_pmi, "PMI.csv", None),
        ("FDR007", fetch_fdr007, "FDR007.csv", "按月循环拉取，可能耗时较长"),
        ("中美国债收益率", fetch_bond_rate_cn_us, "中美国债收益率.csv", None),
    ]

    for label, fn, filename, note in tasks:
        print(f"\n[获取] {label} ...")
        if note:
            print(f"  ({note})")
        try:
            df = fn()
            if save_csv(df, filename, label):
                summary.append({
                    "数据集": label, "行数": len(df),
                    "起始": df["日期"].min().date(),
                    "结束": df["日期"].max().date(),
                    "文件": filename,
                })
        except Exception as e:
            print(f"  ❌ {label} 失败: {e}")
            summary.append({"数据集": label, "行数": 0, "错误": str(e)})
        time.sleep(1.5)

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    if summary:
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "_summary.csv"),
                          index=False, encoding="utf-8-sig")

    print(f"\n文件目录: {os.path.abspath(OUTPUT_DIR)}")