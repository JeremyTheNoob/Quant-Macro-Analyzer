import os
import time
import akshare as ak
import pandas as pd
from datetime import datetime

OUTPUT_DIR = "data_market"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# A股指数：腾讯源，start_date 设极早以覆盖完整历史
INDEX_TARGETS = [
    {"name": "沪深300",  "symbol": "sh000300"},
    {"name": "中证500",  "symbol": "sh000905"},
    {"name": "中证A500", "symbol": "sh000510"},
    {"name": "创业板指", "symbol": "sz399006"},
]

# 美股指数：新浪源，一次性返回全量历史（起点约2004年）
US_TARGETS = [
    {"name": "标普500",    "symbol": ".INX"},
    {"name": "纳斯达克100", "symbol": ".IXIC"},  # 综合指数代理，非纯 NDX
]


def fetch_index_tx_full(symbol):
    """腾讯源，指定极早起始日期以获取完整历史"""
    df = ak.stock_zh_index_daily_tx(
        symbol=symbol,
        start_date="19900101",
        end_date=datetime.today().strftime("%Y%m%d"),
    )
    if df is None or df.empty:
        raise ValueError(f"腾讯源返回空数据: {symbol}")

    rename_map = {
        "date": "日期", "open": "开盘", "high": "最高",
        "low": "最低", "close": "收盘", "volume": "成交量",
    }
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=actual_rename)

    if "日期" not in df.columns or "收盘" not in df.columns:
        raise ValueError(f"腾讯源缺少核心列，实际列: {list(df.columns)}")

    df["日期"] = pd.to_datetime(df["日期"])
    return df.sort_values("日期").reset_index(drop=True)


def fetch_us_index(symbol):
    """新浪美股指数，一次性返回全量历史，无日期参数"""
    df = ak.index_us_stock_sina(symbol=symbol)
    if df is None or df.empty:
        raise ValueError(f"美股指数返回空数据: {symbol}")

    rename_map = {
        "date": "日期", "open": "开盘", "high": "最高",
        "low": "最低", "close": "收盘", "volume": "成交量",
        "amount": "成交额",
    }
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=actual_rename)

    if "日期" not in df.columns or "收盘" not in df.columns:
        raise ValueError(f"美股指数缺少核心列，实际列: {list(df.columns)}")

    df["日期"] = pd.to_datetime(df["日期"])
    return df.sort_values("日期").reset_index(drop=True)


def save(df, name):
    df = df.copy()
    df.insert(0, "资产名称", name)
    filepath = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  ✅ {name}: {len(df)} 行 "
          f"({df['日期'].min().date()} ~ {df['日期'].max().date()})")


if __name__ == "__main__":
    print("=" * 70)
    print("股票指数数据获取（A股腾讯源 + 美股新浪源）")
    print("=" * 70)

    print("\n[A股指数] 腾讯源获取完整历史 ...")
    for t in INDEX_TARGETS:
        try:
            df = fetch_index_tx_full(t["symbol"])
            save(df, t["name"])
        except Exception as e:
            print(f"  ❌ {t['name']} 失败: {e}")
        time.sleep(1.5)

    print("\n[美股指数] 新浪源获取（全量历史，起点约2004年）...")
    for t in US_TARGETS:
        try:
            df = fetch_us_index(t["symbol"])
            save(df, t["name"])
        except Exception as e:
            print(f"  ❌ {t['name']} 失败: {e}")
        time.sleep(2)

    print(f"\n任务结束，文件位于: {os.path.abspath(OUTPUT_DIR)}")