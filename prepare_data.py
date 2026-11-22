"""따릉이 시간대별 이용정보(2025.07~2026.06, 12개월)를 대시보드용으로 집계해 parquet로 저장한다.

실행: python prepare_data.py
출력: rental_agg_YYYYMM.parquet (월별 12개), borrow.xlsx (app.py와 같은 폴더)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DOWNLOADS = Path(r"C:\Users\seung\Downloads")
BASE_DATE = pd.Timestamp("2025-07-01")  # app.py와 동일한 day-offset 기준일
OUT_GLOB = "rental_agg_*.parquet"  # 월별 분할 저장 (파일당 25MB 제한 대응)


def target_month_files() -> list[str]:
    """2025-07 ~ 2026-06 대상 파일명 생성."""
    return [
        f"{y}{m:02d}_hourly.csv"
        for y in (2025, 2026)
        for m in range(1, 13)
        if (2025, 7) <= (y, m) <= (2026, 6)
    ]


MONTH_FILES = target_month_files()
USECOLS = [
    "대여일자",
    "대여시간",
    "대여소번호",
    "대여소명",
    "연령대코드",
    "이용건수",
    "이동거리(M)",
    "이용시간(분)",
]
KEYS = ["날짜", "연월", "시간", "요일", "주말여부", "월", "계절", "연령대", "대여소번호", "대여소명"]

WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
SEASON_BY_MONTH = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
                   6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}


def clean_and_derive(chunk: pd.DataFrame) -> pd.DataFrame:
    """청크 단위로 타입 정리 + 파생변수 생성."""
    dt = pd.to_datetime(chunk["대여일자"], format="%Y-%m-%d", errors="coerce")
    chunk = chunk.loc[dt.notna()].copy()
    dt = dt.loc[dt.notna()]

    chunk["날짜"] = dt.dt.normalize()
    chunk["연월"] = dt.dt.strftime("%Y-%m")
    dow = dt.dt.day_of_week
    chunk["요일"] = dow.map(lambda i: WEEKDAYS[i])
    chunk["주말여부"] = dow >= 5
    chunk["월"] = dt.dt.month.astype("int8")
    chunk["계절"] = chunk["월"].map(SEASON_BY_MONTH).astype("category")
    age = chunk["연령대코드"].fillna("미기재").astype(str).str.strip()
    chunk["연령대"] = age.replace({"~10대": "10대", "70대이상": "70대 이상"})
    chunk["시간"] = pd.to_numeric(chunk["대여시간"], errors="coerce").astype("Int16")

    num = pd.to_numeric(chunk["대여소번호"].astype(str).str.strip().str.replace(r"[^0-9]", "", regex=True),
                        errors="coerce")
    chunk["대여소번호"] = num.astype("Int32")
    chunk["대여소명"] = chunk["대여소명"].fillna("미기재").astype(str).str.strip()

    cnt = pd.to_numeric(chunk["이용건수"], errors="coerce").fillna(0)
    dist = pd.to_numeric(chunk["이동거리(M)"], errors="coerce").clip(lower=0, upper=100_000).fillna(0)
    dur = pd.to_numeric(chunk["이용시간(분)"], errors="coerce").clip(lower=0, upper=1_440).fillna(0)

    chunk = chunk.loc[(cnt > 0) & chunk["대여소번호"].notna() & chunk["시간"].notna()].copy()
    chunk["이용건수"] = cnt.loc[chunk.index].astype("int32")
    chunk["거리합"] = (dist.loc[chunk.index] * chunk["이용건수"]).astype("float64")
    chunk["시간합"] = (dur.loc[chunk.index] * chunk["이용건수"]).astype("float64")
    return chunk


def main() -> None:
    parts: list[pd.DataFrame] = []
    total_rows = 0

    missing = [f for f in MONTH_FILES if not (DOWNLOADS / f).exists()]
    if missing:
        print(f"경고: 다음 파일이 없어 제외합니다 -> {', '.join(missing)}", flush=True)

    for fname in MONTH_FILES:
        path = DOWNLOADS / fname
        if not path.exists():
            continue
        month_parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, encoding="cp949", usecols=USECOLS, chunksize=300_000, low_memory=False):
            total_rows += len(chunk)
            clean = clean_and_derive(chunk)
            grouped = (
                clean.groupby(KEYS, as_index=False, observed=True)
                .agg(이용건수=("이용건수", "sum"), 거리합=("거리합", "sum"), 시간합=("시간합", "sum"))
            )
            month_parts.append(grouped)
        month_agg = pd.concat(month_parts, ignore_index=True)
        month_agg = (
            month_agg.groupby(KEYS, as_index=False, observed=True)
            .agg(이용건수=("이용건수", "sum"), 거리합=("거리합", "sum"), 시간합=("시간합", "sum"))
        )
        print(f"{fname}: 원본 {total_rows:,}행 누적 / 집계 {len(month_agg):,}행", flush=True)
        parts.append(month_agg)

    agg = pd.concat(parts, ignore_index=True)
    agg = (
        agg.groupby(KEYS, as_index=False, observed=True)
        .agg(이용건수=("이용건수", "sum"), 거리합=("거리합", "sum"), 시간합=("시간합", "sum"))
    )
    # 메모리/파일 크기를 줄이기 위한 dtype 최적화 (Community Cloud 배포 고려)
    min_date, max_date = agg["날짜"].min(), agg["날짜"].max()
    # datetime(date32) 대신 int16 day-offset으로 저장해 메모리를 약 4배 줄인다.
    agg["날짜"] = (agg["날짜"] - BASE_DATE).dt.days.astype("int16")
    agg["시간"] = agg["시간"].astype("int8")
    agg["대여소번호"] = agg["대여소번호"].astype("int32")
    agg["이용건수"] = agg["이용건수"].astype("int32")
    agg["거리합"] = agg["거리합"].round().astype("int32")   # m
    agg["시간합"] = agg["시간합"].round().astype("int32")   # 분
    for col in ("연월", "요일", "연령대", "계절", "대여소명"):
        agg[col] = agg[col].astype("category")

    agg = agg.sort_values(["날짜", "시간", "대여소번호"]).reset_index(drop=True)
    months = sorted(agg["연월"].dropna().unique().tolist())

    # 월별 파일을 먼저 임시 파일로 모두 만든 뒤 한 번에 교체한다.
    # 중간에 실패해도 기존 파일이 남아 앱이 깨지지 않도록 하기 위함이다.
    outputs: list[tuple[Path, Path]] = []
    for ym in months:
        part = agg[agg["연월"] == ym]
        out = BASE / f"rental_agg_{ym.replace('-', '')}.parquet"
        tmp_path = out.with_suffix(".tmp.parquet")
        part.to_parquet(tmp_path, index=False, compression="zstd")
        outputs.append((tmp_path, out))
        print(f"  준비: {out.name} ({tmp_path.stat().st_size / 1024**2:.1f} MB, {len(part):,}행)", flush=True)

    for tmp_path, out in outputs:
        tmp_path.replace(out)
        print(f"  저장: {out.name}", flush=True)

    # 이전 연도별 파일은 새 월별 파일이 모두 준비된 뒤에만 제거한다.
    for old in BASE.glob("rental_agg_*.parquet"):
        suffix = old.stem.removeprefix("rental_agg_")
        if len(suffix) == 4 and suffix.isdigit():
            old.unlink()
            print(f"  기존 연도 파일 제거: {old.name}", flush=True)

    src = DOWNLOADS / "borrow.xlsx"
    shutil.copy2(src, BASE / "borrow.xlsx")

    print(f"\n완료: 원본 {total_rows:,}행 -> 집계 {len(agg):,}행")
    print(f"기간: {min_date.date()} ~ {max_date.date()}")
    print(f"저장: {OUT_GLOB} (월별 {len(outputs)}개, app.py와 같은 폴더)")


if __name__ == "__main__":
    main()
