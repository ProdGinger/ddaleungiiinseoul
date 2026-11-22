"""서울 따릉이 이용 패턴 대시보드.

실행: streamlit run app.py
"""

from __future__ import annotations

import gc
import io
import json
from math import cos, radians
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="따릉이 이용 패턴", page_icon="🚲", layout="wide")

BASE = Path(__file__).resolve().parent
AGG_PATHS = sorted(BASE.glob("rental_agg_*.parquet"))  # 연도별/월별 모두 자동 인식
STATION_PATH = BASE / "borrow.xlsx"
GEO_PATH = BASE / "seoul_districts.geojson"

BASE_DATE = pd.Timestamp("2025-07-01")  # 날짜를 day-offset(int16)으로 저장하는 기준일
DATA_MIN_DAY = 0   # 2025-07-01
DATA_MAX_DAY = 364  # 2026-06-30
DATA_MONTHS = tuple(
    f"{y}-{m:02d}"
    for y, m in ((2025, 7), (2025, 8), (2025, 9), (2025, 10), (2025, 11), (2025, 12),
                 (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5), (2026, 6))
)
WEEK_ORDER = ["월", "화", "수", "목", "금", "토", "일"]
AGE_ORDER = ["10대", "20대", "30대", "40대", "50대", "60대", "70대 이상", "기타"]
SEASON_ORDER = ["봄", "여름", "가을", "겨울"]


def day_to_date(day: int):
    """day-offset을 실제 날짜로 변환."""
    return (BASE_DATE + pd.Timedelta(days=int(day))).date()


def days_to_dates(days: pd.Series) -> pd.Series:
    """day-offset Series를 datetime Series로 변환(날씨 데이터 조인용)."""
    return BASE_DATE + pd.to_timedelta(days.astype("int64"), unit="D")

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
    font=dict(size=12),
)

MAP_CONFIG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False}


def clear_district() -> None:
    """상세 보기를 닫고, 현재 지도 선택 신호를 한 번 무시하도록 표시."""
    st.session_state["selected_gu"] = None
    st.session_state["_skip_map_sel"] = st.session_state.get("_last_map_sel", "")


@st.cache_resource(show_spinner="대여소 정보를 불러오는 중...")
def load_stations() -> pd.DataFrame:
    raw = pd.read_excel(STATION_PATH, header=None, skiprows=4)
    cols = ["대여소번호", "대여소명", "자치구", "상세주소", "위도", "경도",
            "설치시기", "LCD거치대수", "QR거치대수", "운영방식"]
    raw = raw.iloc[:, : len(cols)].copy()
    raw.columns = cols
    for c in ("대여소번호", "위도", "경도", "LCD거치대수", "QR거치대수"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["대여소번호", "위도", "경도"])
    raw["대여소번호"] = raw["대여소번호"].astype("int64")
    # LCD형과 QR형은 서로 다른 대여소에 설치돼 있으므로 값이 있는 쪽을 사용
    raw["거치대수"] = raw["QR거치대수"].fillna(raw["LCD거치대수"])
    raw = raw.drop_duplicates("대여소번호", keep="first")
    return raw[["대여소번호", "대여소명", "자치구", "위도", "경도", "거치대수"]]


@st.cache_resource(show_spinner="서울 자치구 경계를 불러오는 중...")
def load_seoul_geojson() -> dict:
    """서울 25개 자치구 경계 GeoJSON (WGS84)."""
    with GEO_PATH.open(encoding="utf-8") as fp:
        return json.load(fp)


@st.cache_data(show_spinner="자치구 면적을 계산하는 중...")
def district_areas(geojson: dict) -> pd.DataFrame:
    """GeoJSON 외곽 링에서 구 면적(km²)을 근사 계산."""
    rows = []
    for feat in geojson["features"]:
        geom = feat["geometry"]
        polygons = ([geom["coordinates"]] if geom["type"] == "Polygon"
                    else geom["coordinates"] if geom["type"] == "MultiPolygon" else [])
        area = 0.0
        for polygon in polygons:
            ring = polygon[0] if polygon else []
            if len(ring) < 3:
                continue
            lat0 = sum(p[1] for p in ring) / len(ring)
            xs = [p[0] * 111.32 * cos(radians(lat0)) for p in ring]
            ys = [p[1] * 110.57 for p in ring]
            n = len(ring)
            area += 0.5 * abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                                  for i in range(n)))
        rows.append({"자치구": feat["properties"]["name"], "면적km2": area})
    return pd.DataFrame(rows)


# 집계 결과는 읽기 전용으로 사용한다(변경이 필요하면 호출부에서 copy()).
@st.cache_resource(show_spinner="필터 조건을 집계하는 중...")
def compute_aggregates(
    start_day: int | None,
    end_day: int | None,
    months: tuple[str, ...] | None,
    daypart: str,
    ages: tuple[str, ...],
    hour_start: int,
    hour_end: int,
    seasons: tuple[str, ...],
) -> dict | None:
    """필터 조건에 맞는 작은 집계 결과만 캐시한다.

    parquet를 파일별로 하나씩 읽어 집계한 뒤 즉시 해제하므로,
    전체 2,470만 행을 동시에 메모리에 올리지 않는다.
    구역 선택/전체 보기처럼 필터가 그대로인 인터랙션은 이 캐시를 재사용한다.
    """
    pieces: list[dict] = []

    for path in AGG_PATHS:
        part_df = pd.read_parquet(path)
        # 이전 버전 parquet의 date32 날짜도 day-offset으로 변환
        if not pd.api.types.is_integer_dtype(part_df["날짜"].dtype):
            part_df["날짜"] = (
                pd.to_datetime(part_df["날짜"]) - BASE_DATE
            ).dt.days.astype("int16")

        if months:
            mask = part_df["연월"].isin(months)
        else:
            mask = part_df["날짜"].between(start_day, end_day)
        mask &= part_df["시간"].between(hour_start, hour_end)
        mask &= part_df["연령대"].isin(ages)
        mask &= part_df["계절"].isin(seasons)
        if daypart == "평일":
            mask &= ~part_df["주말여부"].astype(bool)
        elif daypart == "주말":
            mask &= part_df["주말여부"].astype(bool)

        f = part_df[mask]
        if not f.empty:
            pieces.append({
                "cnt": int(f["이용건수"].sum()),
                "dist": float(f["거리합"].sum()),
                "dur": float(f["시간합"].sum()),
                "days": f["날짜"].drop_duplicates().to_numpy(),
                "stations": f["대여소번호"].drop_duplicates().to_numpy(),
                "hourly": f.groupby("시간", observed=True)["이용건수"].sum(),
                "by_age": (
                    f.groupby("연령대", observed=True, as_index=False)["이용건수"].sum()
                ),
                "heat": (
                    f.groupby(["요일", "시간"], observed=True, as_index=False)["이용건수"].sum()
                ),
                "by_station": (
                    f.groupby(["대여소번호", "대여소명"], observed=True, as_index=False)
                    .agg(이용건수=("이용건수", "sum"), 거리합=("거리합", "sum"),
                         시간합=("시간합", "sum"))
                ),
                "by_station_hour": (
                    f.groupby(["대여소번호", "시간"], observed=True, as_index=False)["이용건수"].sum()
                ),
                "by_season": (
                    f.groupby("계절", observed=True, as_index=False)["이용건수"].sum()
                ),
                "by_month": (
                    f.groupby("연월", observed=True, as_index=False)["이용건수"].sum()
                ),
                "daily": (
                    f.groupby("날짜", observed=True, as_index=False)
                    .agg(이용건수=("이용건수", "sum"))
                ),
            })
        del f, mask, part_df
        gc.collect()

    if not pieces:
        return None
    total = sum(p["cnt"] for p in pieces)
    if total == 0:
        return None

    all_days: set[int] = set()
    all_stations: set[int] = set()
    for p in pieces:
        all_days.update(int(v) for v in p["days"])
        all_stations.update(int(v) for v in p["stations"])

    def _sum_counts(name: str, keys: str | list[str]) -> pd.DataFrame:
        merged = pd.concat([p[name] for p in pieces], ignore_index=True)
        return merged.groupby(keys, observed=True, as_index=False)["이용건수"].sum()

    by_station = pd.concat([p["by_station"] for p in pieces], ignore_index=True)
    by_station = (
        by_station.groupby(["대여소번호", "대여소명"], observed=True, as_index=False)
        .agg(이용건수=("이용건수", "sum"), 거리합=("거리합", "sum"), 시간합=("시간합", "sum"))
    )
    hourly = pd.concat([p["hourly"] for p in pieces]).groupby(level=0).sum()

    return {
        "total": total,
        "n_days": len(all_days),
        "avg_dist": sum(p["dist"] for p in pieces) / total / 1000,
        "avg_dur": sum(p["dur"] for p in pieces) / total,
        "station_count": len(all_stations),
        "hourly": (
            hourly.reindex(range(24), fill_value=0)
            .rename_axis("시간").reset_index(name="이용건수")
        ),
        "by_age": _sum_counts("by_age", "연령대"),
        "heat": _sum_counts("heat", ["요일", "시간"]),
        "by_station": by_station,
        "by_station_hour": _sum_counts("by_station_hour", ["대여소번호", "시간"]),
        "by_season": _sum_counts("by_season", "계절"),
        "by_month": _sum_counts("by_month", "연월").sort_values("연월"),
        "daily": _sum_counts("daily", "날짜"),
    }


def district_outline(sub_geojson: dict) -> tuple[list, list]:
    """선택한 구 GeoJSON의 외곽 링을 Scattergeo 선용 lon/lat 목록으로 변환."""
    lons: list = []
    lats: list = []
    for feat in sub_geojson["features"]:
        geom = feat["geometry"]
        polygons = ([geom["coordinates"]] if geom["type"] == "Polygon"
                    else geom["coordinates"] if geom["type"] == "MultiPolygon" else [])
        for polygon in polygons:
            ring = polygon[0] if polygon else []
            for lon, lat in ring:
                lons.append(lon)
                lats.append(lat)
            lons.append(None)  # 링 사이를 끊어 별도 선으로 그리기
            lats.append(None)
    return lons, lats


def read_uploaded_csv(file) -> pd.DataFrame | None:
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="replace")


def pick_col(columns, candidates) -> str | None:
    for cand in candidates:
        for col in columns:
            if str(col).strip().lower() == cand.lower():
                return col
    return None


def main() -> None:
    st.title("🚲 따릉이 이용 패턴 대시보드")

    if not AGG_PATHS:
        st.error("집계 데이터가 없습니다. 먼저 `python prepare_data.py`를 실행해주세요.")
        st.stop()

    stations = load_stations()
    # 필터 옵션은 데이터 전체를 읽지 않고 고정 메타데이터로 제공한다.
    min_day, max_day = DATA_MIN_DAY, DATA_MAX_DAY
    st.caption(
        f"{day_to_date(min_day):%Y.%m} ~ {day_to_date(max_day):%Y.%m} · "
        "시간대 / 요일 / 연령대 / 대여소별 이용 행태와 계절 변화"
    )

    # ---------- 사이드바 필터 ----------
    st.sidebar.header("🔎 필터")
    min_d, max_d = day_to_date(min_day), day_to_date(max_day)
    period_mode = st.sidebar.radio("기간 선택 방식", ["달력 선택", "월별 선택"], horizontal=True)

    if period_mode == "달력 선택":
        date_range = st.sidebar.date_input("기간 선택", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        if len(date_range) != 2:
            st.sidebar.warning("시작일과 종료일을 모두 선택해주세요.")
            st.stop()
        start_day = (pd.Timestamp(date_range[0]) - BASE_DATE).days
        end_day = (pd.Timestamp(date_range[1]) - BASE_DATE).days
        period_args = (start_day, end_day, None)
    else:
        month_options = list(DATA_MONTHS)
        months = st.sidebar.pills(
            "월별 선택", month_options, selection_mode="multi", default=month_options,
            format_func=lambda m: m.replace("-", ".")[2:], wrap=True,
            help="원하는 달만 눌러 선택하세요. 달력 기간과 동시에 적용되지 않습니다.",
        )
        if not months:
            st.sidebar.warning("한 개 이상의 월을 선택해주세요.")
            st.stop()
        period_args = (None, None, tuple(months))

    daypart = st.sidebar.radio("요일", ["전체", "평일", "주말"], horizontal=True)

    age_options = list(AGE_ORDER)
    ages = st.sidebar.pills("연령대", age_options, selection_mode="multi", default=age_options, wrap=True)

    hour_start, hour_end = st.sidebar.slider("시간대 범위", 0, 23, (0, 23))
    season_options = list(SEASON_ORDER)
    seasons = st.sidebar.pills("계절", season_options, selection_mode="multi", default=season_options, wrap=True)
    top_n = st.sidebar.slider("대여소 순위 표시 수", 5, 30, 10)
    st.sidebar.divider()
    st.sidebar.caption("필터를 바꾸면 KPI와 모든 차트가 함께 갱신됩니다.")

    agg = compute_aggregates(
        *period_args, daypart, tuple(ages), hour_start, hour_end, tuple(seasons)
    )
    if agg is None:
        st.warning("선택한 조건에 해당하는 이용 records가 없습니다. 필터를 넓혀주세요.")
        st.stop()

    # ---------- KPI ----------
    total = agg["total"]
    n_days = agg["n_days"]
    avg_dist = agg["avg_dist"]
    avg_dur = agg["avg_dur"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("총 이용건수", f"{total:,}", help="필터 조건 내 이용건수 합계")
    k2.metric("일평균 이용건수", f"{total / n_days:,.0f}", help=f"선택 기간의 데이터 존재일 {n_days}일 기준")
    k3.metric("평균 이동거리", f"{avg_dist:.2f} km", help="이용건수 가중 평균")
    k4.metric("평균 이용시간", f"{avg_dur:.1f} 분", help="이용건수 가중 평균")
    k5.metric("이용 대여소 수", f"{agg['station_count']:,}")
    st.divider()

    # ---------- 1. 시간대별 이용 건수 ----------
    left, right = st.columns([2, 1])
    with left:
        st.subheader("1. 시간대별 이용 건수")
        hourly = agg["hourly"]
        fig = px.bar(hourly, x="시간", y="이용건수", color="이용건수",
                     color_continuous_scale="Blues", labels={"이용건수": "이용 건수"})
        fig.update_traces(hovertemplate="%{x}시<br>이용 %{y:,}건<extra></extra>")
        fig.update_layout(**PLOT_LAYOUT, xaxis=dict(dtick=1), coloraxis_showscale=False, height=330)
        st.plotly_chart(fig, width="stretch")

    # ---------- 3. 연령대별 이용 비중 ----------
    with right:
        st.subheader("2. 연령대별 이용 비중")
        by_age = agg["by_age"].copy()
        by_age["_순서"] = by_age["연령대"].map({a: i for i, a in enumerate(AGE_ORDER)}).fillna(99)
        by_age = by_age.sort_values("_순서")
        fig = px.pie(by_age, values="이용건수", names="연령대", hole=0.42,
                     category_orders={"연령대": by_age["연령대"].tolist()},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textinfo="percent+label", hovertemplate="%{label}<br>%{value:,}건 (%{percent})<extra></extra>")
        fig.update_layout(**PLOT_LAYOUT, height=330, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    # ---------- 2. 요일 x 시간대 히트맵 ----------
    st.subheader("3. 요일 × 시간대 이용 패턴")
    piv = agg["heat"].pivot(index="요일", columns="시간", values="이용건수")
    piv = piv.reindex(index=WEEK_ORDER).reindex(columns=range(hour_start, hour_end + 1))
    piv = piv.dropna(how="all").dropna(axis=1, how="all")
    if not piv.empty:
        # Plotly 기본 축 방향(첫 항목이 아래)을 보정해 월→일 순으로 위에서 아래로 표시
        fig = go.Figure(go.Heatmap(
            z=piv.values[::-1], x=[f"{h}시" for h in piv.columns], y=piv.index.tolist()[::-1],
            colorscale="Blues", hovertemplate="%{y} %{x}<br>이용 %{z:,}건<extra></extra>",
            colorbar=dict(title="이용건수", thickness=12),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=300)
        st.plotly_chart(fig, width="stretch")

    # 호버 줌은 전체 지도에만 적용한다. gu-hover 클래스는 아래 스크립트가 붙이며,
    # pointer-events는 Plotly 기본값(none)을 유지해 기본 툴팁(구·이용건수·대여소 수·평균 이용시간)이 살아있다.
    st.markdown(
        """
        <style>
        .js-plotly-plot .choroplethlocation {
            transition: transform 150ms ease, filter 150ms ease;
            transform-box: fill-box;
            transform-origin: center;
        }
        .js-plotly-plot .choroplethlocation.gu-hover {
            transform: scale(1.04);
            filter: brightness(1.08) drop-shadow(0 0 3px rgba(31, 78, 121, 0.45));
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # 구역 클릭을 직접 감지해 Streamlit의 plotly_selected로 전달하고,
    # 전체 지도의 plotly_hover/unhover로 호버 줌 클래스를 토글한다(상세 지도는 제외).
    st.iframe(
        """
        <script>
        (function () {
          var w = window.parent;
          if (!w || w.__guMapScriptInstalled) return;
          w.__guMapScriptInstalled = true;
          var doc = w.document;

          function isMain(plot) {
            var locs = plot && plot.data && plot.data[0] && plot.data[0].locations;
            return !!(locs && locs.length > 1);
          }

          // pointer-events가 none이어도 클릭 좌표가 속한 구 polygon을 찾는다.
          doc.addEventListener('click', function (e) {
            var plot = e.target && e.target.closest ? e.target.closest('.js-plotly-plot') : null;
            if (!isMain(plot)) return;
            var svg = plot.querySelector('.main-svg');
            if (!svg || !svg.createSVGPoint) return;
            var sp = svg.createSVGPoint();
            sp.x = e.clientX;
            sp.y = e.clientY;
            var paths = plot.querySelectorAll('.choroplethlocation');
            var locs = plot.data[0].locations;
            for (var i = 0; i < paths.length; i++) {
              var m = paths[i].getScreenCTM();
              if (!m) continue;
              var p = sp.matrixTransform(m.inverse());
              if (paths[i].isPointInFill(p)) {
                var gu = locs[i];
                if (gu) plot.emit('plotly_selected', {
                  points: [{curveNumber: 0, pointNumber: i, pointIndex: i,
                            location: gu, customdata: gu}],
                  selections: []
                });
                break;
              }
            }
          }, true);

          function attachHover() {
            var plots = doc.querySelectorAll('.js-plotly-plot');
            for (var j = 0; j < plots.length; j++) {
              var plot = plots[j];
              if (!isMain(plot) || plot.dataset.guHoverAttached) continue;
              plot.dataset.guHoverAttached = '1';
              (function (p) {
                p.on('plotly_hover', function (ev) {
                  var pt = ev && ev.points && ev.points[0];
                  if (!pt) return;
                  var idx = pt.pointIndex != null ? pt.pointIndex : pt.pointNumber;
                  var paths = p.querySelectorAll('.choroplethlocation');
                  if (paths[idx]) paths[idx].classList.add('gu-hover');
                });
                p.on('plotly_unhover', function () {
                  var paths = p.querySelectorAll('.choroplethlocation.gu-hover');
                  for (var k = 0; k < paths.length; k++) paths[k].classList.remove('gu-hover');
                });
              })(plot);
            }
          }
          attachHover();
          new w.MutationObserver(attachHover).observe(doc.body, {childList: true, subtree: true});
        })();
        </script>
        """,
        height=1,
    )

    # ---------- 4. 자치구 지도 + 인기 대여소 ----------
    st.subheader("4. 자치구별 이용 현황 + 인기 대여소 Top {}".format(top_n))
    top = agg["by_station"].sort_values("이용건수", ascending=False).head(top_n).copy()
    top["평균이용시간"] = top["시간합"] / top["이용건수"]
    top["대여소명"] = top["대여소명"].astype(str)
    top["라벨"] = top["대여소명"] + " (" + top["대여소번호"].astype(str) + ")"

    # 대여소별 집계에 마스터(자치구·좌표·거치대수)를 붙인 작은 테이블 재사용.
    # 대여소명이 연도별로 달라질 수 있어 번호 기준으로 먼저 합친다.
    station_rides = (
        agg["by_station"].groupby("대여소번호", observed=True, as_index=False)
        .agg(대여소명=("대여소명", "first"), 이용건수=("이용건수", "sum"),
             거리합=("거리합", "sum"), 시간합=("시간합", "sum"))
    )
    station_stats = station_rides.merge(
        stations[["대여소번호", "자치구", "위도", "경도", "거치대수"]],
        on="대여소번호", how="left",
    )

    map_col, rank_col = st.columns([3, 2])
    with map_col:
        geojson = load_seoul_geojson()
        names = [feat["properties"]["name"] for feat in geojson["features"]]
        name_set = set(names)
        selected = st.session_state.get("selected_gu")
        select_names = sorted(names)  # 가나다(한글 유니코드) 순

        def build_district_detail(district_name: str):
            """캐시된 집계로 상세 지도/표에 필요한 테이블을 한 번만 만든다."""
            gu_stations = stations[stations["자치구"] == district_name].copy()
            district_detail = station_stats[station_stats["자치구"] == district_name].copy()
            district_detail["평균이용시간"] = (
                district_detail["시간합"] / district_detail["이용건수"]
            )
            district_detail["대여소명"] = district_detail["대여소명"].astype(str)

            # 시간대별 평균 즉시 탑승 가능 자전거 대수(추정).
            # 실시간 재고가 없어 "거치대수 - 해당 시간대 일평균 대여건수"로 근사하고,
            # 0 미만은 0으로 제한한 뒤 시간대 평균을 대여소별로 구한다.
            station_ids = set(district_detail["대여소번호"])
            avail_hourly = agg["by_station_hour"]
            avail_hourly = avail_hourly[
                avail_hourly["대여소번호"].isin(station_ids)
            ].copy()
            avail_hourly["시간평균대여"] = (
                avail_hourly["이용건수"] / max(agg["n_days"], 1)
            )
            rack_by_station = district_detail.set_index("대여소번호")["거치대수"]
            avail_hourly["즉시탑승가능"] = (
                avail_hourly["대여소번호"].map(rack_by_station)
                - avail_hourly["시간평균대여"]
            ).clip(lower=0)
            avail_mean = (
                avail_hourly.groupby("대여소번호")["즉시탑승가능"]
                .mean().rename("즉시탑승가능")
            )
            district_detail = district_detail.merge(
                avail_mean, on="대여소번호", how="left"
            )
            return gu_stations, district_detail

        def apply_direct_selection(value: str) -> None:
            """지도 아래 드롭다운 선택을 세션 상태에 반영."""
            current = st.session_state.get("selected_gu")
            if value != "-" and value != current:
                st.session_state["selected_gu"] = value
                st.session_state["_skip_map_sel"] = st.session_state.get("_last_map_sel", "")
                st.rerun()
            if value == "-" and current:
                clear_district()
                st.rerun()

        detail = None
        gu_stations = None
        if selected and selected in name_set:
            # 선택한 구를 메인 지도 자리에 그대로 덮어쓴다.
            # 상세 데이터는 여기서 한 번만 만들고, 아래 상세 정보 섹션에서 재사용한다.
            gu_stations, detail = build_district_detail(selected)

            sub_geojson = {
                "type": "FeatureCollection",
                "features": [feat for feat in geojson["features"]
                             if feat["properties"]["name"] == selected],
            }
            fig = go.Figure(go.Choropleth(
                geojson=sub_geojson, locations=[selected],
                featureidkey="properties.name",
                z=[0], zmin=0, zmax=1,
                colorscale=[[0, "#dce9f8"], [1, "#dce9f8"]],
                marker_line_color="#1f4e79", marker_line_width=2,
                showscale=False, hoverinfo="skip",
            ))
            if not detail.empty:
                max_cnt_d = detail["이용건수"].max()
                sizes = (6 + 16 * (detail["이용건수"] / max_cnt_d).pow(0.5)) if max_cnt_d else 8
                rack_max = detail["거치대수"].max()
                rack_max = float(rack_max) if pd.notna(rack_max) else 1
                detail["표시"] = (
                    detail["대여소명"] + "<br>이용 " + detail["이용건수"].map(lambda v: f"{int(v):,}") + "건"
                    + "<br>평균 " + detail["평균이용시간"].map(lambda v: f"{v:.1f}") + "분"
                    + "<br>거치대 " + detail["거치대수"].map(
                        lambda v: "-" if pd.isna(v) else f"{int(v):,}대")
                )
                fig.add_trace(go.Scattergeo(
                    lon=detail["경도"], lat=detail["위도"], mode="markers",
                    marker=dict(
                        size=sizes, color=detail["거치대수"], colorscale="Viridis",
                        cmin=0, cmax=rack_max, opacity=0.9,
                        line=dict(width=0.5, color="white"),
                        colorbar=dict(title="거치대수", thickness=10, len=0.7),
                    ),
                    text=detail["표시"], hovertemplate="%{text}<extra></extra>",
                ))
            # geo 레이어는 choropleth가 scatter보다 항상 아래에 오므로,
            # 점들 위에 겹쳐 보이도록 같은 구 경계를 Scattergeo 선으로 한 번 더 그린다.
            ol_lon, ol_lat = district_outline(sub_geojson)
            fig.add_trace(go.Scattergeo(
                lon=ol_lon, lat=ol_lat, mode="lines", hoverinfo="skip",
                line=dict(color="#1f4e79", width=2.5),
            ))
            fig.update_geos(fitbounds="locations", visible=False)
            # 구 이름은 지도 위 전용 라벨 공간(상단 마진)에 고정 배치한다.
            # geo 좌표로 찍으면 구 크기/fitbounds에 따라 잘리거나 점과 겹칠 수 있다.
            fig.update_layout(**{**PLOT_LAYOUT, "height": 460,
                                 "margin": dict(l=0, r=0, t=50, b=0), "dragmode": False},
                              annotations=[dict(
                                  x=0.5, y=1.05, xref="paper", yref="paper",
                                  text=f"<b>{selected}</b>", showarrow=False,
                                  xanchor="center", yanchor="middle",
                                  font=dict(size=22, color="#1f4e79",
                                            family="Arial Black, sans-serif"),
                              )])
            st.plotly_chart(fig, width="stretch", config=MAP_CONFIG)
            st.caption(f"🔍 {selected} 포커스 · 점 크기=이용건수, 색=거치대수 (구 경계가 점들 뒤에 표시됨)")
            # 두 컨트롤을 지도 바로 아래에 나란히 배치: 선택 드롭다운 → 전체 보기
            direct = st.selectbox(
                "구 직접 선택", ["-"] + select_names,
                index=(select_names.index(selected) + 1) if selected in name_set else 0,
            )
            apply_direct_selection(direct)
            st.button("전체 보기", on_click=clear_district, width="stretch")
        else:
            district = (
                station_stats.groupby("자치구", as_index=False)
                .agg(이용건수=("이용건수", "sum"), 거리합=("거리합", "sum"),
                     시간합=("시간합", "sum"), 대여소수=("대여소번호", "nunique"))
            )
            gdf = pd.DataFrame({"자치구": names}).merge(district, on="자치구", how="left")
            gdf["이용건수"] = gdf["이용건수"].fillna(0)
            gdf["대여소수"] = gdf["대여소수"].fillna(0).astype(int)
            gdf["평균이용시간"] = (gdf["시간합"] / gdf["이용건수"].replace(0, pd.NA)).fillna(0)
            max_cnt = gdf["이용건수"].max() if len(gdf) else 0

            gdf["정보"] = (
                "이용 " + gdf["이용건수"].map(lambda v: f"{int(v):,}") + "건<br>"
                + "대여소 " + gdf["대여소수"].map(str) + "개<br>"
                + "평균 " + gdf["평균이용시간"].map(lambda v: f"{v:.1f}") + "분<br>"
                + "<b>클릭하면 상세 보기</b>"
            )
            fig = go.Figure(go.Choropleth(
                geojson=geojson,
                locations=gdf["자치구"],
                featureidkey="properties.name",
                z=gdf["이용건수"],
                zmin=0,
                zmax=max_cnt if max_cnt else 1,
                colorscale="Blues",
                marker_line_color="white",
                marker_line_width=1,
                text=gdf["정보"],
                customdata=gdf["자치구"],
                hovertemplate="<b>%{location}</b><br>%{text}<extra></extra>",
                colorbar=dict(title="이용건수", thickness=10, len=0.7),
            ))
            # fitbounds가 서울 전체를 영역에 맞추고, 확대/이동 컨트롤은 모두 끈다
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(**{**PLOT_LAYOUT, "height": 460, "margin": dict(l=0, r=0, t=10, b=0), "dragmode": False})
            event = st.plotly_chart(
                fig, width="stretch", config=MAP_CONFIG,
                on_select="rerun", selection_mode="points", key="seoul_map",
            )
            points = event.selection.points
            sel_sig = json.dumps(
                [{k: str(v) for k, v in pt.items()} for pt in points],
                ensure_ascii=False, sort_keys=True,
            )
            st.session_state["_last_map_sel"] = sel_sig
            if points and sel_sig != st.session_state.get("_skip_map_sel", ""):
                pt = points[-1]
                gu = pt.get("customdata")
                if isinstance(gu, (list, tuple)):
                    gu = gu[0] if gu else None
                if gu not in name_set:
                    idx = pt.get("point_index")
                    if idx is not None and 0 <= int(idx) < len(gdf):
                        gu = gdf.iloc[int(idx)]["자치구"]
                if gu in name_set and st.session_state.get("selected_gu") != gu:
                    st.session_state["selected_gu"] = gu
                    st.rerun()
            st.session_state["_skip_map_sel"] = ""
            matched = int((gdf["이용건수"] > 0).sum())
            st.caption(
                f"경계 {len(gdf)}개 구 중 {matched}개 구에 이용 데이터가 연결됐습니다. "
                "색이 진할수록 이용건수가 많고, **구역을 클릭하면** 해당 구의 상세 지도가 열립니다."
            )
            direct = st.selectbox(
                "구 직접 선택", ["-"] + select_names,
                index=0,
            )
            apply_direct_selection(direct)

    with rank_col:
        fig = px.bar(top.sort_values("이용건수"), x="이용건수", y="라벨", orientation="h",
                     color="평균이용시간", color_continuous_scale="Teal",
                     labels={"이용건수": "이용 건수", "라벨": "", "평균이용시간": "평균 이용시간(분)"})
        fig.update_traces(hovertemplate="%{y}<br>이용 %{x:,}건<extra></extra>")
        fig.update_layout(**PLOT_LAYOUT, height=420, coloraxis_colorbar=dict(title="평균<br>이용시간", thickness=10))
        st.plotly_chart(fig, width="stretch")

    # ---------- 4-1. 구역 클릭 시 상세 정보 (지도는 위 map_col에 표시) ----------
    selected = st.session_state.get("selected_gu")
    if selected and detail is not None:
        st.divider()
        st.subheader(f"🔍 {selected} 상세 정보")
        areas = district_areas(geojson)
        area_rows = areas.loc[areas["자치구"] == selected, "면적km2"]
        area = float(area_rows.iloc[0]) if len(area_rows) else 0.0
        d_total = int(detail["이용건수"].sum()) if len(detail) else 0
        days = agg["n_days"]
        rack_mean = gu_stations["거치대수"].mean()
        density = len(gu_stations) / area if area else 0.0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("이용건수", f"{d_total:,}건")
        k2.metric("일평균 이용건수", f"{int(d_total / days):,}건" if days else "-")
        k3.metric("이용 대여소", f"{detail['대여소번호'].nunique():,} / {len(gu_stations):,}개")
        k4.metric("평균 거치대수", f"{rack_mean:.1f}대" if pd.notna(rack_mean) else "-")
        k5.metric("대여소 밀도", f"{density:.1f}개/km²")

        view = detail.sort_values("이용건수", ascending=False).head(5).copy()
        view["이용건수"] = view["이용건수"].map(lambda v: f"{int(v):,}건")
        view["거치대수"] = view["거치대수"].map(lambda v: "-" if pd.isna(v) else f"{int(v):,}대")
        view["즉시탑승가능"] = view["즉시탑승가능"].map(
            lambda v: "-" if pd.isna(v) else f"{v:.1f}대")
        view["평균이용시간"] = view["평균이용시간"].map(lambda v: f"{v:.1f}분")
        st.caption("주요 대여소 (이용건수 상위 5곳) · 위 지도의 점 크기는 이용건수, 색은 거치대수")
        st.dataframe(
            view[["대여소명", "이용건수", "거치대수", "즉시탑승가능", "평균이용시간"]].rename(
                columns={"대여소명": "대여소", "즉시탑승가능": "즉시탑승 가능(추정)",
                         "평균이용시간": "평균 이용시간"}),
            width="stretch", hide_index=True,
        )
        st.caption(
            f"면적 약 {area:.1f}km² 기준 · 원본 이용 데이터에는 실시간 자전거 수가 없어 "
            "대여소 마스터의 거치대수(자전거 수용 대수)와, "
            "`거치대수 − 해당 시간대 일평균 대여건수`(0 미만 0)의 시간대 평균으로 "
            "즉시 탑승 가능 대수를 추정했습니다."
        )

    # ---------- 5. 계절 / 월별 이용량 ----------
    st.subheader("5. 계절·월별 이용량")
    s1, s2 = st.columns([1, 2])
    with s1:
        by_season = (
            agg["by_season"].set_index("계절").reindex(SEASON_ORDER).dropna().reset_index()
        )
        fig = px.bar(by_season, x="계절", y="이용건수", color="계절",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(**PLOT_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with s2:
        by_month = agg["by_month"]
        fig = px.line(by_month, x="연월", y="이용건수", markers=True,
                      labels={"연월": "월", "이용건수": "이용 건수"})
        fig.update_traces(line=dict(width=3), hovertemplate="%{x}<br>%{y:,}건<extra></extra>")
        fig.update_layout(**PLOT_LAYOUT, height=300, yaxis=dict(gridcolor="#eeeeee"))
        st.plotly_chart(fig, width="stretch")

    # ---------- (선택) 날씨 데이터 연동 ----------
    with st.expander("🌦 날씨 데이터로 확장하기 (선택)"):
        st.markdown("일별 `날짜, 기온, 강수량` 컬럼이 있는 CSV를 올리면 기온·강수와 이용량을 함께 볼 수 있습니다.")
        up = st.file_uploader("기상 데이터 CSV 업로드", type=["csv"])
        if up is not None:
            weather = read_uploaded_csv(up)
            dcol = pick_col(weather.columns, ["날짜", "date", "일자"])
            tcol = pick_col(weather.columns, ["기온", "평균기온", "temperature", "avg_temp", "temp"])
            rcol = pick_col(weather.columns, ["강수량", "precipitation", "rain", "강수"])
            if not all([dcol, tcol, rcol]):
                st.warning(f"컬럼을 찾지 못했습니다. 필요한 컬럼: 날짜/기온/강수량 (현재: {list(weather.columns)})")
            else:
                weather["날짜"] = pd.to_datetime(weather[dcol], errors="coerce").dt.normalize()
                weather["기온"] = pd.to_numeric(weather[tcol], errors="coerce")
                weather["강수량"] = pd.to_numeric(weather[rcol], errors="coerce").fillna(0)
                daily = agg["daily"].copy()
                daily["날짜"] = days_to_dates(daily["날짜"]).dt.normalize()
                merged = daily.merge(weather[["날짜", "기온", "강수량"]].dropna(subset=["날짜"]),
                                     on="날짜", how="inner")
                if merged.empty:
                    st.info("업로드한 기상 데이터의 날짜가 선택 기간과 겹치지 않습니다.")
                else:
                    merged["강수여부"] = merged["강수량"].gt(0).map({True: "강수 있음", False: "강수 없음"})
                    corr = merged["기온"].corr(merged["이용건수"])
                    st.metric("기온–이용건수 상관계수", f"{corr:.2f}" if pd.notna(corr) else "-")
                    fig = px.scatter(merged, x="기온", y="이용건수", color="강수여부",
                                     labels={"기온": "일 평균 기온(℃)", "이용건수": "일 이용건수"},
                                     color_discrete_map={"강수 있음": "#4C78A8", "강수 없음": "#F58518"})
                    fig.update_layout(**PLOT_LAYOUT, height=360)
                    st.plotly_chart(fig, width="stretch")

    # ---------- 출처 / 한계 ----------
    with st.expander("ℹ️ 데이터 출처 및 해석 시 주의점"):
        st.markdown(
            """
            - **출처**: 서울시 따릉이 시간대별 이용정보(2025.07~2026.06), 서울시 공공자전거 대여소 정보, 자치구 경계는 southkorea/seoul-maps(2013년 행정구역 기준) GeoJSON
            - **집계 기준**: 대여일시 기준 시간대·요일·연령대·대여소 단위로 재집계했습니다. 원본은 이미 이용건수 단위로 요약된 데이터입니다.
            - **가중 평균**: 이동거리·이용시간은 이용건수로 가중 평균했습니다(이상치는 운동량/거리/시간 물리 범위로 clip).
            - **날씨**: 제공된 원본에 기온·강수 자료가 없어, 기본 차트에서는 **계절(월 기준)을 대리 변수**로 사용했습니다. 위 업로더로 기상 데이터를 붙여 보다 직접적으로 확인할 수 있습니다.
            - **해석 주의**: 출퇴근 시간대와 이용량의 관계는 상관관계이며 인과관계를 의미하지 않습니다. 2025년 하반기~2026년 상반기 12개월에 한정된 결과입니다.
            """
        )


if __name__ == "__main__":
    main()
