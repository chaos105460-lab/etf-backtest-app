"""ETF 배당금 재투자 백테스트 — Streamlit 단일 파일 앱.

실행: streamlit run app.py
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="나만의 ETF 백테스트",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ETF_A_OPTIONS = ["QQQ", "QLD", "SPY", "VOO", "QQQM", "SCHD", "직접 입력"]
ETF_B_OPTIONS = ["DIVO", "JEPI", "SCHD", "JEPQ", "QDVO", "SPYI", "QQQI", "직접 입력"]


@st.cache_data(ttl=3600, show_spinner=False)
def load_etf(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Yahoo Finance의 분할 조정 종가와 배당락일 배당액을 조회한다.

    yfinance history의 end는 exclusive이므로 하루를 더한다.
    """
    raw = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
    )
    if raw.empty or "Close" not in raw.columns:
        raise ValueError(f"{ticker}: 주가 데이터가 없습니다. 티커·기간을 확인해 주세요.")

    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    dividends = raw["Dividends"] if "Dividends" in raw.columns else pd.Series(0.0, index=raw.index)
    result = pd.DataFrame({"price": raw["Close"].astype(float), "dividend": dividends.astype(float)})
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=["price"])
    result["dividend"] = result["dividend"].fillna(0.0)
    if (result["price"] <= 0).any() or (result["dividend"] < 0).any():
        raise ValueError(f"{ticker}: 유효하지 않은 주가 또는 배당 데이터가 있습니다.")
    return result


def simulate_portfolio(
    asset_a: pd.DataFrame,
    asset_b: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    initial_money: float,
    initial_weight_a: float,
) -> tuple[pd.DataFrame, dict]:
    """매도 없이 배당금으로 부족 비중을 우선 매수. 모든 금액은 USD.

    가정: 배당락일에 배당금 지급 및 해당 일자 종가 매수,
          세금·수수료 없음, 소수점 주식 허용.
    """
    if not (initial_money > 0) or not (0 <= initial_weight_a <= 1):
        raise ValueError("투자금은 양수, 비중은 0~100%여야 합니다.")

    # 모든 전략에 동일한 거래일 기준으로 두 ETF 데이터를 정렬한다.
    joined = asset_a.rename(columns={"price": "pa", "dividend": "da"}).join(
        asset_b.rename(columns={"price": "pb", "dividend": "db"}), how="inner"
    )
    joined = joined.dropna(subset=["pa", "pb"])
    joined[["da", "db"]] = joined[["da", "db"]].fillna(0)
    if len(joined) < 3:
        raise ValueError("두 ETF가 공통으로 거래된 날짜가 부족합니다. 분석 기간을 넓혀 주세요.")

    first = joined.iloc[0]
    shares_a = initial_money * initial_weight_a / first["pa"]
    shares_b = initial_money * (1 - initial_weight_a) / first["pb"]
    total_distributions = 0.0
    rows = []

    for i, (day, r) in enumerate(joined.iterrows()):
        pa, pb = float(r["pa"]), float(r["pb"])
        # 매수 시작일 이전 보유분은 없으므로 첫 거래일 배당은 제외.
        cash = (shares_a * r["da"] + shares_b * r["db"]) if i else 0.0
        total_distributions += cash

        value_a, value_b = shares_a * pa, shares_b * pb
        if cash > 0:
            target_a = (value_a + value_b + cash) * initial_weight_a
            # A의 목표 평가액에 미달한 만큼 먼저 매수; 나머지는 B.
            spend_a = min(cash, max(0.0, target_a - value_a))
            spend_b = cash - spend_a
            shares_a += spend_a / pa
            shares_b += spend_b / pb

        value_a, value_b = shares_a * pa, shares_b * pb
        portfolio_value = value_a + value_b
        rows.append({
            "날짜": day,
            "포트폴리오($)": portfolio_value,
            f"{symbol_a} 평가금($)": value_a,
            f"{symbol_b} 평가금($)": value_b,
            f"{symbol_a} 비중(%)": value_a / portfolio_value * 100,
            f"{symbol_b} 비중(%)": value_b / portfolio_value * 100,
            "당일 재투자 배당금($)": cash,
        })

    daily = pd.DataFrame(rows).set_index("날짜")
    values = daily["포트폴리오($)"]
    dd = (values / values.cummax() - 1) * 100
    daily["낙폭(%)"] = dd
    returns = values.pct_change(fill_method=None).dropna()
    daily["60일 연율화 변동성(%)"] = returns.rolling(60).std(ddof=1) * np.sqrt(252) * 100

    # 최대 낙폭 저점과 그 직전 '마지막' 최고점
    trough_idx = int(np.argmin(dd.to_numpy()))
    peak_price = float(values.iloc[:trough_idx + 1].max())
    prior = values.iloc[:trough_idx + 1].to_numpy()
    peak_idx = int(np.flatnonzero(np.isclose(prior, peak_price, rtol=1e-10, atol=1e-8))[-1])
    recoveries = np.flatnonzero(values.to_numpy()[trough_idx:] >= peak_price * (1 - 1e-10))
    recovery_idx = trough_idx + int(recoveries[0]) if len(recoveries) else None
    peak_day, trough_day = daily.index[peak_idx], daily.index[trough_idx]
    recovery_day = daily.index[recovery_idx] if recovery_idx is not None else None

    elapsed_years = (daily.index[-1] - daily.index[0]).days / 365.25
    volatility = float(returns.std(ddof=1))
    end_value = float(values.iloc[-1])

    metrics = {
        "start": daily.index[0], "end": daily.index[-1],
        "initial": initial_money, "end_value": end_value,
        "total_return": (end_value / initial_money - 1) * 100,
        "cagr": ((end_value / initial_money) ** (1 / elapsed_years) - 1) * 100,
        "mdd": float(dd.min()),
        "daily_vol": volatility * 100,
        "annual_vol": volatility * np.sqrt(252) * 100,
        "peak_day": peak_day, "trough_day": trough_day,
        "recovery_day": recovery_day,
        "peak_to_recovery_days": (recovery_day - peak_day).days if recovery_day is not None else None,
        "trough_to_recovery_days": (recovery_day - trough_day).days if recovery_day is not None else None,
        "cumulative_dividends": total_distributions,
        "final_weight_a": float(daily[f"{symbol_a} 비중(%)"].iloc[-1]),
        "final_weight_b": float(daily[f"{symbol_b} 비중(%)"].iloc[-1]),
    }
    return daily, metrics


def plot_lines(series: dict[str, pd.Series], title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    for label, values in series.items():
        fig.add_trace(go.Scatter(x=values.index, y=values, name=label, mode="lines"))
    fig.update_layout(
        title=title, yaxis_title=ytitle, xaxis_title="날짜",
        hovermode="x unified", margin=dict(l=12, r=12, t=55, b=12),
        legend=dict(orientation="h", y=-0.2), height=390,
    )
    return fig


def render_app() -> None:
    st.title("📈 나만의 ETF 백테스트")
    st.caption("배당금으로만 비중 조절 · 주식 매도 없음 · 기본 설정 QQQ 50% + DIVO 50%")

    with st.sidebar:
        st.header("⚙️ 투자 설정")
        with st.form("backtest_form"):
            a_option = st.selectbox("ETF A", ETF_A_OPTIONS, index=0)
            ticker_a = st.text_input("ETF A 티커", "QQQ") if a_option == "직접 입력" else a_option
            b_option = st.selectbox("ETF B", ETF_B_OPTIONS, index=0)
            ticker_b = st.text_input("ETF B 티커", "DIVO") if b_option == "직접 입력" else b_option
            weight = st.slider("ETF A 최초 비중 (%)", 0, 100, 50, 5)
            st.caption(f"ETF B 최초 비중: {100 - weight}%")
            money = st.number_input("초기 투자금 (USD)", min_value=100.0, value=10000.0, step=1000.0)
            start = st.date_input("분석 시작일", value=date(2020, 5, 21), min_value=date(1993, 1, 1), max_value=date.today())
            end = st.date_input("분석 종료일", value=date.today(), min_value=date(1993, 1, 1), max_value=date.today())
            submitted = st.form_submit_button("📊 백테스트 실행", type="primary", use_container_width=True)
        st.caption("가격 데이터: Yahoo Finance(yfinance). 날짜와 티커에 따라 결과가 달라질 수 있습니다.")

    # 다운로드나 다른 화면 상호작용으로 앱이 재실행돼도 마지막 분석을 유지한다.
    if submitted:
        st.session_state["last_parameters"] = (
            ticker_a, ticker_b, weight, money, start, end
        )
    if "last_parameters" not in st.session_state:
        st.info("왼쪽 설정에서 ETF와 투자 비중을 선택한 뒤 **백테스트 실행**을 눌러 주세요.")
        st.stop()
    ticker_a, ticker_b, weight, money, start, end = st.session_state["last_parameters"]
    ticker_a, ticker_b = ticker_a.strip().upper(), ticker_b.strip().upper()
    if not ticker_a or not ticker_b or ticker_a == ticker_b:
        st.error("서로 다른 ETF 두 개를 선택해 주세요.")
        st.stop()
    if start >= end:
        st.error("분석 종료일이 시작일보다 늦어야 합니다.")
        st.stop()

    with st.spinner("ETF 가격과 배당 내역을 가져와 계산하는 중입니다..."):
        try:
            asset_a = load_etf(ticker_a, start, end)
            asset_b = load_etf(ticker_b, start, end)
            daily, m = simulate_portfolio(asset_a, asset_b, ticker_a, ticker_b, money, weight / 100)
        except Exception as exc:
            st.error(f"분석할 수 없습니다: {exc}")
            st.info("티커가 정확한지, 두 ETF의 공통 거래 기간이 충분한지 확인한 뒤 다시 시도해 주세요.")
            st.stop()

    st.subheader(f"{ticker_a} {weight}% + {ticker_b} {100 - weight}%")
    st.caption(f"실제 공통 데이터 기간: {m['start']:%Y-%m-%d} ~ {m['end']:%Y-%m-%d}")
    if pd.Timestamp(start) < m["start"]:
        st.warning("요청한 시작일보다 늦게 분석이 시작됐습니다. 신규 출시 ETF 또는 공통 거래일 제한 때문일 수 있습니다.")

    one = st.columns(4)
    one[0].metric("최대 낙폭 (MDD)", f"{m['mdd']:.2f}%")
    one[1].metric("연평균 복리 수익률", f"{m['cagr']:.2f}%")
    one[2].metric("연율화 변동성", f"{m['annual_vol']:.2f}%")
    one[3].metric("최종 자산", f"${m['end_value']:,.0f}")

    two = st.columns(4)
    two[0].metric("누적 총수익률", f"{m['total_return']:.2f}%")
    two[1].metric("일간 변동성", f"{m['daily_vol']:.2f}%")
    two[2].metric("고점 → 회복", f"{m['peak_to_recovery_days']}일" if m["peak_to_recovery_days"] is not None else "미회복")
    two[3].metric("누적 재투자 분배금", f"${m['cumulative_dividends']:,.0f}")

    st.write(
        f"**MDD 고점:** {m['peak_day']:%Y-%m-%d}　"
        f"**저점:** {m['trough_day']:%Y-%m-%d}　"
        f"**회복:** {m['recovery_day']:%Y-%m-%d}" if m["recovery_day"] is not None else
        f"**MDD 고점:** {m['peak_day']:%Y-%m-%d}　**저점:** {m['trough_day']:%Y-%m-%d}　**회복:** 미회복"
    )
    st.caption(
        f"저점 → 회복: {m['trough_to_recovery_days']}일" if m["trough_to_recovery_days"] is not None
        else "저점 이후 현재까지 기존 고점을 회복하지 못했습니다."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["📈 자산 성장", "📉 하락과 변동성", "⚖️ 실제 투자 비중", "📄 데이터 다운로드"])
    with tab1:
        st.plotly_chart(plot_lines({"포트폴리오": daily["포트폴리오($)"]}, "포트폴리오 자산 추이", "미국 달러 ($)"), use_container_width=True)
    with tab2:
        fig = plot_lines({"고점 대비 하락률": daily["낙폭(%)"]}, "고점 대비 하락률 (Drawdown)", "%")
        fig.add_trace(go.Scatter(
            x=[m["trough_day"]], y=[m["mdd"]],
            mode="markers", name=f"MDD {m['mdd']:.2f}%",
            marker=dict(size=11, symbol="diamond"),
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(plot_lines(
            {"최근 60거래일 변동성": daily["60일 연율화 변동성(%)"]},
            "60거래일 이동 연율화 변동성", "%",
        ), use_container_width=True)
    with tab3:
        st.write(f"최종 실제 비중: **{ticker_a} {m['final_weight_a']:.2f}% / {ticker_b} {m['final_weight_b']:.2f}%**")
        st.plotly_chart(plot_lines({
            ticker_a: daily[f"{ticker_a} 비중(%)"],
            ticker_b: daily[f"{ticker_b} 비중(%)"],
            f"{ticker_a} 목표": pd.Series(weight, index=daily.index),
        }, "실제 보유 비중 변화", "%"), use_container_width=True)
        st.info("분배금만으로 부족한 종목을 사기 때문에, 50:50으로 시작해도 실제 비중은 시간이 지나며 달라집니다.")
    with tab4:
        st.dataframe(daily.tail(40).sort_index(ascending=False).round(3), use_container_width=True)
        csv = daily.to_csv(index=True, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ 전체 결과 CSV 다운로드", csv,
            file_name=f"ETF_backtest_{ticker_a}_{ticker_b}_{weight}_{100-weight}.csv",
            mime="text/csv", use_container_width=True,
        )

    st.divider()
    st.caption(
        "**모형 가정:** 최초 1회 투자, 추가 납입·매도 없음. 분배금은 배당락일에 수령하여 "
        "해당일 종가에 즉시 재투자, 소수점 매수 가능, 세금·수수료·환율 제외. "
        "배당을 별도로 투자하므로 배당을 포함한 Adj Close가 아닌 분할 조정 Close를 사용합니다. "
        "과거 MDD는 미래 최대손실을 보장하지 않습니다."
    )


if __name__ == "__main__":
    render_app()
