import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize

# =================== CONFIG ===================
st.set_page_config(page_title="Fortune Financial Strategies - Asset Allocation", layout="wide")
st.title("Fortune Financial Strategies - Asset Allocation")
st.caption("Build: v18.7 — % em todos os gráficos + labels de baldes em % + nomes + catálogo + Correlação Macro + Regimes/Baldes + Pesos Manuais Y/Y")

# >>> Tolerância numérica para checagem de DD <<<
EPS = 1e-6

# ====== CACHE DE OTIMIZAÇÃO EM SESSION_STATE (não depende do benchmark) ======
if "otm_cache" not in st.session_state:
    st.session_state["otm_cache"] = {}
if "last_otm_key" not in st.session_state:
    st.session_state["last_otm_key"] = None

def make_cache_key(ativos_ok, perfil, criterio, retorno_alvo, max_dd_user):
    return (
        "v18.7-session",
        tuple(ativos_ok),
        perfil,
        criterio,
        round(float(retorno_alvo), 8),
        round(float(max_dd_user) if max_dd_user is not None else -1.0, 8),
    )

# Arquivos locais
CAMINHO_PLANILHA_ATIVOS = "ativos.xlsx"
CAMINHO_CLASSIFICACAO  = "classificacao_ativos.xlsx"
CAMINHO_BENCHMARK      = "benchmark.xlsx"
CAMINHO_RF             = "taxa livre de risco.xlsx"
CAMINHO_MACRO          = "macro.xlsx"

# Regras
MIN_PESO = 0.01
CLASS_ORDER = ["Caixa", "Renda Fixa", "Ações", "Commodities"]

# Limite por ativo específico (AT1)
AT1_TICKER_EXATO = "AT1 LN Equity"
AT1_CAPS = {"Conservador": 0.05, "Moderado": 0.10, "Agressivo": 0.15}
def is_at1(ticker: str) -> bool:
    return str(ticker).strip() == AT1_TICKER_EXATO

# ================ CACHE HELPERS ================
@st.cache_data(show_spinner=False)
def load_excel_cached(path: str):
    return pd.read_excel(path)

@st.cache_data(show_spinner=False)
def prep_prices(df_raw: pd.DataFrame):
    df = df_raw.copy()
    if df.shape[1] == 0:
        raise ValueError("A planilha de ativos não possui colunas.")
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index().ffill()
    df.columns = df.columns.astype(str).str.strip()
    returns = df.pct_change().dropna()
    return df, returns

@st.cache_data(show_spinner=False)
def load_risk_free(source, dayfirst=True):
    rf = pd.read_excel(source)
    rf.columns = rf.columns.astype(str).str.strip()
    taxa_cols = [c for c in rf.columns if c.lower() != "date" and pd.api.types.is_numeric_dtype(rf[c])]
    if not taxa_cols:
        return None
    col = taxa_cols[0]
    rf["Date"] = pd.to_datetime(rf["Date"], dayfirst=dayfirst, errors="coerce")
    rf = rf.dropna(subset=["Date"]).set_index("Date").sort_index()
    rf_daily = (1 + rf[col] / 100.0) ** (1/252) - 1
    return rf_daily

@st.cache_data(show_spinner=False)
def compute_stats(returns_sel: pd.DataFrame):
    mean_returns = returns_sel.mean() * 252
    cov_matrix   = returns_sel.cov() * 252
    return mean_returns, cov_matrix

# ================ MACRO LOADER (robusto) ================
@st.cache_data(show_spinner=False)
def load_macro_flex(path_macro: str):
    """
    Lê macro.xlsx e mapeia colunas por substring (case-insensitive):
      - Fed Funds: 'fedl01','fed funds','fedfund','effective fed funds','effr','us000fed'
      - CPI YoY:   contém 'cpi' e 'yoy' (aceita 'xyoy'), detecta 'core'
      - PCE YoY:   contém 'pce' e 'yoy' (detecta 'core')
      - DXY:       'dxy'
    Retorna colunas padronizadas: 'fedfunds','cpi_yoy','cpi_core_yoy','pce_yoy','pce_core_yoy','dxy'
    """
    if not os.path.exists(path_macro):
        return None

    def _read_sheet(sheet=None):
        dfm = pd.read_excel(path_macro, sheet_name=sheet)
        if "Date" not in dfm.columns:
            dfm.rename(columns={dfm.columns[0]: "Date"}, inplace=True)
        dfm["Date"] = pd.to_datetime(dfm["Date"], dayfirst=True, errors="coerce")
        dfm = dfm.dropna(subset=["Date"]).set_index("Date").sort_index()
        dfm.rename(columns={c: str(c).strip() for c in dfm.columns}, inplace=True)
        low_cols = {c.lower().strip(): c for c in dfm.columns}
        return dfm, low_cols

    def _to_numeric_clean(s):
        if s.dtype == object:
            s = s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
        return pd.to_numeric(s, errors="coerce")

    try:
        sheets = pd.ExcelFile(path_macro).sheet_names
    except Exception:
        sheets = [None]

    out = None
    for sh in sheets:
        dfm, low_cols = _read_sheet(sh)

        def _find(pat_list, require_all=False):
            for low, orig in low_cols.items():
                s = low.replace(" ", "").replace("_", "")
                ok = all(p in s for p in pat_list) if require_all else any(p in s for p in pat_list)
                if ok:
                    return orig
            return None

        fed_col = _find(["fedl01", "fedfund", "fedfunds", "effectivefedfunds", "effr", "us000fed"])
        cpi_core_col = _find(["cpi", "yoy", "core"], require_all=True)
        cpi_col      = _find(["cpi", "yoy"]) if cpi_core_col is None else None
        pce_core_col = _find(["pce", "yoy", "core"], require_all=True)
        pce_col      = _find(["pce", "yoy"]) if pce_core_col is None else None
        dxy_col      = _find(["dxy"])

        tmp = pd.DataFrame(index=dfm.index)
        if fed_col is not None and fed_col in dfm.columns:
            tmp["fedfunds"] = _to_numeric_clean(dfm[fed_col])
        if cpi_core_col is not None and cpi_core_col in dfm.columns:
            tmp["cpi_core_yoy"] = _to_numeric_clean(dfm[cpi_core_col])
        if cpi_col is not None and cpi_col in dfm.columns:
            tmp["cpi_yoy"] = _to_numeric_clean(dfm[cpi_col])
        if pce_core_col is not None and pce_core_col in dfm.columns:
            tmp["pce_core_yoy"] = _to_numeric_clean(dfm[pce_core_col])
        if pce_col is not None and pce_col in dfm.columns:
            tmp["pce_yoy"] = _to_numeric_clean(dfm[pce_col])
        if dxy_col is not None and dxy_col in dfm.columns:
            tmp["dxy"] = _to_numeric_clean(dfm[dxy_col])

        tmp = tmp.dropna(how="all")
        if not tmp.empty:
            out = tmp if out is None else out.combine_first(tmp)

    if out is None:
        return None

    out = out.sort_index()
    for c in list(out.columns):
        if out[c].dropna().empty:
            out.drop(columns=[c], inplace=True)
    return out if not out.empty else None

# ================ FUNÇÕES BASE (métricas/otimização) ================
def calc_sharpe(returns, rf_daily=None):
    if rf_daily is None or (hasattr(rf_daily, "empty") and getattr(rf_daily, "empty", False)):
        ret_ann = (1 + returns.mean())**252 - 1
        vol_ann = returns.std() * np.sqrt(252)
        return ret_ann / vol_ann if vol_ann != 0 else float("nan")
    ret_alinh, rf_alinh = returns.align(rf_daily, join="inner")
    if ret_alinh.empty:
        return float("nan")
    excess = ret_alinh - rf_alinh
    mu_excess_ann = excess.mean() * 252
    vol_ann = excess.std() * np.sqrt(252)
    return mu_excess_ann / vol_ann if vol_ann != 0 else float("nan")

def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    return ret, vol

def class_constraints(tickers, perfil, restricoes_por_perfil, classe_ativos):
    restricoes = restricoes_por_perfil[perfil]
    classes = pd.Series([classe_ativos.get(t, "Outros") for t in tickers])
    cons = []
    for classe, (mn, mx) in restricoes.items():
        idxs = [i for i, c in enumerate(classes) if c == classe]
        if idxs:
            cons.append({'type': 'ineq', 'fun': (lambda idxs=idxs, mv=mn: lambda x: np.sum(x[idxs]) - mv)()})
            cons.append({'type': 'ineq', 'fun': (lambda idxs=idxs, mv=mx: lambda x: mv - np.sum(x[idxs]))()})
    return cons

def build_bounds(tickers, perfil, lower_is_minpeso: bool):
    bounds = []
    cap_at1 = AT1_CAPS.get(perfil, 1.0)
    for t in tickers:
        ub = cap_at1 if is_at1(t) else 1.0
        lb = MIN_PESO if lower_is_minpeso else 0.0
        bounds.append((lb, ub))
    return tuple(bounds)

def max_drawdown_of_weights(w, returns_df):
    pr = (returns_df * w).sum(axis=1)
    cum = (1 + pr).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min()), pr

def dd_constraint_factory(returns_df, dd_max):
    return lambda x: dd_max - abs(max_drawdown_of_weights(x, returns_df)[0])

def minimize_volatility_with_constraints(mean_returns, cov_matrix, target_return, tickers, perfil, restricoes_por_perfil, classe_ativos):
    n = len(tickers)
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x, mu=mean_returns.values: np.dot(x, mu) - target_return}
    ]
    cons += class_constraints(tickers, perfil, restricoes_por_perfil, classe_ativos)
    bounds = build_bounds(tickers, perfil, lower_is_minpeso=False)
    x0 = np.ones(n) / n
    Sigma = cov_matrix.values
    obj = lambda x: float(np.sqrt(((Sigma @ x).dot(x))))
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise ValueError("Otimização falhou para o retorno alvo informado.")
    return res.x

@st.cache_data(show_spinner=False)
def gerar_fronteira_eficiente(mean_returns, cov_matrix, tickers, perfil, restricoes_por_perfil, classe_ativos):
    rets, vols = [], []
    for r in np.linspace(0.01, 0.15, 40):
        try:
            w = minimize_volatility_with_constraints(mean_returns, cov_matrix, r, tickers, perfil, restricoes_por_perfil, classe_ativos)
            rr, vv = portfolio_performance(w, mean_returns.values, cov_matrix.values)
            rets.append(rr); vols.append(vv)
        except Exception:
            pass
    return np.array(rets), np.array(vols)

def gmvp_weights(mean_returns, cov_matrix, tickers, perfil, restricoes_por_perfil, classe_ativos):
    n = len(tickers)
    bounds = build_bounds(tickers, perfil, lower_is_minpeso=False)
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    cons += class_constraints(tickers, perfil, restricoes_por_perfil, classe_ativos)
    x0 = np.ones(n)/n
    Sigma = cov_matrix.values
    obj = lambda x: float(np.sqrt(((Sigma @ x).dot(x))))
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise ValueError("Falha na otimização GMVP.")
    return res.x

def max_sharpe_weights(mean_returns, cov_matrix, tickers, perfil, restricoes_por_perfil, classe_ativos, rf_daily=None):
    rf_ann = 0.0 if (rf_daily is None or getattr(rf_daily, "empty", False)) else float((1 + rf_daily.mean())**252 - 1)
    n = len(tickers)
    bounds = build_bounds(tickers, perfil, lower_is_minpeso=False)
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    cons += class_constraints(tickers, perfil, restricoes_por_perfil, classe_ativos)
    x0 = np.ones(n)/n
    mu = mean_returns.values
    Sigma = cov_matrix.values
    mu_excess = mu - rf_ann
    def neg_sharpe(x):
        vol = float(np.sqrt(np.dot(x.T, np.dot(Sigma, x))))
        if vol <= 1e-12: return 1e6
        return -float(np.dot(x, mu_excess)/vol)
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise ValueError("Falha na otimização de Máx. Sharpe.")
    return res.x

# ========= ENFORCE 0% OU >=1% =========
def enforce_min1(weights, tickers, perfil, mean_returns, cov_matrix,
                 restricoes_por_perfil, classe_ativos,
                 mode="min_vol_target", target_return=None,
                 returns_sel=None, max_dd=None, rf_daily=None):
    w0 = np.array(weights, float)
    order = np.argsort(-w0)
    ativos = list(tickers)

    rf_ann = 0.0 if (rf_daily is None or getattr(rf_daily, "empty", False)) else float((1 + rf_daily.mean())**252 - 1)

    def _solve_subset(idxs):
        t_sub = [ativos[i] for i in idxs]
        mu  = mean_returns.loc[t_sub].values
        Sig = cov_matrix.loc[t_sub, t_sub].values

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        cons += class_constraints(t_sub, perfil, restricoes_por_perfil, classe_ativos)
        if mode in ("min_vol_target", "min_vol_target_dd"):
            cons.append({'type': 'eq', 'fun': lambda x, mu=mu: np.dot(x, mu) - float(target_return)})
        if mode == "min_vol_target_dd":
            assert returns_sel is not None and max_dd is not None
            ret_sub = returns_sel[t_sub]
            cons.append({'type':'ineq','fun': dd_constraint_factory(ret_sub, max_dd)})

        bounds = [(MIN_PESO, 1.0) for _ in t_sub]
        x0 = np.ones(len(t_sub)) / len(t_sub)

        if mode in ("min_vol_target", "min_vol_target_dd", "gmvp"):
            obj = lambda x: float(np.sqrt(((Sig @ x).dot(x))))
        elif mode == "max_sharpe":
            mu_exc = mu - rf_ann
            def obj(x):
                vol = float(np.sqrt(np.dot(x.T, np.dot(Sig, x))))
                if vol <= 1e-12: return 1e6
                return -float(np.dot(x, mu_exc)/vol)
        else:
            raise ValueError("mode inválido")

        return minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)

    active = [int(i) for i in order if w0[i] >= MIN_PESO]
    if not active:
        active = [int(order[0])]

    for k in range(len(active), len(ativos) + 1):
        idxs = sorted(active[:k])
        res = _solve_subset(idxs)
        if res.success:
            w_full = np.zeros(len(ativos))
            for j, i in enumerate(idxs):
                w_full[i] = res.x[j]
            return w_full

    w0[w0 < MIN_PESO] = 0.0
    s = w0.sum()
    return w0 / s if s > 0 else np.ones(len(ativos)) / len(ativos)

# ========= UI HELPERS =========
def percent_axis(ax, axis='y', mode='unit'):
    """
    mode:
      - 'unit'  -> dados em fração (0.05 = 5%)        => PercentFormatter(1.0)
      - 'pp'    -> dados em pontos percentuais (5=5%) => PercentFormatter(100.0)
    """
    fmt = PercentFormatter(1.0 if mode == 'unit' else 100.0)
    if axis == 'y':
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)

# [%] helpers: rótulos com % para baldes
def _fmt_pct_value(v):
    """Formata número em % com casas dinâmicas (para 100+, 10+, <10)."""
    if abs(v) >= 100:
        return f"{v:.0f}%"
    elif abs(v) >= 10:
        return f"{v:.1f}%"
    else:
        return f"{v:.2f}%"

def make_bucket_label(a, b, kind):
    """
    kind='pct' -> a,b em fração (ex: 0.01=1%); kind='pp' -> a,b já em %.
    Retorna 'X%–Y%'.
    """
    if kind == 'pct':
        a *= 100.0; b *= 100.0
    return f"{_fmt_pct_value(a)}–{_fmt_pct_value(b)}"

def indicadores_totais(portfolio_returns, rf_daily=None):
    ann_return = (1 + portfolio_returns.mean())**252 - 1
    ann_vol    = portfolio_returns.std() * np.sqrt(252)
    sharpe     = calc_sharpe(portfolio_returns, rf_daily)
    cum = (1 + portfolio_returns).cumprod(); peak = cum.cummax()
    dd = (cum - peak) / peak
    max_drawdown = dd.min()
    return ann_return, ann_vol, sharpe, max_drawdown

def indicadores_por_ano(portfolio_returns, rf_daily=None):
    df_ = portfolio_returns.to_frame("ret"); df_["ano"] = df_.index.year
    out = []
    for ano, g in df_.groupby("ano"):
        ret = (1 + g["ret"]).prod() - 1
        vol = g["ret"].std() * np.sqrt(252)
        rf_sub = None
        if rf_daily is not None:
            _, rf_sub = g["ret"].align(rf_daily, join="inner")
        sharpe = calc_sharpe(g["ret"], rf_sub)
        cum = (1 + g["ret"]).cumprod(); peak = cum.cummax()
        dd = (cum - peak)/peak; max_dd = dd.min()
        out.append([str(ano), f"{ret*100:.2f}%", f"{vol*100:.2f}%", f"{sharpe:.2f}", f"{max_dd*100:.2f}%"])
    return pd.DataFrame(out, columns=["Ano", "Retorno", "Volatilidade", "Sharpe", "Máx. Drawdown"])

# ========= OTIMIZAÇÃO PRINCIPAL =========
@st.cache_data(show_spinner=False)
def otimizar_portfolio(criterio, perfil, retorno_alvo, max_dd_user,
                       df_sel_columns, mean_returns, cov_matrix,
                       limites_demo, classe_ativos, returns_sel, rf_daily):
    if criterio == "Retorno alvo":
        w_opt_base = minimize_volatility_with_constraints(
            mean_returns, cov_matrix, retorno_alvo,
            df_sel_columns, perfil, limites_demo, classe_ativos
        )
        pesos_otimizados = enforce_min1(
            w_opt_base, df_sel_columns, perfil, mean_returns, cov_matrix,
            limites_demo, classe_ativos,
            mode="min_vol_target", target_return=retorno_alvo, rf_daily=rf_daily
        )
        achieved_dd, _ = max_drawdown_of_weights(pesos_otimizados, returns_sel)
        dd_info = None
    else:
        n = len(df_sel_columns)
        bounds = build_bounds(df_sel_columns, perfil, lower_is_minpeso=False)
        Sigma = cov_matrix.values
        mu = mean_returns.values
        x0 = np.ones(n)/n
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x, mu=mu: np.dot(x, mu) - retorno_alvo},
            {'type': 'ineq','fun': dd_constraint_factory(returns_sel, max_dd_user)}
        ]
        cons += class_constraints(df_sel_columns, perfil, limites_demo, classe_ativos)
        obj = lambda x: float(np.sqrt(((Sigma @ x).dot(x))))
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            pesos_otimizados = enforce_min1(
                res.x, df_sel_columns, perfil, mean_returns, cov_matrix,
                limites_demo, classe_ativos,
                mode="min_vol_target_dd", target_return=retorno_alvo,
                returns_sel=returns_sel, max_dd=max_dd_user, rf_daily=rf_daily
            )
            achieved_dd, _ = max_drawdown_of_weights(pesos_otimizados, returns_sel)
            dd_info = ("ok", achieved_dd) if abs(achieved_dd) <= (max_dd_user + EPS) else ("violado", achieved_dd)
        else:
            raise ValueError("❌ Nenhuma carteira factível encontrada para Retorno alvo + Máx. DD "
                             "(relaxe o DD, ajuste limites por classe, inclua Caixa ou amplie tolerância).")
    return pesos_otimizados, dd_info

# ========= Seleção ordenada por classe =========
def class_index(classe: str) -> int:
    return CLASS_ORDER.index(classe) if classe in CLASS_ORDER else len(CLASS_ORDER) + 1

def sort_columns_by_class(cols, classe_ativos):
    return sorted(list(cols), key=lambda t: (class_index(classe_ativos.get(t, "Outros")),
                                             classe_ativos.get(t, "Outros"), str(t)))

# ========= UTIL: nomes amigáveis =========
DISPLAY_COL = "Nome"
SHORT_COL   = "NomeCurto"
def dedup_labels(labels, keys):
    seen, out = {}, []
    for lab, key in zip(labels, keys):
        base = (lab or str(key)).strip()
        out.append(base if base not in seen else f"{base} [{key}]")
        seen[base] = 1
    return out

def render_dashboard(pesos, titulo, returns, mean_returns, cov_matrix, df, classe_ativos,
                     benchmark_retornos=None, benchmark_nome=None,
                     mostrar_fronteira=False, frontier_vols=None, frontier_returns=None,
                     rf_daily=None, extra_points=None, get_name=lambda x: x):
    st.subheader(titulo)
    cols_ord = sort_columns_by_class(df.columns, classe_ativos)
    pesos_map = pd.Series(pesos, index=df.columns)
    pesos_ord = pesos_map.reindex(cols_ord).fillna(0.0).values
    labels_amig = dedup_labels([get_name(t) for t in cols_ord], cols_ord)

    result_df = pd.DataFrame({
        "Ativo": labels_amig,
        "Classe": [classe_ativos.get(t, "Outros") for t in cols_ord],
        "Peso (%)": np.round(pesos_ord * 100, 2),
    })
    st.dataframe(result_df[result_df["Peso (%)"] > 0], use_container_width=True)

    portfolio_returns = (returns * pesos).sum(axis=1)

    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        dados = result_df[result_df["Peso (%)"] > 0]
        if not dados.empty:
            ax.pie(dados["Peso (%)"], labels=dados["Ativo"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal"); ax.set_title("Gráfico de Pizza")
        st.pyplot(fig)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        port_cum = (1 + portfolio_returns).cumprod() - 1
        port_cum.plot(ax=ax2, label="Carteira", lw=2)
        if benchmark_retornos is not None and not getattr(benchmark_retornos, "empty", True):
            port_alinh, bench_alinh = portfolio_returns.align(benchmark_retornos, join="inner")
            ((1 + bench_alinh).cumprod() - 1).plot(ax=ax2, ls="--", lw=1.5, label=(benchmark_nome or "Benchmark"))
        percent_axis(ax2, 'y', 'unit')
        ax2.set_title("Evolução do Portfólio (acumulado, %)")
        ax2.legend(); ax2.grid(alpha=0.25, linewidth=0.6)
        st.pyplot(fig2)

    with c3:
        fig3, ax3 = plt.subplots(figsize=(5, 3.2))
        if mostrar_fronteira and (frontier_vols is not None) and (frontier_returns is not None):
            ax3.plot(frontier_vols, frontier_returns, lw=1.2, label="Fronteira")
            r_now, v_now = portfolio_performance(pesos, mean_returns.values, cov_matrix.values)
            ax3.scatter(v_now, r_now, label="Ponto atual")
            if extra_points:
                for name, w, color, marker in extra_points:
                    r, v = portfolio_performance(w, mean_returns.values, cov_matrix.values)
                    ax3.scatter(v, r, label=name)
            percent_axis(ax3, 'x', 'unit')
            percent_axis(ax3, 'y', 'unit')
            ax3.set_xlabel("Volatilidade (%)"); ax3.set_ylabel("Retorno (%)"); ax3.legend()
            ax3.set_title("Fronteira Eficiente"); ax3.grid(alpha=0.25, linewidth=0.6)
        else:
            ax3.axis("off")
        st.pyplot(fig3)

    st.subheader("📊 Indicadores de Performance Totais")
    ann_return, ann_vol, sharpe, max_dd = indicadores_totais(portfolio_returns, rf_daily)
    st.table({
        "Retorno Anualizado": f"{ann_return*100:.2f}%",
        "Volatilidade Anualizada": f"{ann_vol*100:.2f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Máximo Drawdown": f"{max_dd*100:.2f}%",
    })
    st.subheader("📆 Indicadores por Ano")
    st.dataframe(indicadores_por_ano(portfolio_returns, rf_daily), use_container_width=True)

# =================== SIDEBAR ===================
st.sidebar.header("Configurações")
with st.sidebar.form("otimizacao_form", clear_on_submit=False):
    criterio = st.radio("⚙️ Critério de alocação", ["Retorno alvo", "Retorno alvo + Máx. DD"])
    perfil = st.selectbox("Perfil do investidor:", ["Conservador", "Moderado", "Agressivo"])
    retorno_alvo = st.slider("🎯 Retorno alvo anual (%)", 2.0, 20.0, 6.0, 0.1) / 100
    max_dd_user = None
    if criterio == "Retorno alvo + Máx. DD":
        max_dd_user = st.slider("📉 Máx. Drawdown permitido (%)", 1.0, 50.0, 20.0, 0.5) / 100
    limites_demo = {
        "Conservador": {"Caixa": (0, 1.0), "Ações": (0, 0.2), "Commodities": (0, 0.05), "Renda Fixa": (0.0, 1.0)},
        "Moderado":    {"Caixa": (0, 0.5), "Ações": (0, 0.5), "Commodities": (0, 0.15), "Renda Fixa": (0.0, 1.0)},
        "Agressivo":   {"Caixa": (0, 0.25), "Ações": (0, 1.0), "Commodities": (0, 0.3), "Renda Fixa": (0, 0.6)},
    }
    st.markdown("### Limites por Classe")
    for classe in CLASS_ORDER:
        if classe in limites_demo[perfil]:
            mn, mx = limites_demo[perfil][classe]
            st.write(f"*{classe}*: {mn*100:.0f}% – {mx*100:.0f}%")
    st.write(f"*{AT1_TICKER_EXATO} — teto por ativo*: até {int(AT1_CAPS[perfil]*100)}%")
    do_calc = st.form_submit_button("🚀 Calcular otimização")

# ================ PIPELINE ================
try:
    # 1) Preços
    df_raw = load_excel_cached(CAMINHO_PLANILHA_ATIVOS)
    df, returns = prep_prices(df_raw)

    # 2) RF (opcional)
    rf_daily = None
    if os.path.exists(CAMINHO_RF):
        try:
            rf_daily = load_risk_free(CAMINHO_RF, dayfirst=True)
        except Exception:
            rf_daily = None

    # 3) Classes + nomes amigáveis
    df_classes_raw = load_excel_cached(CAMINHO_CLASSIFICACAO)
    df_classes_raw.columns = df_classes_raw.columns.astype(str).str.strip()
    for col_need in ["Ativo", "Classe"]:
        if col_need not in df_classes_raw.columns:
            raise ValueError(f"A planilha de classificação precisa das colunas '{col_need}'.")
    classe_ativos = dict(zip(df_classes_raw["Ativo"], df_classes_raw["Classe"]))
    if "Nome" not in df_classes_raw.columns: df_classes_raw["Nome"] = None
    if "NomeCurto" not in df_classes_raw.columns: df_classes_raw["NomeCurto"] = None
    DISPLAY_COL, SHORT_COL = "Nome", "NomeCurto"
    name_map_long  = df_classes_raw.set_index("Ativo")[DISPLAY_COL].to_dict()
    name_map_short = df_classes_raw.set_index("Ativo")[SHORT_COL].to_dict()

    def get_name(ticker: str, prefer_short: bool = True) -> str:
        t = str(ticker).strip()
        nm_short = (name_map_short.get(t) or "").strip()
        if nm_short:
            return nm_short
        nm_long = (name_map_long.get(t) or "").strip()
        return nm_long if nm_long else t

    # 4) Benchmarks (séries + nomes)
    bench_series = {}
    bench_name_map_long = {}
    bench_name_map_short = {}
    if os.path.exists(CAMINHO_BENCHMARK):
        try:
            bench_raw = load_excel_cached(CAMINHO_BENCHMARK)
            bench = bench_raw.copy()
            bench.columns = bench.columns.astype(str).str.strip()
            if "Date" not in bench.columns and bench.shape[1] > 0:
                bench.rename(columns={bench.columns[0]: "Date"}, inplace=True)
            bench["Date"] = pd.to_datetime(bench["Date"], dayfirst=True, errors="coerce")
            bench = bench.dropna(subset=["Date"]).set_index("Date").sort_index().ffill()
            for c in bench.columns:
                if c.lower() != "date" and pd.api.types.is_numeric_dtype(bench[c]):
                    bench_series[c] = bench[c].pct_change().dropna()
        except Exception:
            pass

        try:
            xls = pd.ExcelFile(CAMINHO_BENCHMARK)
            if "bench_nomes" in xls.sheet_names:
                bn = pd.read_excel(CAMINHO_BENCHMARK, sheet_name="bench_nomes")
                bn.columns = bn.columns.astype(str).str.strip()
                if "Ticker" in bn.columns:
                    if "Nome" not in bn.columns: bn["Nome"] = None
                    if "NomeCurto" not in bn.columns: bn["NomeCurto"] = None
                    bench_name_map_long  = bn.set_index("Ticker")["Nome"].to_dict()
                    bench_name_map_short = bn.set_index("Ticker")["NomeCurto"].to_dict()
        except Exception:
            pass

    def get_bench_name(ticker: str, prefer_short: bool = True) -> str:
        t = str(ticker).strip()
        nm_short = (bench_name_map_short.get(t) or "").strip()
        if nm_short:
            return nm_short
        nm_long = (bench_name_map_long.get(t) or "").strip()
        return nm_long if nm_long else t

    # ------------------- ABAS -------------------
    tab_sel, tab_otm, tab_manual, tab_comp, tab_corr, tab_catalog = st.tabs(
        ["Seleção", "Otimização", "Pesos manuais", "Comparar Ativos", "Correlação Macro", "Catálogo (Nomes & Classes)"]
    )

    # --------- Seleção por classe ----------
    with tab_sel:
        st.subheader("Selecione os ativos por classe")
        if "ativos_selecionados" not in st.session_state:
            st.session_state["ativos_selecionados"] = list(df.columns)

        total_disponiveis = list(df.columns)
        todos_marcados = len(st.session_state["ativos_selecionados"]) == len(total_disponiveis)
        if st.button("Selecionar todos" if not todos_marcados else "Desmarcar todos", key="toggle_sel_all"):
            st.session_state["ativos_selecionados"] = [] if todos_marcados else total_disponiveis

        grupos = {}
        for t in df.columns:
            grupos.setdefault(classe_ativos.get(t, "Outros"), []).append(t)

        classes_ordenadas = [c for c in CLASS_ORDER if c in grupos] + [c for c in grupos if c not in CLASS_ORDER]
        colunas = st.columns(len(classes_ordenadas) if classes_ordenadas else 1)
        novos = set(st.session_state["ativos_selecionados"])
        for col, classe in zip(colunas, classes_ordenadas):
            with col:
                st.markdown(f"*{classe}*")
                for ativo in sorted(grupos[classe]):
                    ck = st.checkbox(get_name(ativo), value=(ativo in novos), key=f"sel_{ativo}")
                    if ck:  novos.add(ativo)
                    else:   novos.discard(ativo)
        st.session_state["ativos_selecionados"] = sorted(novos)
        st.info(f"Ativos selecionados: {len(st.session_state['ativos_selecionados'])}")

    # aplica seleção
    ativos_ok = st.session_state.get("ativos_selecionados", list(df.columns)) or list(df.columns)
    df_sel = df[ativos_ok].copy()
    returns_sel = returns[ativos_ok].copy()
    if df_sel.empty or returns_sel.empty:
        st.error("❌ Nenhum ativo válido foi encontrado. Verifique a seleção/planilha.")
        st.stop()

    # 4) Estatísticas
    mean_returns, cov_matrix = compute_stats(returns_sel)

    # 5)–8) OTIMIZAÇÃO — roda no botão
    cache_key = make_cache_key(df_sel.columns, perfil, criterio, retorno_alvo, max_dd_user)
    if do_calc:
        try:
            pesos_otimizados, dd_info = otimizar_portfolio(
                criterio, perfil, retorno_alvo, max_dd_user,
                tuple(df_sel.columns), mean_returns, cov_matrix,
                limites_demo, classe_ativos, returns_sel, rf_daily
            )
            try:
                w_gmvp = gmvp_weights(mean_returns, cov_matrix, df_sel.columns, perfil, limites_demo, classe_ativos)
                w_gmvp = enforce_min1(w_gmvp, df_sel.columns, perfil, mean_returns, cov_matrix,
                                      limites_demo, classe_ativos, mode="gmvp", rf_daily=rf_daily)
            except Exception:
                w_gmvp = None
            try:
                w_maxsh = max_sharpe_weights(mean_returns, cov_matrix, df_sel.columns, perfil, limites_demo, classe_ativos, rf_daily=rf_daily)
                w_maxsh = enforce_min1(w_maxsh, df_sel.columns, perfil, mean_returns, cov_matrix,
                                       limites_demo, classe_ativos, mode="max_sharpe", rf_daily=rf_daily)
            except Exception:
                w_maxsh = None
            frontier_rets, frontier_vols = gerar_fronteira_eficiente(mean_returns, cov_matrix,
                                                                     tuple(df_sel.columns), perfil, limites_demo, classe_ativos)
            st.session_state["otm_cache"][cache_key] = {
                "pesos_otimizados": pesos_otimizados, "dd_info": dd_info,
                "w_gmvp": w_gmvp, "w_maxsh": w_maxsh,
                "frontier_rets": frontier_rets, "frontier_vols": frontier_vols,
            }
            st.session_state["last_otm_key"] = cache_key
            st.success("✅ Otimização concluída.")
        except Exception as e:
            st.error(str(e))

    # 7) OTIMIZAÇÃO (visualização)
    with tab_otm:
        bench_options = {"(Sem benchmark)": None}
        for k, s in bench_series.items():
            bench_options[get_bench_name(k)] = s
        if "bench_sel" not in st.session_state:
            st.session_state["bench_sel"] = "(Sem benchmark)"
        bench_escolhido_label = st.selectbox("Benchmark de comparação",
            list(bench_options.keys()),
            index=list(bench_options.keys()).index(st.session_state["bench_sel"])
        )
        st.session_state["bench_sel"] = bench_escolhido_label
        benchmark_retornos = bench_options[bench_escolhido_label]
        benchmark_nome = bench_escolhido_label if bench_escolhido_label != "(Sem benchmark)" else None

        show_key = st.session_state.get("last_otm_key", None)
        if show_key is None or show_key not in st.session_state["otm_cache"]:
            st.info("⚙️ Configure os parâmetros no sidebar e clique em **Calcular otimização** para ver os resultados.")
        else:
            hit = st.session_state["otm_cache"][show_key]
            pesos_otimizados = hit["pesos_otimizados"]; dd_info = hit["dd_info"]
            w_gmvp = hit["w_gmvp"]; w_maxsh = hit["w_maxsh"]
            frontier_rets = hit["frontier_rets"]; frontier_vols = hit["frontier_vols"]

            extras = []
            if w_gmvp is not None: extras.append(("GMVP", w_gmvp, None, "D"))
            if w_maxsh is not None: extras.append(("Máx. Sharpe", w_maxsh, None, "^"))

            subt = f"📊 Alocação Ótima — Perfil {perfil} (Retorno alvo: {retorno_alvo*100:.1f}%)"
            if criterio == "Retorno alvo + Máx. DD" and (max_dd_user is not None):
                subt += f" • Máx. DD: {max_dd_user*100:.1f}%"

            render_dashboard(
                pesos=pesos_otimizados, titulo=subt,
                returns=returns_sel, mean_returns=mean_returns, cov_matrix=cov_matrix, df=df_sel,
                classe_ativos=classe_ativos, benchmark_retornos=benchmark_retornos, benchmark_nome=benchmark_nome,
                mostrar_fronteira=True, frontier_vols=frontier_vols, frontier_returns=frontier_rets,
                rf_daily=rf_daily, extra_points=extras, get_name=get_name
            )

            if dd_info is not None:
                modo, dd_val = dd_info
                st.success(f"Máx. Drawdown: {abs(dd_val)*100:.2f}% (dentro do alvo).") if modo=="ok" else \
                st.warning(f"⚠️ Máx. Drawdown: {abs(dd_val)*100:.2f}% — ACIMA do limite.")

            def metrics(s):
                r = (1 + s.mean())**252 - 1
                v = s.std() * np.sqrt(252)
                sh = calc_sharpe(s, rf_daily)
                cum = (1 + s).cumprod(); peak = cum.cummax()
                dd = (cum - peak)/peak; mdd = dd.min()
                return r, v, sh, mdd

            linhas = []
            port_opt = (returns_sel * pesos_otimizados).sum(axis=1)
            r_o, v_o, s_o, dd_o = metrics(port_opt)
            linhas.append(["Otimizada", f"{r_o*100:.2f}%", f"{v_o*100:.2f}%", f"{s_o:.2f}", f"{dd_o*100:.2f}%"])
            if w_gmvp is not None:
                r_g, v_g, s_g, dd_g = metrics((returns_sel * w_gmvp).sum(axis=1))
                linhas.append(["GMVP", f"{r_g*100:.2f}%", f"{v_g*100:.2f}%", f"{s_g:.2f}", f"{dd_g*100:.2f}%"])
            if w_maxsh is not None:
                r_m, v_m, s_m, dd_m = metrics((returns_sel * w_maxsh).sum(axis=1))
                linhas.append(["Máx. Sharpe", f"{r_m*100:.2f}%", f"{v_m*100:.2f}%", f"{s_m:.2f}", f"{dd_m*100:.2f}%"])
            if benchmark_retornos is not None and not getattr(benchmark_retornos, "empty", True):
                _, bench_alinh = port_opt.align(benchmark_retornos, join="inner")
                r_b, v_b, s_b, dd_b = metrics(bench_alinh)
                linhas.append([f"{benchmark_nome}", f"{r_b*100:.2f}%", f"{v_b*100:.2f}%", f"{s_b:.2f}", f"{dd_b*100:.2f}%"])
            st.subheader("📑 Comparação de Métricas")
            st.dataframe(pd.DataFrame(linhas, columns=["Carteira", "Retorno (a.a.)", "Vol. (a.a.)", "Sharpe", "Máx. DD"]),
                         use_container_width=True)

            if w_maxsh is not None:
                st.subheader("🟢 Pesos — Carteira de Máximo Sharpe")
                cols_ord = sort_columns_by_class(df_sel.columns, classe_ativos)
                w_series = pd.Series(w_maxsh, index=df_sel.columns).reindex(cols_ord).fillna(0.0)
                labels_ms = dedup_labels([get_name(t) for t in w_series.index], list(w_series.index))
                df_pesos_ms = pd.DataFrame({"Ativo": labels_ms,
                                            "Classe": [classe_ativos.get(t, "Outros") for t in w_series.index],
                                            "Peso (%)": (w_series.values * 100).round(2)})
                st.dataframe(df_pesos_ms[df_pesos_ms["Peso (%)"] > 0], use_container_width=True)
                st.subheader("📆 Indicadores por Ano — Máximo Sharpe")
                st.dataframe(indicadores_por_ano((returns_sel * w_maxsh).sum(axis=1), rf_daily), use_container_width=True)

    # 10) PESOS MANUAIS
    with tab_manual:
        bench_options = {"(Sem benchmark)": None}
        for k, s in bench_series.items():
            bench_options[get_bench_name(k)] = s
        bench_escolhido_label = st.selectbox("Benchmark de comparação", list(bench_options.keys()),
                                             index=list(bench_options.keys()).index(st.session_state.get("bench_sel", "(Sem benchmark)")),
                                             key="bench_sel_manual")
        st.session_state["bench_sel"] = bench_escolhido_label
        benchmark_retornos = bench_options[bench_escolhido_label]
        benchmark_nome = bench_escolhido_label if bench_escolhido_label != "(Sem benchmark)" else None

        st.subheader("✍️ Pesos manuais (em %)")
        cols_ord = sort_columns_by_class(df_sel.columns, classe_ativos)
        labels_amig = dedup_labels([get_name(t) for t in cols_ord], cols_ord)
        df_pesos = pd.DataFrame({"Ativo": labels_amig,
                                 "Classe": [classe_ativos.get(t, "Outros") for t in cols_ord],
                                 "Peso (%)": np.zeros(len(cols_ord))})
        if st.checkbox("Iniciar com pesos iguais"):
            df_pesos["Peso (%)"] = np.round(100 / len(cols_ord), 2)

        edited = st.data_editor(df_pesos, num_rows="fixed", use_container_width=True,
                                column_config={"Peso (%)": st.column_config.NumberColumn(step=0.1, min_value=0.0, max_value=100.0),
                                               "Classe": st.column_config.TextColumn(disabled=True)})
        soma = float(edited["Peso (%)"].sum()); st.write(f"*Soma atual:* {soma:.2f}%")
        normalizar = st.checkbox("Normalizar automaticamente para 100%", value=True)

        if st.button("Aplicar pesos manuais"):
            try:
                map_label_to_ticker = {lab: tick for lab, tick in zip(labels_amig, cols_ord)}
                vals = edited["Peso (%)"].astype(float).values / 100.0
                labs = list(edited["Ativo"])
                ticks = [map_label_to_ticker.get(l, l) for l in labs]
                w_series = pd.Series(vals, index=ticks).reindex(df_sel.columns).fillna(0.0)
                pesos_man = w_series.values
            except Exception as e:
                st.error(f"Erro ao ler pesos manuais: {e}")
                pesos_man = np.ones(len(df_sel.columns)) / len(df_sel.columns)

            if normalizar and pesos_man.sum() > 0:
                pesos_man = pesos_man / pesos_man.sum()
            elif not np.isclose(pesos_man.sum(), 1.0):
                st.error("A soma dos pesos precisa ser 100% (ou marque Normalizar).")
                st.stop()

            st.subheader("📊 Carteira com Pesos Manuais")
            cols_ord2 = sort_columns_by_class(df_sel.columns, classe_ativos)
            pesos_map2 = pd.Series(pesos_man, index=df_sel.columns)
            pesos_ord2 = pesos_map2.reindex(cols_ord2).fillna(0.0).values
            labels2 = dedup_labels([get_name(t) for t in cols_ord2], cols_ord2)
            result_df2 = pd.DataFrame({"Ativo": labels2,
                                       "Classe": [classe_ativos.get(t, "Outros") for t in cols_ord2],
                                       "Peso (%)": np.round(pesos_ord2 * 100, 2)})
            st.dataframe(result_df2[result_df2["Peso (%)"] > 0], use_container_width=True)

            portfolio_returns2 = (returns_sel * pesos_man).sum(axis=1)
            fig2, ax2 = plt.subplots(figsize=(5, 3.2))
            ((1 + portfolio_returns2).cumprod() - 1).plot(ax=ax2, label="Carteira", lw=2)
            if benchmark_retornos is not None and not getattr(benchmark_retornos, "empty", True):
                port_alinh, bench_alinh = portfolio_returns2.align(benchmark_retornos, join="inner")
                ((1 + bench_alinh).cumprod() - 1).plot(ax=ax2, ls="--", lw=1.5, label=(benchmark_nome or "Benchmark"))
            percent_axis(ax2, 'y', 'unit')
            ax2.set_title("Evolução do Portfólio (acumulado, %)")
            ax2.legend(); ax2.grid(alpha=0.25, linewidth=0.6)
            st.pyplot(fig2)

            ann_return, ann_vol, sharpe, max_dd = indicadores_totais(portfolio_returns2, rf_daily)
            st.table({"Retorno Anualizado": f"{ann_return*100:.2f}%",
                      "Volatilidade Anualizada": f"{ann_vol*100:.2f}%",
                      "Sharpe": f"{sharpe:.2f}",
                      "Máximo Drawdown": f"{max_dd*100:.2f}%"})

            if benchmark_retornos is not None and not getattr(benchmark_retornos, "empty", True):
                st.subheader("📊 Comparação com Benchmark (Pesos Manuais)")
                port_alinh, bench_alinh = portfolio_returns2.align(benchmark_retornos, join="inner")
                def _metrics_cmp(s):
                    r = (1 + s.mean())**252 - 1
                    v = s.std() * (252**0.5)
                    sh = calc_sharpe(s, rf_daily)
                    cum = (1 + s).cumprod(); peak = cum.cummax(); dd = (cum - peak)/peak; mdd = dd.min()
                    return r, v, sh, mdd
                r_pm, v_pm, sh_pm, dd_pm = _metrics_cmp(port_alinh)
                r_bm, v_bm, sh_bm, dd_bm = _metrics_cmp(bench_alinh)
                comp_manual = pd.DataFrame({"": ["Carteira", "Benchmark"],
                                            "Retorno (a.a.)": [f"{r_pm*100:.2f}%", f"{r_bm*100:.2f}%"],
                                            "Vol. (a.a.)":    [f"{v_pm*100:.2f}%", f"{v_bm*100:.2f}%"],
                                            "Sharpe":         [f"{sh_pm:.2f}", f"{sh_bm:.2f}"],
                                            "Máx DD":         [f"{dd_pm*100:.2f}%", f"{dd_bm*100:.2f}%"]})
                st.dataframe(comp_manual, use_container_width=True)

                # -------- Indicadores por ano — Carteira e Benchmark (empilhados) --------
                st.subheader("📆 Indicadores por Ano — Carteira (Pesos Manuais)")
                tab_port_ano = indicadores_por_ano(portfolio_returns2, rf_daily)
                st.dataframe(tab_port_ano, use_container_width=True)

                st.subheader(f"📆 Indicadores por Ano — {benchmark_nome or 'Benchmark'}")
                port_alinh_y, bench_alinh_y = portfolio_returns2.align(benchmark_retornos, join="inner")
                tab_bench_ano = indicadores_por_ano(bench_alinh_y, rf_daily)
                st.dataframe(tab_bench_ano, use_container_width=True)

                # (Opcional) Gráfico anual Carteira vs Benchmark
                st.markdown("**Gráfico anual — Carteira vs Benchmark (opcional)**")
                show_annual_chart = st.checkbox("Mostrar gráfico anual (Carteira vs Benchmark)",
                                                key="manual_show_annual_chart", value=False)
                if show_annual_chart:
                    df_y = pd.DataFrame({"Carteira": port_alinh_y, "Benchmark": bench_alinh_y}).dropna()
                    if not df_y.empty:
                        df_y["Ano"] = df_y.index.year
                        gy = df_y.groupby("Ano").apply(
                            lambda g: pd.Series({
                                "Carteira":  (1 + g["Carteira"]).prod()  - 1.0,
                                "Benchmark": (1 + g["Benchmark"]).prod() - 1.0,
                            })
                        )
                        anos = gy.index.astype(str)
                        pos  = np.arange(len(anos))
                        w    = 0.45
                        figy, axy = plt.subplots(figsize=(6.8, 3.2))
                        axy.bar(pos - w/2, gy["Carteira"].values, width=w, label="Carteira")
                        axy.bar(pos + w/2, gy["Benchmark"].values, width=w, label=(benchmark_nome or "Benchmark"))
                        axy.set_xticks(pos); axy.set_xticklabels(anos)
                        percent_axis(axy, 'y', 'unit')
                        axy.set_ylabel("Retorno (%)")
                        axy.grid(axis="y", alpha=0.25, linewidth=0.6)
                        axy.legend(loc="best", frameon=False)
                        st.pyplot(figy)

    # 11) COMPARAR ATIVOS
    with tab_comp:
        st.subheader("Comparar Ativos")
        LIMITE_ATIVOS = 6
        if "ativos_comp" not in st.session_state:
            st.session_state["ativos_comp"] = list(df_sel.columns[:min(LIMITE_ATIVOS, len(df_sel.columns))])

        total_disponiveis_comp = list(df_sel.columns)
        todos_marcados_comp = len(st.session_state["ativos_comp"]) == len(total_disponiveis_comp)
        if st.button("Selecionar todos" if not todos_marcados_comp else "Desmarcar todos", key="toggle_comp_all"):
            if todos_marcados_comp:
                st.session_state["ativos_comp"] = []
            else:
                st.session_state["ativos_comp"] = total_disponiveis_comp[:LIMITE_ATIVOS]
                if len(total_disponiveis_comp) > LIMITE_ATIVOS:
                    st.info(f"Selecionados os {LIMITE_ATIVOS} primeiros ativos (limite atingido).")

        grupos = {}
        for t in df_sel.columns:
            grupos.setdefault(classe_ativos.get(t, "Outros"), []).append(t)
        classes_ordenadas = [c for c in CLASS_ORDER if c in grupos] + [c for c in grupos if c not in CLASS_ORDER]
        cols = st.columns(len(classes_ordenadas) if classes_ordenadas else 1)
        escolhidos = []
        base_inicial = set(st.session_state["ativos_comp"])
        for col, classe in zip(cols, classes_ordenadas):
            with col:
                st.markdown(f"*{classe}*")
                for ativo in sorted(grupos[classe]):
                    ck = st.checkbox(get_name(ativo), value=(ativo in base_inicial), key=f"comp_{ativo}")
                    if ck:
                        escolhidos.append(ativo)
        if len(escolhidos) > LIMITE_ATIVOS:
            escolhidos = escolhidos[:LIMITE_ATIVOS]
            st.warning(f"Limite de {LIMITE_ATIVOS} ativos para comparação. Apenas os primeiros foram mantidos.")
        st.session_state["ativos_comp"] = escolhidos
        ativos_comp = st.session_state["ativos_comp"]

        if ativos_comp:
            ret_comp = returns_sel[ativos_comp].dropna(how="all")
            if ret_comp.empty:
                st.warning("Não há dados válidos para os ativos selecionados.")
            else:
                st.markdown("**Evolução acumulada (%, base = 0%)**")
                cum = (1 + ret_comp).cumprod() - 1
                figc, axc = plt.subplots(figsize=(3.6, 1.4)); figc.set_dpi(180)
                for c in cum.columns:
                    axc.plot(cum.index, cum[c], lw=1.0, label=get_name(c))
                locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
                axc.xaxis.set_major_locator(locator)
                axc.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                percent_axis(axc, 'y', 'unit')
                axc.tick_params(axis="x", labelsize=8); axc.tick_params(axis="y", labelsize=8)
                axc.margins(x=0); axc.grid(alpha=0.25, linewidth=0.5)
                axc.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False, ncol=1, handlelength=1.6)
                axc.set_xlabel(""); axc.set_ylabel(""); st.pyplot(figc)
                linhas = []
                for c in ativos_comp:
                    s = ret_comp[c].dropna()
                    if s.empty: continue
                    r = (1 + s.mean())**252 - 1
                    v = s.std() * np.sqrt(252)
                    sh = calc_sharpe(s, rf_daily)
                    dd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()
                    linhas.append([get_name(c), classe_ativos.get(c, "Outros"),
                                   f"{r*100:.2f}%", f"{v*100:.2f}%", f"{sh:.2f}", f"{dd*100:.2f}%"])
                st.dataframe(pd.DataFrame(linhas, columns=["Ativo", "Classe", "Retorno (a.a.)", "Vol. (a.a.)", "Sharpe", "Máx. DD"]),
                             use_container_width=True)
        else:
            st.warning("Selecione ao menos um ativo para comparar.")

    # 12) CORRELAÇÃO MACRO — variável + scatter + REGIMES/BALDES
    with tab_corr:
        st.subheader("🔗 Relação ativos × variável macro (mensal) + Regimes/Baldes")
        macro_df = load_macro_flex(CAMINHO_MACRO)

        if macro_df is None:
            st.info("Crie **macro.xlsx** com Date + colunas de Fed/CPI/PCE/DXY (nomes flexíveis).")
        else:
            infl_pref = None; infl_label = "Inflação YoY"
            if "cpi_core_yoy" in macro_df.columns: infl_pref, infl_label = "cpi_core_yoy", "Core CPI YoY (pp)"
            elif "pce_core_yoy" in macro_df.columns: infl_pref, infl_label = "pce_core_yoy", "Core PCE YoY (pp)"
            elif "cpi_yoy" in macro_df.columns: infl_pref, infl_label = "cpi_yoy", "CPI YoY (pp)"
            elif "pce_yoy" in macro_df.columns: infl_pref, infl_label = "pce_yoy", "PCE YoY (pp)"

            macro_opts = {}
            if "dxy" in macro_df.columns:
                macro_opts["DXY (var. % mensal)"] = ("dxy", "pct")
            if infl_pref in ["cpi_core_yoy", "cpi_yoy"]:
                macro_opts["Core CPI YoY (nível, pp)" if infl_pref=="cpi_core_yoy" else "CPI YoY (nível, pp)"] = (infl_pref, "level")
            if "fedfunds" in macro_df.columns:
                if infl_pref is not None:
                    macro_df["fed_real"] = macro_df["fedfunds"] - macro_df[infl_pref]
                    macro_opts[f"Fed Funds real = Fed - {infl_label}"] = ("fed_real", "level")
                macro_opts["Fed Funds (nível, pp)"] = ("fedfunds", "level")

            if not macro_opts:
                st.warning("Nenhuma série macro reconhecida em macro.xlsx.")
            else:
                var_label = st.selectbox("Variável macro", list(macro_opts.keys()))
                var_col, var_kind = macro_opts[var_label]

                ret_m = (1 + returns_sel).resample("M").prod() - 1
                m_raw = macro_df[var_col].resample("M").last()
                x_series = m_raw.pct_change() if var_kind == "pct" else m_raw
                x_name = var_label
                base = ret_m.join(x_series.rename("X"), how="inner").dropna(how="any")

                if base.empty:
                    st.warning("Sem interseção de datas entre ativos e a variável macro selecionada.")
                else:
                    all_asset_labels = [get_name(t) for t in ret_m.columns]
                    label_to_ticker  = {get_name(t): t for t in ret_m.columns}
                    chosen_assets = st.multiselect("Ativos para plotar", all_asset_labels,
                                                   all_asset_labels[:min(6, len(all_asset_labels))])

                    if chosen_assets:
                        # Sliders de faixa + formatação em %
                        if var_kind == "level":
                            if "CPI" in x_name:
                                xmin, xmax = st.slider("Faixa do eixo X (CPI YoY, %)",
                                                       min_value=0.0, max_value=max(10.0, float(np.ceil(base["X"].max()))),
                                                       value=(0.0, 10.0), step=0.1)
                                mode_x = 'pp'
                            else:
                                x_min_obs = float(np.floor(base["X"].min()))
                                x_max_obs = float(np.ceil(base["X"].max()))
                                xmin, xmax = st.slider("Faixa do eixo X (nível, %)",
                                                       min_value=min(0.0, x_min_obs), max_value=max(6.0, x_max_obs),
                                                       value=(min(0.0, x_min_obs), max(6.0, x_max_obs)), step=0.1)
                                mode_x = 'pp'
                        else:
                            q5, q95 = np.percentile(base["X"].values, [5, 95])
                            pad = float(max(abs(q5), abs(q95)))
                            lim = float(np.ceil((pad + 0.01) * 100) / 100)
                            xmin, xmax = st.slider("Faixa do eixo X (DXY, var.% m/m)",
                                                   min_value=-lim, max_value=lim, value=(-lim, lim), step=0.001)
                            mode_x = 'unit'

                        show_ols = st.checkbox("Mostrar linha de tendência (OLS)", value=False)
                        show_corr_table = st.checkbox("Mostrar tabela de correlação (ativo vs variável)", value=True)

                        fig, ax = plt.subplots(figsize=(6.8, 4.2))
                        for lab in chosen_assets:
                            tkr = label_to_ticker[lab]
                            df_sc = base[[tkr, "X"]].dropna()
                            if df_sc.empty: continue
                            ax.scatter(df_sc["X"].values, df_sc[tkr].values, s=18, alpha=0.85, label=lab)
                            if show_ols and len(df_sc) >= 3:
                                try:
                                    b1, b0 = np.polyfit(df_sc["X"].values, df_sc[tkr].values, 1)
                                    xs = np.linspace(xmin, xmax, 80)
                                    ax.plot(xs, b1*xs + b0, linewidth=1.0)
                                except Exception:
                                    pass
                        ax.set_xlim(xmin, xmax)
                        percent_axis(ax, 'x', mode_x)          # X em %
                        percent_axis(ax, 'y', 'unit')          # retornos mensais em %
                        ax.set_xlabel(f"{x_name}")
                        ax.set_ylabel("Retorno mensal dos ativos (%)")
                        ax.axhline(0, lw=0.8, alpha=0.5)
                        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
                        ax.grid(alpha=0.25, linewidth=0.6)
                        st.pyplot(fig)

                        if show_corr_table:
                            rows = []
                            for lab in chosen_assets:
                                tkr = label_to_ticker[lab]
                                s = base[[tkr, "X"]].dropna()
                                if not s.empty:
                                    rows.append([lab, round(s[tkr].corr(s["X"]), 3), len(s)])
                            if rows:
                                st.dataframe(pd.DataFrame(rows, columns=["Ativo", "Correlação", "N"]), use_container_width=True)
                    else:
                        st.info("Selecione ao menos um ativo para plotar.")

                    # ====== REGIMES / BALDES ======
                    st.markdown("---")
                    st.subheader("🎛️ Regimes/Baldes da variável macro")
                    colm1, colm2 = st.columns([1,1])
                    with colm1:
                        bucket_method = st.radio("Método de balde", ["Faixas fixas", "Quantis"], index=0)
                    with colm2:
                        metric_choice = st.selectbox("Métrica na tabela",
                                                     ["Média (%)", "Vol (%)", "Hit Rate (%)", "Sharpe (m)", "Sharpe (a)"], index=0)

                    def default_edges():
                        if var_kind == "pct":
                            return [-1.0, -0.02, -0.01, 0.0, 0.01, 0.02, 1.0]  # fração (-100% a 100%)
                        else:
                            if "CPI" in x_name: return [0.0, 2.5, 3.5, 4.5, 6.0, 99.9]  # já em %
                            return [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 99.9]               # já em %

                    # Regimes/Baldes — rótulos do X em %
                    if bucket_method == "Faixas fixas":
                        edges_str = st.text_input("Cortes (crescentes, vírgulas)",
                                                  value=",".join(str(e) for e in default_edges()))
                        try:
                            edges = [float(x.strip()) for x in edges_str.split(",") if x.strip() != ""]
                            edges = sorted(edges)
                        except Exception:
                            st.error("Cortes inválidos. Usando padrão."); edges = default_edges()
                        if len(edges) < 3:
                            st.warning("Poucos cortes; adicione mais pontos para formar faixas.")

                        # Labels com % (fração -> %, nível -> %)
                        if var_kind == "pct":
                            labels = [make_bucket_label(edges[i], edges[i+1], 'pct') for i in range(len(edges)-1)]
                        else:
                            labels = [make_bucket_label(edges[i], edges[i+1], 'pp') for i in range(len(edges)-1)]

                        Xb = pd.cut(base["X"], bins=edges, labels=labels, include_lowest=True, right=True)
                        order_cols = labels[:]  # mantém a ordem dos rótulos
                    else:
                        q = st.slider("Número de quantis", min_value=3, max_value=10, value=4, step=1)
                        try:
                            Xb = pd.qcut(base["X"], q=q, duplicates="drop")
                        except Exception:
                            st.warning("Quantis indisponíveis (dados repetidos/insuficientes). Ajustando…")
                            Xb = pd.qcut(base["X"].rank(method="first"), q=q, duplicates="drop")

                        # Renomeia categorias para % (ex.: (0.5%, 1.2%])
                        if pd.api.types.is_categorical_dtype(Xb):
                            cats = list(Xb.cat.categories)
                            new_labels = [make_bucket_label(iv.left, iv.right, 'pct' if var_kind=='pct' else 'pp') for iv in cats]
                            Xb = Xb.cat.rename_categories(new_labels)
                            order_cols = new_labels[:]
                        else:
                            order_cols = [str(c) for c in sorted(pd.unique(Xb))]

                    # Base + padronização de coluna de balde
                    base_b = base.copy()
                    base_b["__bucket__"] = Xb
                    if "bucket" in base_b.columns:
                        mask_nan = base_b["__bucket__"].isna()
                        if mask_nan.any():
                            base_b.loc[mask_nan, "__bucket__"] = base_b.loc[mask_nan, "bucket"]
                        base_b.drop(columns=["bucket"], inplace=True, errors="ignore")

                    base_b = base_b.dropna(subset=["__bucket__"])
                    if base_b.empty:
                        st.warning("Sem observações após a aplicação dos baldes.")
                    else:
                        bucket_col = "__bucket__"
                        asset_cols = list(ret_m.columns)

                        def _metrics_by_bucket(df, asset_cols, bucket_col):
                            out_rows = []
                            cols_needed = set(asset_cols) | {bucket_col}
                            gg = df[[c for c in cols_needed if c in df.columns]].dropna(subset=[bucket_col])
                            if gg.empty: return pd.DataFrame()
                            for a in asset_cols:
                                if a not in gg.columns: continue
                                grp = gg.groupby(bucket_col)[a]
                                mean = grp.mean()
                                vol  = grp.std()
                                hit  = grp.apply(lambda s: (s > 0).mean())
                                N    = grp.size()
                                sh_m = mean / vol.replace(0, np.nan)
                                sh_a = sh_m * np.sqrt(12)
                                tmp = pd.DataFrame({"mean": mean, "vol": vol, "hit": hit, "N": N, "sh_m": sh_m, "sh_a": sh_a})
                                tmp["asset"] = a
                                out_rows.append(tmp.reset_index())
                            return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()

                        metr = _metrics_by_bucket(base_b, asset_cols, bucket_col=bucket_col)
                        if metr.empty:
                            st.info("Sem métricas calculadas (verifique baldes).")
                        else:
                            metr["asset_label"] = metr["asset"].map(lambda t: get_name(t))
                            metric_map = {
                                "Média (%)": ("mean", 100.0),
                                "Vol (%)": ("vol", 100.0),
                                "Hit Rate (%)": ("hit", 100.0),
                                "Sharpe (m)": ("sh_m", 1.0),
                                "Sharpe (a)": ("sh_a", 1.0),
                            }
                            col_m, scale = metric_map[metric_choice]
                            metr["_val"] = metr[col_m] * scale

                            piv = metr.pivot_table(index="asset_label", columns=bucket_col, values="_val", aggfunc="first")
                            # Garante ordem dos buckets conforme definimos
                            piv = piv.reindex(columns=order_cols)
                            st.subheader("Tabela por baldes")
                            st.dataframe(piv.round(2), use_container_width=True)

                            csv = piv.round(4).to_csv().encode("utf-8")
                            st.download_button("⬇️ Baixar CSV da tabela", data=csv, file_name="regimes_pivot.csv", mime="text/csv")

                            st.markdown("**Barras por balde (métrica escolhida)**")
                            one_asset = st.selectbox("Ativo", sorted(metr["asset_label"].unique()))
                            sub = metr.loc[metr["asset_label"] == one_asset].copy()

                            # Ordena pelo mesmo order_cols
                            sub = sub.set_index(bucket_col)
                            sub.index = sub.index.astype(str)
                            sub = sub.reindex(order_cols).reset_index()

                            figb, axb = plt.subplots(figsize=(6.8, 3.6))
                            axb.bar(sub[bucket_col].astype(str), sub["_val"].values)
                            axb.set_xlabel("Balde/regime")
                            axb.set_ylabel(metric_choice)
                            if metric_choice in ["Média (%)", "Vol (%)", "Hit Rate (%)"]:
                                percent_axis(axb, 'y', 'pp')  # valores estão em 0..100
                            axb.grid(alpha=0.25, linewidth=0.6, axis="y")
                            plt.setp(axb.get_xticklabels(), rotation=0, ha="center")
                            st.pyplot(figb)

    # 13) CATÁLOGO — Ticker, Classe e NomeCurto (ativos) + benchmarks
    with tab_catalog:
        st.subheader("📚 Catálogo de Ativos — Ticker, Classe e NomeCurto")
        rows = [{"Ticker": t, "Classe": classe_ativos.get(t, "Outros"),
                 "NomeCurto": (name_map_short.get(t) or "").strip()} for t in sort_columns_by_class(list(df.columns), classe_ativos)]
        df_cat = pd.DataFrame(rows, columns=["Ticker", "Classe", "NomeCurto"])
        if (df_cat["NomeCurto"].eq("") | df_cat["NomeCurto"].isna()).any():
            st.warning("Alguns ativos estão sem NomeCurto na planilha de classificação.")
        st.dataframe(df_cat, use_container_width=True)

        st.subheader("🏁 Catálogo de Benchmarks — Ticker e Nomes")
        if bench_series:
            rows_b = [{"Ticker": bt, "NomeCurto": (bench_name_map_short.get(bt) or "").strip(),
                       "Nome": (bench_name_map_long.get(bt) or "").strip()} for bt in sorted(bench_series.keys())]
            st.dataframe(pd.DataFrame(rows_b, columns=["Ticker", "NomeCurto", "Nome"]), use_container_width=True)
        else:
            st.info("Nenhum benchmark carregado. Verifique 'benchmark.xlsx'.")

except Exception as e:
    st.error(f"❌ Erro ao processar: {e}")

# Aviso se benchmark vazio
if 'benchmark_retornos' in locals() and benchmark_retornos is not None and getattr(benchmark_retornos, "empty", True):
    st.warning("⚠️ O benchmark selecionado não possui dados válidos.")
    benchmark_retornos = None
