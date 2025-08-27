import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize

# =================== CONFIG ===================
st.set_page_config(page_title="Fortune Financial Strategies - Asset Allocation", layout="wide")
st.title("Fortune Financial Strategies - Asset Allocation")
st.caption("Build: v16.5 — by Pedro Freitas de Amorim")

# >>> Tolerância numérica para checagem de DD <<<
EPS = 1e-6

# ====== CACHE DE OTIMIZAÇÃO EM SESSION_STATE (não depende do benchmark) ======
if "otm_cache" not in st.session_state:
    st.session_state["otm_cache"] = {}
if "last_otm_key" not in st.session_state:
    st.session_state["last_otm_key"] = None  # guarda a última key otimizada com sucesso
if "otm_error" not in st.session_state:
    st.session_state["otm_error"] = None     # guarda a última mensagem de erro (inviável)

def make_cache_key(ativos_ok, perfil, criterio, retorno_alvo, max_dd_user):
    return (
        "v16.5-session",                      # versão do cache
        tuple(ativos_ok),                     # ordem dos ativos
        perfil,
        criterio,
        round(float(retorno_alvo), 8),
        round(float(max_dd_user) if max_dd_user is not None else -1.0, 8),
    )

# Arquivos locais
CAMINHO_PLANILHA_ATIVOS = "ativos.xlsx"
CAMINHO_CLASSIFICACAO  = "classificacao_ativos.xlsx"
CAMINHO_BENCHMARK      = "benchmark.xlsx"
CAMINHO_RF             = "taxa livre de risco.xlsx"  # % a.a. (coluna numérica), 1ª coluna = Date

# Regras
MIN_PESO = 0.01  # 1% (regra 0% OU ≥1%)
CLASS_ORDER = ["Caixa", "Renda Fixa", "Ações", "Commodities"]

# Limite por ativo específico (AT1)
AT1_TICKER_EXATO = "AT1 LN Equity"
AT1_CAPS = {
    "Conservador": 0.05,
    "Moderado":    0.10,
    "Agressivo":   0.15,
}
def is_at1(ticker: str) -> bool:
    return str(ticker).strip() == AT1_TICKER_EXATO

# Classes consideradas "Caixa"
CASH_CLASS_ALIASES = {"Caixa", "Cash"}

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

# ================ FUNÇÕES BASE ================
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
    vol_ann = ret_alinh.std() * np.sqrt(252)
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

# ---- Drawdown helpers
def max_drawdown_of_weights(w, returns_df):
    pr = (returns_df * w).sum(axis=1)
    cum = (1 + pr).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min()), pr

def dd_constraint_factory(returns_df, dd_max):
    return lambda x: dd_max - abs(max_drawdown_of_weights(x, returns_df)[0])  # g(x) >= 0

# ---- Otimizadores
def minimize_volatility_with_constraints(mean_returns, cov_matrix, target_return, tickers, perfil, restricoes_por_perfil, classe_ativos):
    n = len(tickers)
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x, mu=mean_returns.values: np.dot(x, mu) - target_return}
    ]
    cons += class_constraints(tickers, perfil, restricoes_por_perfil, classe_ativos)
    bounds = build_bounds(tickers, perfil, lower_is_minpeso=False)  # lb=0 aqui; a regra ≥1% será aplicada depois
    x0 = np.ones(n) / n
    Sigma = cov_matrix.values
    obj = lambda x: float(np.sqrt(np.dot(x.T, np.dot(Sigma, x))))
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
    obj = lambda x: float(np.sqrt(np.dot(x.T, np.dot(Sigma, x))))
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

# ========= Seleção ordenada por classe =========
def class_index(classe: str) -> int:
    return CLASS_ORDER.index(classe) if classe in CLASS_ORDER else len(CLASS_ORDER) + 1

def sort_columns_by_class(cols, classe_ativos):
    return sorted(
        list(cols),
        key=lambda t: (class_index(classe_ativos.get(t, "Outros")), classe_ativos.get(t, "Outros"), str(t))
    )

# ========= UI helpers =========
def indicadores_totais(portfolio_returns, rf_daily=None):
    ann_return = (1 + portfolio_returns.mean())**252 - 1
    ann_vol    = portfolio_returns.std() * np.sqrt(252)
    sharpe     = calc_sharpe(portfolio_returns, rf_daily)
    cum = (1 + portfolio_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_drawdown = dd.min()
    return ann_return, ann_vol, sharpe, max_drawdown

def indicadores_por_ano(portfolio_returns, rf_daily=None):
    df = portfolio_returns.to_frame("ret")
    df["ano"] = df.index.year
    out = []
    for ano, g in df.groupby("ano"):
        ret = (1 + g["ret"]).prod() - 1
        vol = g["ret"].std() * np.sqrt(252)
        rf_sub = None
        if rf_daily is not None:
            _, rf_sub = g["ret"].align(rf_daily, join="inner")
        sharpe = calc_sharpe(g["ret"], rf_sub)
        cum = (1 + g["ret"]).cumprod()
        peak = cum.cummax()
        dd = (cum - peak)/peak
        max_dd = dd.min()
        out.append([str(ano), f"{ret*100:.2f}%", f"{vol*100:.2f}%", f"{sharpe:.2f}", f"{max_dd*100:.2f}%"])
    return pd.DataFrame(out, columns=["Ano", "Retorno", "Volatilidade", "Sharpe", "Máx. Drawdown"])

# ========= ENFORCE 0% OU >=1% =========
def enforce_min1(weights, tickers, perfil, mean_returns, cov_matrix,
                 restricoes_por_perfil, classe_ativos,
                 mode="min_vol_target", target_return=None,
                 returns_sel=None, max_dd=None, rf_daily=None):
    """
    Reotimiza no subconjunto de ativos com peso >= MIN_PESO impondo lb=MIN_PESO,
    mantendo as mesmas restrições (soma=1, classe, retorno alvo e/ou DD).
    mode: 'min_vol_target' | 'min_vol_target_dd' | 'gmvp' | 'max_sharpe'
    """
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

        bounds = [(MIN_PESO, 1.0) for _ in t_sub]  # >=1% dentro do subset
        x0 = np.ones(len(t_sub)) / len(t_sub)

        if mode in ("min_vol_target", "min_vol_target_dd", "gmvp"):
            # >>> sem parêntese extra aqui <<<
            obj = lambda x: float(np.sqrt(np.dot(Sig @ x, x)))
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
        if k < len(order):
            nxt = int(order[k])
            if nxt not in active:
                active.append(nxt)

    # fallback
    w0[w0 < MIN_PESO] = 0.0
    s = w0.sum()
    return w0 / s if s > 0 else np.ones(len(ativos)) / len(ativos)

# ========= CACHE para otimização principal =========
@st.cache_data(show_spinner=False)
def otimizar_portfolio(criterio, perfil, retorno_alvo, max_dd_user,
                       df_sel_columns, mean_returns, cov_matrix,
                       limites_demo, classe_ativos, returns_sel, rf_daily):
    if criterio == "Retorno alvo":
        w_opt_base = minimize_volatility_with_constraints(
            mean_returns, cov_matrix, retorno_alvo,
            df_sel_columns, perfil, limites_demo, classe_ativos
        )
        # Se o solver falhar, a função acima levanta ValueError.
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
        obj = lambda x: float(np.sqrt(np.dot(x.T, np.dot(Sigma, x))))
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            pesos_otimizados = enforce_min1(
                res.x, df_sel_columns, perfil, mean_returns, cov_matrix,
                limites_demo, classe_ativos,
                mode="min_vol_target_dd", target_return=retorno_alvo,
                returns_sel=returns_sel, max_dd=max_dd_user, rf_daily=rf_daily
            )
            achieved_dd, _ = max_drawdown_of_weights(pesos_otimizados, returns_sel)

            # Checagem explícita do teto de DD -> BLOQUEIA exibição se violado
            if abs(achieved_dd) <= (max_dd_user + EPS):
                dd_info = ("ok", achieved_dd)
            else:
                raise ValueError(
                    f"Sem solução factível para **Retorno alvo + Máx. DD** com os parâmetros atuais. "
                    f"(DD obtido {abs(achieved_dd)*100:.2f}% > limite {max_dd_user*100:.2f}%)"
                )
        else:
            raise ValueError("❌ Nenhuma carteira factível encontrada para **Retorno alvo + Máx. DD** "
                             "(tente relaxar o DD, ajustar limites por classe, incluir Caixa ou ampliar tolerância).")
    return pesos_otimizados, dd_info

def render_dashboard(pesos, titulo, returns, mean_returns, cov_matrix, df, classe_ativos,
                     benchmark_retornos=None, benchmark_nome=None,
                     mostrar_fronteira=False, frontier_vols=None, frontier_returns=None,
                     rf_daily=None, extra_points=None):
    st.subheader(titulo)

    cols_ord = sort_columns_by_class(df.columns, classe_ativos)
    pesos_map = pd.Series(pesos, index=df.columns)
    pesos_ord = pesos_map.reindex(cols_ord).fillna(0.0).values

    result_df = pd.DataFrame({
        "Ativo": cols_ord,
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
        (1 + portfolio_returns).cumprod().plot(ax=ax2, label="Carteira", lw=2)
        if benchmark_retornos is not None and not getattr(benchmark_retornos, "empty", True):
            port_alinh, bench_alinh = portfolio_returns.align(benchmark_retornos, join="inner")
            (1 + bench_alinh).cumprod().plot(ax=ax2, ls="--", lw=1.5, label=(benchmark_nome or "Benchmark"))
        ax2.set_title("Evolução do Portfólio"); ax2.legend()
        st.pyplot(fig2)

    with c3:
        fig3, ax3 = plt.subplots(figsize=(5, 3.2))
        if mostrar_fronteira and (frontier_vols is not None) and (frontier_returns is not None):
            ax3.plot(frontier_vols, frontier_returns, "b--", lw=1.2, label="Fronteira")
            ret_otimo, vol_otimo = portfolio_performance(pesos, mean_returns.values, cov_matrix.values)
            ax3.scatter(vol_otimo, ret_otimo, color="red", label="Ponto atual")
            if extra_points:
                for name, w, color, marker in extra_points:
                    r, v = portfolio_performance(w, mean_returns.values, cov_matrix.values)
                    ax3.scatter(v, r, c=color, marker=marker, s=50, label=name)
            ax3.set_xlabel("Volatilidade"); ax3.set_ylabel("Retorno"); ax3.legend()
            ax3.set_title("Fronteira Eficiente")
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

# =================== SIDEBAR (FORM PARA OTIMIZAÇÃO) ===================
st.sidebar.header("Configurações")

with st.sidebar.form("otimizacao_form", clear_on_submit=False):
    criterio = st.radio("⚙️ Critério de alocação", ["Retorno alvo", "Retorno alvo + Máx. DD"])
    perfil = st.selectbox("Perfil do investidor:", ["Conservador", "Moderado", "Agressivo"])
    retorno_alvo = st.slider("🎯 Retorno alvo anual (%)", 2.0, 20.0, 6.0, 0.1) / 100
    max_dd_user = None
    if criterio == "Retorno alvo + Máx. DD":
        max_dd_user = st.slider("📉 Máx. Drawdown permitido (%)", 1.0, 50.0, 20.0, 0.5) / 100

    # Limites por perfil
    limites_demo = {
        "Conservador": {"Caixa": (0, 1.0), "Ações": (0, 0.2), "Commodities": (0, 0.05), "Renda Fixa": (0.0, 1.0)},
        "Moderado":    {"Caixa": (0, 0.5), "Ações": (0, 0.5), "Commodities": (0, 0.15), "Renda Fixa": (0.0, 1.0)},
        "Agressivo":   {"Caixa": (0, 0.25), "Ações": (0, 1.0), "Commodities": (0, 0.3), "Renda Fixa": (0, 0.6)},
    }

    st.markdown("### Limites por Classe")
    limites_perfil = limites_demo[perfil]
    for classe in CLASS_ORDER:
        if classe in limites_perfil:
            mn, mx = limites_perfil[classe]
            st.write(f"*{classe}*: {mn*100:.0f}% – {mx*100:.0f}%")
    for classe in limites_perfil:
        if classe not in CLASS_ORDER:
            mn, mx = limites_perfil[classe]
            st.write(f"*{classe}*: {mn*100:.0f}% – {mx*100:.0f}%")
    st.write(f"*{AT1_TICKER_EXATO} — teto por ativo*: até {int(AT1_CAPS[perfil]*100)}%")

    # BOTÃO que dispara a otimização
    do_calc = st.form_submit_button("🚀 Calcular otimização")

# ================ PIPELINE ================
try:
    # 1) Preços (com cache)
    df_raw = load_excel_cached(CAMINHO_PLANILHA_ATIVOS)
    df, returns = prep_prices(df_raw)

    # 2) RF (opcional, com cache)
    rf_daily = None
    if os.path.exists(CAMINHO_RF):
        try:
            rf_daily = load_risk_free(CAMINHO_RF, dayfirst=True)
        except Exception:
            rf_daily = None

    # 3) Classes (com cache)
    df_classes_raw = load_excel_cached(CAMINHO_CLASSIFICACAO)
    classe_ativos = dict(zip(df_classes_raw["Ativo"], df_classes_raw["Classe"]))

    # ------------------- ABAS -------------------
    tab_sel, tab_otm, tab_manual, tab_comp = st.tabs(["Seleção", "Otimização", "Pesos manuais", "Comparar Ativos"])

    # --------- Seleção por classe ----------
    with tab_sel:
        st.subheader("Selecione os ativos por classe")

        if "ativos_selecionados" not in st.session_state:
            st.session_state["ativos_selecionados"] = list(df.columns)

        total_disponiveis = list(df.columns)
        todos_marcados = len(st.session_state["ativos_selecionados"]) == len(total_disponiveis)
        label_btn = "Desmarcar todos" if todos_marcados else "Selecionar todos"
        if st.button(label_btn, key="toggle_sel_all"):
            st.session_state["ativos_selecionados"] = [] if todos_marcados else total_disponiveis

        grupos = {}
        for t in df.columns:
            grupos.setdefault(classe_ativos.get(t, "Outros"), []).append(t)

        classes_ordenadas = [c for c in CLASS_ORDER if c in grupos] + [c for c in grupos if c não in CLASS_ORDER]
        colunas = st.columns(len(classes_ordenadas) if classes_ordenadas else 1)

        novos = set(st.session_state["ativos_selecionados"])
        for col, classe in zip(colunas, classes_ordenadas):
            with col:
                st.markdown(f"*{classe}*")
                for ativo in sorted(grupos[classe]):
                    ck = st.checkbox(ativo, value=(ativo in novos), key=f"sel_{ativo}")
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

    # 4) Estatísticas (com cache)
    mean_returns, cov_matrix = compute_stats(returns_sel)

    # 5)–8) OTIMIZAÇÃO/GMVP/MÁX.SHARPE/FRONTEIRA — RODAM APENAS QUANDO CLICAR NO BOTÃO
    cache_key = make_cache_key(df_sel.columns, perfil, criterio, retorno_alvo, max_dd_user)

    if do_calc:
        # Executa otimização e salva no cache da sessão
        try:
            st.session_state["otm_error"] = None
            pesos_otimizados, dd_info = otimizar_portfolio(
                criterio, perfil, retorno_alvo, max_dd_user,
                tuple(df_sel.columns), mean_returns, cov_matrix,
                limites_demo, classe_ativos, returns_sel, rf_daily
            )
            try:
                w_gmvp = gmvp_weights(mean_returns, cov_matrix, df_sel.columns, perfil, limites_demo, classe_ativos)
                w_gmvp = enforce_min1(
                    w_gmvp, df_sel.columns, perfil, mean_returns, cov_matrix,
                    limites_demo, classe_ativos,
                    mode="gmvp", rf_daily=rf_daily
                )
            except Exception:
                w_gmvp = None

            try:
                w_maxsh = max_sharpe_weights(mean_returns, cov_matrix, df_sel.columns, perfil, limites_demo, classe_ativos, rf_daily=rf_daily)
                w_maxsh = enforce_min1(
                    w_maxsh, df_sel.columns, perfil, mean_returns, cov_matrix,
                    limites_demo, classe_ativos,
                    mode="max_sharpe", rf_daily=rf_daily
                )
            except Exception:
                w_maxsh = None

            frontier_rets, frontier_vols = gerar_fronteira_eficiente(
                mean_returns, cov_matrix, tuple(df_sel.columns), perfil, limites_demo, classe_ativos
            )

            st.session_state["otm_cache"][cache_key] = {
                "pesos_otimizados": pesos_otimizados,
                "dd_info": dd_info,
                "w_gmvp": w_gmvp,
                "w_maxsh": w_maxsh,
                "frontier_rets": frontier_rets,
                "frontier_vols": frontier_vols,
            }
            st.session_state["last_otm_key"] = cache_key
            st.success("✅ Otimização concluída.")
        except Exception as e:
            # Marca erro e limpa a chave corrente para não mostrar resultado antigo
            st.session_state["otm_error"] = str(e)
            st.session_state["last_otm_key"] = None
            st.error(str(e))

    # 7) Benchmark (com cache leve de arquivo) — indep. do botão
    bench_series = {}
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

    # 9) OTIMIZAÇÃO (gráficos + comparações) — só mostra se houver resultado em cache ATUAL
    with tab_otm:
        nomes_bench = ["(Sem benchmark)"] + list(bench_series.keys())
        if "bench_sel" not in st.session_state:
            st.session_state["bench_sel"] = nomes_bench[0]
        bench_escolhido = st.selectbox(
            "Benchmark de comparação",
            nomes_bench,
            index=nomes_bench.index(st.session_state["bench_sel"])
        )
        st.session_state["bench_sel"] = bench_escolhido
        if bench_escolhido != "(Sem benchmark)":
            benchmark_retornos = bench_series[bench_escolhido]
            benchmark_nome = bench_escolhido
        else:
            benchmark_retornos = None
            benchmark_nome = None

        show_key = st.session_state.get("last_otm_key", None)
        erro_msg = st.session_state.get("otm_error", None)

        if erro_msg:
            # Inviável: mostra somente a mensagem
            st.warning("Não foi possível encontrar carteira com os parâmetros atuais de retorno e/ou Máx. DD.")
            st.info(erro_msg)
        elif show_key is None or show_key not in st.session_state["otm_cache"]:
            st.info("⚙️ Configure os parâmetros no sidebar e clique em **Calcular otimização** para ver os resultados.")
        else:
            hit = st.session_state["otm_cache"][show_key]
            pesos_otimizados = hit["pesos_otimizados"]
            dd_info          = hit["dd_info"]
            w_gmvp           = hit["w_gmvp"]
            w_maxsh          = hit["w_maxsh"]
            frontier_rets    = hit["frontier_rets"]
            frontier_vols    = hit["frontier_vols"]

            extras = []
            if w_gmvp is not None: extras.append(("GMVP", w_gmvp, "tab:blue", "D"))
            if w_maxsh is not None: extras.append(("Máx. Sharpe", w_maxsh, "tab:green", "^"))

            subt = f"📊 Alocação Ótima — Perfil {perfil} (Retorno alvo: {retorno_alvo*100:.1f}%)"
            if criterio == "Retorno alvo + Máx. DD" and ('max_dd_user' in locals()) and max_dd_user is not None:
                subt += f" • Máx. DD: {max_dd_user*100:.1f}%"

            render_dashboard(
                pesos=pesos_otimizados,
                titulo=subt,
                returns=returns_sel,
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                df=df_sel,
                classe_ativos=classe_ativos,
                benchmark_retornos=benchmark_retornos,
                benchmark_nome=benchmark_nome,
                mostrar_fronteira=True,
                frontier_vols=frontier_vols,
                frontier_returns=frontier_rets,
                rf_daily=rf_daily,
                extra_points=extras
            )

            if dd_info is not None:
                modo, dd_val = dd_info
                if modo == "ok":
                    st.success(f"Máx. Drawdown da carteira exibida: {abs(dd_val)*100:.2f}% (dentro do alvo).")

    # 10) PESOS MANUAIS — independe do botão
    with tab_manual:
        nomes_bench = ["(Sem benchmark)"] + list(bench_series.keys())
        bench_escolhido = st.selectbox("Benchmark de comparação", nomes_bench,
                                       index=nomes_bench.index(st.session_state.get("bench_sel", nomes_bench[0])),
                                       key="bench_sel_manual")
        st.session_state["bench_sel"] = bench_escolhido
        if bench_escolhido != "(Sem benchmark)":
            benchmark_retornos = bench_series[bench_escolhido]
            benchmark_nome = bench_escolhido
        else:
            benchmark_retornos = None
            benchmark_nome = None

        st.subheader("✍️ Pesos manuais (em %)")
        cols_ord = sort_columns_by_class(df_sel.columns, classe_ativos)
        df_pesos = pd.DataFrame({"Ativo": cols_ord, "Classe": [classe_ativos.get(t, "Outros") for t in cols_ord], "Peso (%)": np.zeros(len(cols_ord))})
        if st.checkbox("Iniciar com pesos iguais"):
            df_pesos["Peso (%)"] = np.round(100 / len(cols_ord), 2)

        edited = st.data_editor(
            df_pesos,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "Peso (%)": st.column_config.NumberColumn(step=0.1, min_value=0.0, max_value=100.0),
                "Classe": st.column_config.TextColumn(disabled=True),
            }
        )
        soma = float(edited["Peso (%)"].sum())
        st.write(f"*Soma atual:* {soma:.2f}%")
        normalizar = st.checkbox("Normalizar automaticamente para 100%", value=True)

        if st.button("Aplicar pesos manuais"):
            try:
                w_series = pd.Series(edited["Peso (%)"].astype(float).values / 100.0, index=edited["Ativo"])
                w_series = w_series.reindex(df_sel.columns).fillna(0.0)
                pesos_man = w_series.values
            except Exception as e:
                st.error(f"Erro ao ler pesos manuais: {e}")
                pesos_man = np.ones(len(df_sel.columns)) / len(df_sel.columns)

            if normalizar and pesos_man.sum() > 0:
                pesos_man = pesos_man / pesos_man.sum()
            elif not np.isclose(pesos_man.sum(), 1.0):
                st.error("A soma dos pesos precisa ser 100% (ou marque Normalizar).")
                st.stop()

            render_dashboard(
                pesos=pesos_man,
                titulo="📊 Carteira com Pesos Manuais",
                returns=returns_sel,
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                df=df_sel,
                classe_ativos=classe_ativos,
                benchmark_retornos=benchmark_retornos,
                benchmark_nome=benchmark_nome,
                mostrar_fronteira=False,
                rf_daily=rf_daily,
            )

    # 11) COMPARAR ATIVOS — independe do botão
    with tab_comp:
        st.subheader("Comparar Ativos")

        LIMITE_ATIVOS = 6
        if "ativos_comp" not in st.session_state:
            st.session_state["ativos_comp"] = list(df_sel.columns[:min(LIMITE_ATIVOS, len(df_sel.columns))])

        total_disponiveis_comp = list(df_sel.columns)
        todos_marcados_comp = len(st.session_state["ativos_comp"]) == len(total_disponiveis_comp)
        label_btn_comp = "Desmarcar todos" if todos_marcados_comp else "Selecionar todos"
        if st.button(label_btn_comp, key="toggle_comp_all"):
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
                    marcado = ativo in base_inicial
                    ck = st.checkbox(ativo, value=marcado, key=f"comp_{ativo}")
                    if ck:
                        escolhidos.append(ativo)

        escolhidos = list(dict.fromkeys(escolhidos))
        if len(escolhidos) > LIMITE_ATIVOS:
            escolhidos = escolhidos[:LIMITE_ATIVOS]
            st.warning(f"Limite de {LIMITE_ATIVOS} ativos para comparação. Apenas os primeiros foram mantidos.")

        st.session_state["ativos_comp"] = escolhidos
        ativos_comp = st.session_state["ativos_comp"]

        if not ativos_comp:
            st.warning("Selecione ao menos um ativo para comparar.")
        else:
            ret_comp = returns_sel[ativos_comp].dropna(how="all")
            if ret_comp.empty:
                st.warning("Não há dados válidos para os ativos selecionados.")
            else:
                st.markdown("**Evolução acumulada (base = 1.0)**")
                cum = (1 + ret_comp).cumprod()

                figc, axc = plt.subplots(figsize=(3.2, 1.2))
                figc.set_dpi(180)
                for c in cum.columns:
                    axc.plot(cum.index, cum[c], lw=1.0, label=c)

                locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
                axc.xaxis.set_major_locator(locator)
                axc.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

                ymin, ymax = float(cum.min().min()), float(cum.max().max())
                if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
                    ymin, ymax = 0.95, 1.05
                axc.set_yticks(np.linspace(ymin, ymax, 3))
                axc.tick_params(axis="x", labelsize=8)
                axc.tick_params(axis="y", labelsize=8)
                axc.margins(x=0)
                axc.grid(alpha=0.25, linewidth=0.5)

                axc.legend(
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    fontsize=8,
                    frameon=False,
                    ncol=1,
                    handlelength=1.6,
                )
                axc.set_xlabel("")
                axc.set_ylabel("")
                st.pyplot(figc)

                linhas = []
                for c in ativos_comp:
                    s = ret_comp[c].dropna()
                    if s.empty:
                        continue
                    r = (1 + s.mean())**252 - 1
                    v = s.std() * np.sqrt(252)
                    sh = calc_sharpe(s, rf_daily)
                    cum_s = (1 + s).cumprod(); pk = cum_s.cummax()
                    dd = (cum_s - pk) / pk; mdd = dd.min()
                    linhas.append([c, classe_ativos.get(c, "Outros"),
                                   f"{r*100:.2f}%", f"{v*100:.2f}%", f"{sh:.2f}", f"{mdd*100:.2f}%"])
                df_comp = pd.DataFrame(linhas, columns=["Ativo", "Classe", "Retorno (a.a.)", "Vol. (a.a.)", "Sharpe", "Máx. DD"])
                st.dataframe(df_comp, use_container_width=True)

except Exception as e:
    st.error(f"❌ Erro ao processar: {e}")

# Aviso se benchmark vazio
if 'benchmark_retornos' in locals() and benchmark_retornos is not None and getattr(benchmark_retornos, "empty", True):
    st.warning("⚠️ O benchmark selecionado não possui dados válidos.")
    benchmark_retornos = None