import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
import os
import base64
import streamlit.components.v1 as components

st.set_page_config(page_title="Simulador de Balanceo", layout="wide")

# =========================
# 🔢 Funciones auxiliares
# =========================
def decimal_a_sexagesimal(valor):
    minutos = int(valor)
    segundos = int(round((valor - minutos) * 60))
    return f"{minutos}:{segundos:02d}"

# =========================
# 🎨 Estilos generales
# =========================
st.markdown("""
    <style>
    .main { background-color: #f0f6fc; }
    h1, h2, h3 { color: #1e3a8a !important; }
    div[data-testid="stMetricValue"] { color: #1e40af; font-weight: bold; }
    section[data-testid="stSidebar"] { background-color: #e0f2fe; }
    </style>
""", unsafe_allow_html=True)

st.title("Simulador de saturación — Operadores y Puestos")

# =========================
# Cargar Excel
# =========================
archivo = "Base de datos tiempos IVECO 2025 - copia.xlsx"
df_base_raw = pd.read_excel(archivo, sheet_name="Base de datos")
df_takt = pd.read_excel(archivo, sheet_name="DR line")

# =========================
# Normalizar columnas
# =========================
colmap = {}
for c in df_base_raw.columns:
    cl = c.lower()
    if "puest" in cl or "work" in cl: colmap[c] = "Puesto"
    elif "model" in cl: colmap[c] = "Modelo"
    elif "oper" in cl: colmap[c] = "Operador"
    elif "desc" in cl or "task" in cl: colmap[c] = "Descripcion"
    elif "std" in cl: colmap[c] = "STD_min"
    elif "paq" in cl or "pack" in cl: colmap[c] = "Paquete"
df_base = df_base_raw.rename(columns=colmap)

colmap_takt = {}
for c in df_takt.columns:
    cl = c.lower().strip()
    if "dr" in cl: colmap_takt[c] = "DR"
    elif "line" in cl: colmap_takt[c] = "Linea"
    elif "tack" in cl and "time" in cl: colmap_takt[c] = "Takt_min"
    elif "puest" in cl: colmap_takt[c] = "Puesto"
df_takt = df_takt.rename(columns=colmap_takt)

# Limpiar strings
for col in ["Puesto", "Modelo", "Operador"]:
    if col in df_base.columns:
        df_base[col] = df_base[col].astype(str).str.strip()
if "Puesto" in df_takt.columns:
    df_takt["Puesto"] = df_takt["Puesto"].astype(str).str.strip()
if "Linea" in df_takt.columns:
    df_takt["Linea"] = df_takt["Linea"].astype(str).str.strip()

# ---- Mapa Puesto→Línea
map_linea = df_takt[["Puesto", "Linea"]].dropna().drop_duplicates()
df_base = df_base.merge(map_linea, on="Puesto", how="left")

# Completar columnas base
if "Paquete" not in df_base.columns:
    df_base["Paquete"] = df_base.index + 1

# ⚠️ STD en SEGUNDOS → pasamos a MINUTOS
df_base["STD_min"] = pd.to_numeric(df_base["STD_min"], errors="coerce").fillna(0) / 60.0

# Operador simple
df_base["Operador_simple"] = df_base["Operador"].astype(str).str.strip().str[0]
df_base["Movido"] = "No"

# DR line: tipos
df_takt["DR"] = pd.to_numeric(df_takt["DR"], errors="coerce")
df_takt["Takt_min"] = pd.to_numeric(df_takt["Takt_min"], errors="coerce")

# =========================
# Estado de simulación
# =========================
if "base_original" not in st.session_state:
    st.session_state.base_original = df_base.copy()
if "df_sim" not in st.session_state:
    st.session_state.df_sim = st.session_state.base_original.copy()
if "mov_hist" not in st.session_state:
    # cada item: {"idx":[...], "operador_origen":str, "operador_dest":str, "prev_mov":[...], "paquete": valor, "moved_time": float}
    st.session_state.mov_hist = []

# =========================
# Parámetros (sidebar)
# =========================
st.sidebar.header("⚙️ Parámetros de planificación")
dr_sel = st.sidebar.selectbox("Seleccionar DR", sorted(df_takt["DR"].dropna().unique()))
linea_sel = st.sidebar.selectbox("Seleccionar línea", sorted(df_takt["Linea"].dropna().unique()))

puestos_linea = df_takt.loc[df_takt["Linea"] == linea_sel, "Puesto"].dropna().unique()
if len(puestos_linea) == 0:
    st.error(f"No hay puestos asociados a la línea {linea_sel}.")
    st.stop()
puesto_sel = st.sidebar.selectbox("Seleccionar puesto", sorted(puestos_linea))

# Takt seleccionado
match = df_takt[(df_takt["DR"] == dr_sel) & (df_takt["Linea"] == linea_sel) & (df_takt["Puesto"] == puesto_sel)]
if match.empty:
    st.error(f"No encontré Takt_min para DR={dr_sel}, Línea={linea_sel}, Puesto={puesto_sel}")
    st.stop()
takt_time = float(match["Takt_min"].iloc[0])
st.sidebar.metric("Takt Time (min/camión)", decimal_a_sexagesimal(takt_time))


# =========================
# Selección de modelo
# =========================
modelos_disp = st.session_state.df_sim.loc[
    st.session_state.df_sim["Puesto"] == puesto_sel, "Modelo"
].dropna().unique()
if len(modelos_disp) == 0:
    st.warning("No hay modelos asociados a este puesto.")
    st.stop()
filtro_modelo = st.selectbox("Seleccionar modelo", sorted(modelos_disp))

# =========================
# Bloque único de reasignación + Historial
# =========================
st.subheader("🎛 Reasignación de paquetes (bloque único)")

df_filtrado = st.session_state.df_sim[
    (st.session_state.df_sim["Puesto"] == puesto_sel) &
    (st.session_state.df_sim["Modelo"] == filtro_modelo)
].copy()

if df_filtrado.empty:
    st.warning("No hay actividades para este puesto y modelo.")
    st.stop()

# 📋 Tabla de actividades (con Movido resaltado + STD en sexagesimal)
st.subheader("📋 Actividades del puesto y modelo seleccionados")

def resaltar_movido(val):
    return "background-color: #bbf7d0; font-weight: bold;" if str(val).strip().lower() == "sí" else ""

# Nueva columna con conversión a sexagesimal
df_filtrado["STD_sexagesimal"] = df_filtrado["STD_min"].apply(decimal_a_sexagesimal)

st.dataframe(
    df_filtrado[["Descripcion", "Operador", "STD_sexagesimal", "Paquete", "Movido"]]
    .style.applymap(resaltar_movido, subset=["Movido"]),
    hide_index=True, use_container_width=True
)

# UI a la izquierda, historial a la derecha
col_left, col_right = st.columns([1.2, 1.8])

with col_left:
    st.markdown("### 🎯 Movimiento")
    paquete_sel = st.selectbox("Elegir paquete", df_filtrado["Paquete"].dropna().unique())
    operadores_origen = df_filtrado.loc[df_filtrado["Paquete"] == paquete_sel, "Operador_simple"].dropna().unique()
    operador_origen = st.selectbox("Operador origen", sorted(operadores_origen))
    operadores_dest = df_filtrado["Operador_simple"].dropna().unique()
    operadores_dest = [op for op in operadores_dest if op != operador_origen]
    operadores_dest = sorted(operadores_dest) + ["➕ Nuevo operador"]
    operador_dest = st.selectbox("Operador destino", operadores_dest)

    if st.button("Reasignar paquete"):
        if operador_dest == "➕ Nuevo operador":
            existentes = st.session_state.df_sim["Operador_simple"].astype(str).unique().tolist()
            nuevo = f"N{len([o for o in existentes if str(o).startswith('N')]) + 1}"
            operador_dest = nuevo
            st.info(f"👤 Se agregó nuevo operador: {operador_dest}")

        mask = (
            (st.session_state.df_sim["Paquete"] == paquete_sel) &
            (st.session_state.df_sim["Operador_simple"] == operador_origen) &
            (st.session_state.df_sim["Puesto"] == puesto_sel) &
            (st.session_state.df_sim["Modelo"] == filtro_modelo)
        )
        idx = st.session_state.df_sim.index[mask].tolist()
        moved_time = float(st.session_state.df_sim.loc[idx, "STD_min"].sum())
        prev_mov = st.session_state.df_sim.loc[idx, "Movido"].tolist()

        st.session_state.mov_hist.append({
            "idx": idx,
            "operador_origen": operador_origen,
            "operador_dest": operador_dest,
            "prev_mov": prev_mov,
            "paquete": paquete_sel,
            "moved_time": moved_time
        })

        st.session_state.df_sim.loc[idx, "Operador_simple"] = operador_dest
        st.session_state.df_sim.loc[idx, "Movido"] = "Sí"
        st.success(f"📦 Paquete {paquete_sel} movido de {operador_origen} a {operador_dest} ({moved_time:.3f} min)")
        st.rerun()

    if st.button("↩️ Deshacer último movimiento"):
        if len(st.session_state.mov_hist) == 0:
            st.info("No hay movimientos para deshacer.")
        else:
            last = st.session_state.mov_hist.pop()
            idx = last["idx"]
            st.session_state.df_sim.loc[idx, "Operador_simple"] = last["operador_origen"]
            st.session_state.df_sim.loc[idx, "Movido"] = last["prev_mov"]
            st.success("Se deshizo el último movimiento.")
            st.rerun()

with col_right:
    st.markdown("### 🧾 Historial de movimientos")
    # Cabecera
    h1, h2, h3, h4 = st.columns([1, 1, 1, 0.8])
    h1.markdown("**Paquete**")
    h2.markdown("**Origen**")
    h3.markdown("**Destino**")
    h4.markdown("**Acción**")

    # Filas con botón de deshacer puntual
    for i, mov in enumerate(st.session_state.mov_hist):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 0.8])
        c1.write(str(mov.get("paquete", "")))
        c2.write(str(mov["operador_origen"]))
        c3.write(str(mov["operador_dest"]))
        if c4.button("↩️ Deshacer", key=f"undo_row_{i}"):
            idx = mov["idx"]
            st.session_state.df_sim.loc[idx, "Operador_simple"] = mov["operador_origen"]
            st.session_state.df_sim.loc[idx, "Movido"] = mov["prev_mov"]
            # Eliminar SOLO ese movimiento del historial
            st.session_state.mov_hist.pop(i)
            st.success(f"Se deshizo el movimiento del paquete {mov.get('paquete','')}.")
            st.rerun()

st.divider()

# =========================
# Tabla de paquetes + mini gráfico de saturación
# =========================
st.subheader("⏱️ Tiempo por paquete y operador")

col_table, col_metric = st.columns([3,1])

with col_table:
    df_no_ntc = df_filtrado[df_filtrado["Paquete"] != "NTC"]
    pivot_paquete = df_no_ntc.pivot_table(
    index="Paquete", columns="Operador_simple", values="STD_min",
    aggfunc="sum", fill_value=0
    ).round(3)

    pivot_paquete = pivot_paquete.applymap(decimal_a_sexagesimal)
    st.dataframe(pivot_paquete, use_container_width=True)

with col_metric:
    # Calcular saturación en tiempo real
    pivot_oper = df_filtrado.groupby("Operador_simple", as_index=False).agg({"STD_min": "sum"})
    pivot_oper["Saturacion_pct"] = pivot_oper["STD_min"] / takt_time * 100

    # Asignar colores según reglas
    def asignar_color(x):
        if x < 85:
            return "yellow"
        elif x <= 100:
            return "green"
        else:
            return "red"
    pivot_oper["Color"] = pivot_oper["Saturacion_pct"].apply(asignar_color)

    # Mini gráfico horizontal con línea en 100%
    fig_mini = px.bar(
        pivot_oper,
        x="Saturacion_pct",
        y="Operador_simple",
        orientation="h",
        text=pivot_oper["Saturacion_pct"].round(1).astype(str) + "%",
        color="Color",
        color_discrete_map={"yellow":"#facc15", "green":"#22c55e", "red":"#ef4444"},
        range_x=[0, max(100, float(pivot_oper["Saturacion_pct"].max()) + 20)]
    )
    fig_mini.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=16, color="white")
    )
    fig_mini.add_vline(x=100, line_dash="solid", line_color="blue", annotation_text="100%")
    fig_mini.update_layout(showlegend=False, height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_mini, use_container_width=True)

st.divider()

# ======================================
# 📊 Saturación por operador — VA + NVAA
# ======================================
# ... (imports y st.subheader) ...

# ---- Config de columnas (según tu base) ----
COL_TIME   = "STD_min"
COL_CLASE  = "Clasif"
COL_OPER   = "Operador_simple"
CODE_LIST  = ["Q", "WM", "WT", "C", "R", "P"]

# ---- Validaciones ----
faltan = [c for c in [COL_TIME, COL_CLASE, COL_OPER] if c not in df_filtrado.columns]
if faltan:
    st.error(f"Faltan columnas requeridas: {faltan}")
else:
    code_cols = [c for c in CODE_LIST if c in df_filtrado.columns]
    if not code_cols:
        st.error("No encuentro columnas de NVAA (Q/WM/WT/C/R/P).")
    else:
        df = df_filtrado.copy()

        # -----------------------------------------
        # 1) Normalización de unidades (SEG → MIN) auto
        # -----------------------------------------
        raw_time = pd.to_numeric(df[COL_TIME], errors="coerce").fillna(0.0)
        is_seconds_time = (raw_time.quantile(0.90) > 300) or (raw_time.mean() > 120)
        df[COL_TIME] = raw_time / 60.0 if is_seconds_time else raw_time

        for col in code_cols:
            raw_code = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            is_seconds_code = (raw_code.quantile(0.90) > 300) or (raw_code.mean() > 120)
            df[col] = raw_code / 60.0 if is_seconds_code else raw_code

        df[COL_CLASE] = df[COL_CLASE].astype(str).str.upper().str.strip().replace({"N/VAA": "NVAA"})

        # -----------------------------------------
        # 2) Agregados por operador
        # -----------------------------------------
        va_df = (
            df.loc[df[COL_CLASE] == "VA", [COL_OPER, COL_TIME]]
              .groupby(COL_OPER, as_index=False)[COL_TIME]
              .sum()
              .rename(columns={COL_TIME: "VAA"})
        )
        nvaa_df = (
            df.loc[df[COL_CLASE] == "NVAA", [COL_OPER, COL_TIME]]
              .groupby(COL_OPER, as_index=False)[COL_TIME]
              .sum()
              .rename(columns={COL_TIME: "NVAA_TOTAL"})
        )
        pesos_list = []
        for col in code_cols:
            g = (
                df.loc[df[COL_CLASE] == "NVAA"]
                  .groupby(COL_OPER, as_index=False)[col]
                  .sum()
                  .rename(columns={col: f"{col}_peso"})
            )
            pesos_list.append(g)

        from functools import reduce
        dfs_merge = [va_df, nvaa_df] + pesos_list
        agg = reduce(lambda l, r: pd.merge(l, r, on=COL_OPER, how="outer"), dfs_merge).fillna(0.0)

        peso_cols = [f"{c}_peso" for c in code_cols]
        agg["PESOS_SUM"] = agg[peso_cols].sum(axis=1) if peso_cols else 0.0

        for c in code_cols:
            pc = f"{c}_peso"
            agg[c] = np.where(
                agg["PESOS_SUM"] > 0,
                agg["NVAA_TOTAL"] * (agg[pc] / agg["PESOS_SUM"]),
                0.0,
            )

        agg["NVAA_SUM"] = agg[code_cols].sum(axis=1) if code_cols else 0.0
        agg["TOTAL"]    = agg["VAA"] + agg["NVAA_SUM"]

        # -----------------------------------------
        # 3) DF largo (ya no creamos "Label")
        # -----------------------------------------
        segmentos = ["VAA"] + code_cols
        stacked = agg.melt(
            id_vars=[COL_OPER, "TOTAL"],
            value_vars=segmentos,
            var_name="Categoria",
            value_name="Minutos",
        )

        stacked["Pct"] = stacked.apply(
            lambda r: (r["Minutos"] / r["TOTAL"] * 100) if r["TOTAL"] > 0 else 0,
            axis=1
        )
        
        # ===================================================
        # === ARREGLO 1: Renombrar Pct a Porcentaje ===
        # ===================================================
        stacked = stacked.rename(columns={"Pct": "Porcentaje"})
        # ===================================================

        # -----------------------------------------
        # 4) Gráfico apilado (Modificado)
        # -----------------------------------------
        color_map = {
            "VAA": "#00B050", "Q": "#0000FF", "WM": "#7030A0",
            "WT": "#808080", "C": "#FFFF00", "R": "#FFC000", "P": "#E60000",
        }
        ordered = [c for c in ["VAA","Q","WM","WT","C","R","P"] if c in segmentos]

        fig1 = px.bar(
            stacked,
            x=COL_OPER,
            y="Minutos",
            color="Categoria",
            category_orders={"Categoria": ordered},
            color_discrete_map=color_map,
            title="Saturación por operador — VA + NVAA (Q/WM/WT/C/R/P)",
            hover_data={
                "Minutos": ":.2f",
                "Porcentaje": ":.1f", # <-- Ahora sí coincide
                "Categoria": True,
                COL_OPER: True,
                "TOTAL": ":.2f",
            },
        )

        fig1.update_traces(
            cliponaxis=False,
        )
        
        fig1.update_layout(
            barmode="stack",
            xaxis_title="Operador",
            yaxis_title="Minutos",
            legend_title_text="Categoría",
            hoverlabel=dict(
                font_size=20 # <-- ¡Perfecto!
            )
        )

        # ===================================================
        # === ARREGLO 2: Eliminado el loop duplicado ===
        # ===================================================
        
        # Totales arriba (mm:ss) - (UNA SOLA VEZ)
        for _, row in agg.iterrows():
            if row["TOTAL"] > 0:
                fig1.add_annotation(
                    x=row[COL_OPER], y=row["TOTAL"],
                    text=decimal_a_sexagesimal(row["TOTAL"]),
                    showarrow=False, yanchor="bottom", yshift=10, font=dict(size=13),
                )

        # Línea de takt (en minutos) - Esto queda igual
        fig1.add_hline(
            y=takt_time,
            line_dash="solid",
            line_color="blue",
            annotation_text="Takt individual"
        )

        st.plotly_chart(fig1, use_container_width=True)

# =========================
# 🌐 Saturación global de la línea (con selección de modelo por puesto)
# =========================
st.subheader(f"🌐 Saturación global de la línea {linea_sel}")

modelos_por_puesto = {}
with st.expander("⚙️ Selección de modelo por puesto"):
    for p in sorted(puestos_linea):
        modelos_disp_p = st.session_state.df_sim.loc[
            st.session_state.df_sim["Puesto"] == p, "Modelo"
        ].dropna().unique()
        if len(modelos_disp_p) == 0:
            modelos_por_puesto[p] = None
            st.write(f"{p}: ❌ Sin modelos disponibles")
        else:
            modelos_por_puesto[p] = st.selectbox(
                f"Modelo para {p}",
                options=sorted(modelos_disp_p),
                index=0,
                key=f"modelo_global_{p}"
            )

df_linea = pd.DataFrame({"Puesto": sorted(puestos_linea)})
std_por_puesto = []
for p in puestos_linea:
    modelo_p = modelos_por_puesto.get(p)
    if modelo_p:
        df_temp = st.session_state.df_sim[
            (st.session_state.df_sim["Linea"] == linea_sel) &
            (st.session_state.df_sim["Puesto"] == p) &
            (st.session_state.df_sim["Modelo"] == modelo_p)
        ].groupby("Puesto", as_index=False)["STD_min"].sum()
        std_por_puesto.append(df_temp)
if std_por_puesto:
    std_por_puesto = pd.concat(std_por_puesto, ignore_index=True)
    df_linea = df_linea.merge(std_por_puesto, on="Puesto", how="left").fillna({"STD_min": 0})
else:
    df_linea["STD_min"] = 0

oper_por_puesto = st.session_state.df_sim[
    st.session_state.df_sim["Linea"] == linea_sel
].groupby("Puesto")["Operador_simple"].nunique()
df_linea["Operarios_puesto"] = df_linea["Puesto"].map(oper_por_puesto).fillna(0).astype(int)
df_linea["TTK_disp"] = takt_time * df_linea["Operarios_puesto"]
df_linea["Saturacion_pct"] = (df_linea["STD_min"] / df_linea["TTK_disp"].replace(0, pd.NA) * 100).fillna(0)
df_linea["Color"] = df_linea["Saturacion_pct"].apply(lambda x: "red" if x > 100 else "green")

sat_total = (df_linea["STD_min"].sum() / df_linea["TTK_disp"].sum() * 100) if df_linea["TTK_disp"].sum() > 0 else 0
st.metric(label=f"Saturación total de la línea {linea_sel}", value=f"{sat_total:.1f}%")

fig2 = px.bar(
    df_linea, x="Puesto", y="STD_min",
    color="Color",
    text=df_linea["STD_min"].apply(decimal_a_sexagesimal),
    title=f"Saturación global por puesto — Línea {linea_sel}",
    color_discrete_map={"red":"#ef4444","green":"#16a34a"}
)

st.plotly_chart(fig2, use_container_width=True, key="fig_global_linea")

st.divider()


# =========================
# 📥 Exportar simulación con copia en OneDrive
# =========================
st.subheader("📥 Exportar simulación")

# Lista fija de legajos válidos
legajos_validos = [
    "29002467", "29001542", "29001385", "29000869",
    "29005111", "29003780", "29000677", "29000046",
    "29002132", "29020004", "29014765"
]
opciones_legajo = ["--- Seleccione un legajo ---"] + legajos_validos

# Inicializar el estado del legajo si no existe
if "legajo_select" not in st.session_state:
    st.session_state.legajo_select = "--- Seleccione un legajo ---"

# Dropdown
usuario = st.selectbox(
    "👤 Seleccioná tu legajo",
    opciones_legajo,
    index=opciones_legajo.index(st.session_state.legajo_select),
    key="legajo_select"
)

# Botón de control para ejecutar la exportación
if st.button("💾 Guardar y Exportar Simulación", key="export_trigger_button"):

    # 1. Validar el legajo DESPUÉS de hacer clic
    if usuario == "--- Seleccione un legajo ---":
        st.warning("⚠️ Seleccioná tu legajo para poder exportar.")
    
    else:
        # 2. Si el legajo es válido, ejecutar TODA la lógica de exportación
        
        cols_export = ["Puesto", "Modelo", "Descripcion", "Operador",
                       "STD_min", "Paquete", "Linea", "Operador_simple", "Movido"]
        cols_export = [c for c in cols_export if c in st.session_state.df_sim.columns]
        
        st.write("Exportando el estado final de TODOS los puestos/modelos modificados...")
        
        pares_modificados = st.session_state.df_sim.loc[
            st.session_state.df_sim["Movido"] == "Sí", 
            ["Puesto", "Modelo"]
        ].drop_duplicates()

        if pares_modificados.empty:
            st.info("No se detectaron movimientos para exportar (ningún paquete marcado como 'Sí').")
            df_export = pd.DataFrame(columns=cols_export) 
            
        else:
            df_export = pd.merge(
                st.session_state.df_sim, 
                pares_modificados, 
                on=["Puesto", "Modelo"], 
                how="inner"
            )[cols_export]

        # Agregar hoja "Log"
        df_log = pd.DataFrame([{
        "Legajo": usuario,
        "FechaHora": pd.Timestamp.now(),
        "Exportacion": "Exportación de todos los modelos modificados (Opción C)" 
        }])

        # ==================================================================
        # === INICIO DE LA MODIFICACIÓN (Lógica para el nombre) ===
        # ==================================================================
        
        # 1. Obtener el DR actual (de la variable del sidebar 'dr_sel')
        #    Asegúrate de que 'dr_sel' esté disponible aquí. 
        #    (¡Sí lo está, porque se define en el sidebar!)
        dr_actual = str(dr_sel).replace(" ", "_") 

        # 2. Obtener lista de puestos únicos modificados
        #    (Usamos la variable 'pares_modificados' que ya creamos)
        lista_puestos_modificados = pares_modificados["Puesto"].unique().tolist()
        
        nombre_puesto_str = ""
        if len(lista_puestos_modificados) == 1:
            # Solo 1 puesto modificado, usamos su nombre
            nombre_puesto_str = str(lista_puestos_modificados[0])
        elif len(lista_puestos_modificados) > 1:
            # Múltiples puestos modificados
            nombre_puesto_str = "VARIOS-PUESTOS"
        else:
            # Ningún puesto modificado (aunque df_export.empty ya lo chequeó)
            nombre_puesto_str = "NINGUNO"

        # Limpiamos el string del puesto para que sea un nombre de archivo válido
        nombre_puesto_str = nombre_puesto_str.replace(" ", "_").replace("/", "-")

        # --- Creación del nombre de archivo ---
        carpeta_guardado = r"C:\Users\v13912b\OneDrive - Iveco Group\Escritorio\App_Saturacion\Simulaciones"
        os.makedirs(carpeta_guardado, exist_ok=True)
        
        # ¡NUEVO NOMBRE DE ARCHIVO!
        nombre_archivo = f"Simulacion_DR{dr_actual}_{nombre_puesto_str}_{usuario}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        ruta_completa = os.path.join(carpeta_guardado, nombre_archivo)
        
        # ==================================================================
        # === FIN DE LA MODIFICACIÓN ===
        # ==================================================================


        # Solo si el df_export NO está vacío
        if not df_export.empty:
            
            # --- INICIO DE LÓGICA DE EXPORTACIÓN (DEBE ESTAR AQUÍ DENTRO) ---

            # 1. Guardar en OneDrive (Esto dispara el mail)
            with pd.ExcelWriter(ruta_completa, engine="xlsxwriter") as writer:
                df_export.to_excel(writer, index=False, sheet_name="Simulacion")
                df_log.to_excel(writer, index=False, sheet_name="Log")
            st.success(f"💾 Simulación guardada automáticamente en: {ruta_completa}. Power Automate enviará el mail.")

            # 2. Preparar para descarga local (en memoria)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_export.to_excel(writer, index=False, sheet_name="Simulacion")
                df_log.to_excel(writer, index=False, sheet_name="Log")
            data = output.getvalue()

            # 3. Codificar en Base64
            b64 = base64.b64encode(data).decode()

            # 4. Crear el HTML/JS para la descarga automática
            html_payload = f"""
                <html>
                <head>
                <script>
                    function triggerDownload() {{
                        const data = '{b64}';
                        const sliceSize = 512;
                        const byteCharacters = atob(data);
                        const byteArrays = [];

                        for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {{
                            const slice = byteCharacters.slice(offset, offset + sliceSize);
                            const byteNumbers = new Array(slice.length);
                            for (let i = 0; i < slice.length; i++) {{
                                byteNumbers[i] = slice.charCodeAt(i);
                            }}
                            const byteArray = new Uint8Array(byteNumbers);
                            byteArrays.push(byteArray);
                        }}

                        const blob = new Blob(byteArrays, {{type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}});
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.style.display = 'none';
                        a.href = url;
                        a.download = '{nombre_archivo}';
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        a.remove();
                    }}
                </script>
                </head>
                <body onload="triggerDownload()">
                </body>
                </html>
            """
            
            # 5. Ejecutar el "truco"
            st.success(f"📥 Descargando {nombre_archivo}...")
            components.html(html_payload, height=0)

            # --- FIN DE LÓGICA DE EXPORTACIÓN ---
        
        # (Si df_export está vacío, solo se habrá mostrado el st.info)

# NADA DE CÓDIGO DE EXPORTACIÓN DEBE ESTAR AQUÍ AFUERA

# =========================
# 🔄 Reset global (Poner al FINAL de todo el script)
# =========================
st.divider()
st.subheader("🔄 Reset global")

# Advertencia de peligro
st.warning("⚠️ Cuidado: Esta acción borrará todos los movimientos no guardados de la sesión actual.")

# Usar un expander para "esconder" el botón peligroso
with st.expander("Confirmar reseteo de simulación"):
    
    # Usar un color rojo para el botón
    if st.button("🔴 SÍ, BORRAR SIMULACIÓN", key="reset_confirm_button"):
        
        # La lógica de siempre
        st.session_state.df_sim = st.session_state.base_original.copy()
        st.session_state.df_sim["Movido"] = "No"
        st.session_state.mov_hist = []
        
        if "legajo_select" in st.session_state:
            del st.session_state["legajo_select"]
        
        # Limpiar el trigger de descarga
        if "trigger_download" in st.session_state:
            st.session_state.trigger_download = False
            st.session_state.download_data_b64 = None
            st.session_state.download_filename = None
        
        st.info("Se volvió al estado inicial.")
        st.rerun()


