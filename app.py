import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
try:
	import seaborn as sns
except Exception:
	sns = None
try:
	import matplotlib.pyplot as plt
except Exception:
	plt = None
from datetime import datetime
from io import StringIO, BytesIO
import pdfplumber



EXAMPLE_CSV = 'matricula_general.csv'


def read_csv_flexible(path_or_buffer):
	"""Intentar leer CSV con distintos encodings y separadores comunes."""
	# Ensure we can seek the buffer: if it's a stream, copy into BytesIO
	from io import BytesIO
	buf = path_or_buffer
	try:
		if hasattr(path_or_buffer, 'read'):
			data = path_or_buffer.read()
			buf = BytesIO(data)
	except Exception:
		buf = path_or_buffer

	# Try common separators and encodings
	for sep in [';', ',', '\t']:
		for enc in ['utf-8', 'latin-1', 'cp1252']:
			try:
				buf.seek(0)
			except Exception:
				pass
			try:
				df = pd.read_csv(buf, sep=sep, encoding=enc)
				if any('periodo' in str(c).lower() for c in df.columns):
					return df
			except Exception:
				continue

	# final attempt
	try:
		buf.seek(0)
	except Exception:
		pass
	return pd.read_csv(buf, sep=';', encoding='latin-1', engine='python')


def clean_matricula(df: pd.DataFrame) -> pd.DataFrame:
	# Work on a copy to avoid SettingWithCopyWarning
	df = df.copy()

	# Strip and normalize column names (keep originals too)
	original_columns = list(df.columns)
	df.columns = [str(c).strip() for c in df.columns]
	df.columns = [c.lower().replace(' ', '_') for c in df.columns]

	# Drop completely empty columns and unnamed columns
	df = df.loc[:, df.notna().any()]
	df = df.loc[:, ~df.columns.str.contains('^unnamed')]

	# Trim string values safely
	obj_cols = df.select_dtypes(include='object').columns
	for c in obj_cols:
		df.loc[:, c] = df[c].astype(str).str.strip()

	# Normalize gender (in-place on a string column is fine)
	if 'genero' in df.columns:
		df.loc[:, 'genero'] = df['genero'].str.lower().str.replace('fenemino', 'femenino', regex=False)

	# --- Fecha de nacimiento: crear columna fecha_de_nacimiento_dt en vez de sobrescribir ---
	fecha_col = None
	for cand in ['fecha_de_nacimiento', 'fecha_de_nacimiento_', 'fecha_de_nac', 'nacimiento', 'fecha']:
		if cand in df.columns:
			fecha_col = cand
			break

	if fecha_col:
		tmp = df[fecha_col].astype(str)
		try:
			tmp = tmp.str.replace(' ', '', regex=False)
		except Exception:
			tmp = tmp.apply(lambda x: str(x).replace(' ', ''))
		tmp = tmp.replace({'nan': '', 'None': '', 'NaT': ''})
		# Parse dates while suppressing noisy UserWarnings that can flood Streamlit logs
		import warnings
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			dates = pd.to_datetime(tmp, dayfirst=True, errors='coerce', infer_datetime_format=True)
		# If many values are NaT, try the opposite dayfirst setting and prefer non-null
		if dates.isna().sum() > (len(dates) * 0.5):
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				dates_alt = pd.to_datetime(tmp, dayfirst=False, errors='coerce', infer_datetime_format=True)
			# combine: prefer dates_alt where notna, else dates
			dates = dates_alt.combine_first(dates)
		df.loc[:, 'fecha_de_nacimiento_dt'] = dates

	# --- Edad: crear edad_numeric (no sobrescribir) ---
	if 'edad' in df.columns:
		df.loc[:, 'edad_numeric'] = pd.to_numeric(df['edad'], errors='coerce')
	else:
		if 'fecha_de_nacimiento_dt' in df.columns:
			df.loc[:, 'edad_numeric'] = (pd.Timestamp.now().year - df['fecha_de_nacimiento_dt'].dt.year)

	# --- Periodo escolar: crear periodo_escolar_clean en vez de sobrescribir ---
	periodo_candidates = [c for c in df.columns if 'periodo' in c]
	if periodo_candidates:
		pcol = periodo_candidates[0]
		df.loc[:, 'periodo_escolar_clean'] = pd.to_numeric(df[pcol].astype(str).str.extract(r'(\d{4})')[0], errors='coerce').astype('Int64')

	return df


def sidebar_filters(df):
	st.sidebar.header('Filtros')
	periodo_values = sorted(df['periodo_escolar_clean'].dropna().unique().tolist()) if 'periodo_escolar_clean' in df.columns else []
	carreras = sorted(df['carrera'].dropna().unique().tolist()) if 'carrera' in df.columns else []
	generos = sorted(df['genero'].dropna().unique().tolist()) if 'genero' in df.columns else []
	grados = sorted(df['grado'].dropna().unique().tolist()) if 'grado' in df.columns else []
	secciones = sorted(df['seccion'].dropna().unique().tolist()) if 'seccion' in df.columns else []
	jornadas = sorted(df['jornada'].dropna().unique().tolist()) if 'jornada' in df.columns else []

	selected_periodos = st.sidebar.multiselect('Periodo escolar', periodo_values, default=periodo_values)
	selected_carreras = st.sidebar.multiselect('Carrera', carreras, default=carreras)
	selected_generos = st.sidebar.multiselect('Género', generos, default=generos)
	selected_grados = st.sidebar.multiselect('Grado', grados, default=grados)
	selected_secciones = st.sidebar.multiselect('Sección', secciones, default=secciones)
	selected_jornadas = st.sidebar.multiselect('Jornada', jornadas, default=jornadas)

	mask = pd.Series(True, index=df.index)
	if 'periodo_escolar_clean' in df.columns and selected_periodos:
		mask = mask & df['periodo_escolar_clean'].isin(selected_periodos)
	if 'carrera' in df.columns and selected_carreras:
		mask = mask & df['carrera'].isin(selected_carreras)
	if 'genero' in df.columns and selected_generos:
		mask = mask & df['genero'].isin(selected_generos)
	if 'grado' in df.columns and selected_grados:
		mask = mask & df['grado'].isin(selected_grados)
	if 'seccion' in df.columns and selected_secciones:
		mask = mask & df['seccion'].isin(selected_secciones)
	if 'jornada' in df.columns and selected_jornadas:
		mask = mask & df['jornada'].isin(selected_jornadas)

	return df[mask]


def to_csv_bytes(df: pd.DataFrame) -> bytes:
	return df.to_csv(index=False).encode('utf-8')


def extract_tables_from_pdf(fileobj) -> list:
	"""Extrae tablas de un PDF usando pdfplumber.

	Devuelve una lista de DataFrames. Si no se encuentran tablas devuelve lista vacía.
	"""
	tables = []
	try:
		with pdfplumber.open(fileobj) as pdf:
			for page in pdf.pages:
				try:
					page_tables = page.extract_tables()
				except Exception:
					page_tables = []
				for t in page_tables:
					# t es una lista de filas; convertir a DataFrame manejando cabecera si existe
					try:
						df_t = pd.DataFrame(t)
						# Si la primera fila parece ser cabecera (texto no numérico), usarla
						if df_t.shape[0] > 1:
							header = df_t.iloc[0].astype(str).str.strip().tolist()
							df_t = df_t[1:]
							df_t.columns = header
						tables.append(df_t.reset_index(drop=True))
					except Exception:
						continue
	except Exception:
		return []
	return tables


def extract_text_from_pdf(fileobj) -> str:
	"""Extrae texto completo de un PDF y devuelve un string."""
	text_parts = []
	try:
		with pdfplumber.open(fileobj) as pdf:
			for page in pdf.pages:
				try:
					text_parts.append(page.extract_text() or '')
				except Exception:
					continue
	except Exception:
		return ''
	return '\n\n'.join(text_parts)


def parse_text_table(text: str, delimiter: str | None = None, header_row: int = 0) -> pd.DataFrame | None:
	"""Intentar convertir texto tabular (líneas) en DataFrame.

	Si `delimiter` es None intenta detectar entre ';', ',' o '\t'.
	`header_row` indica qué fila usar como cabecera (0 = primera fila).
	Devuelve None si no puede parsear.
	"""
	import csv
	lines = [l for l in text.splitlines() if l.strip()]
	if not lines:
		return None

	# determinar delimitador
	if delimiter is None:
		sample = lines[:5]
		scores = {
			';': sum(line.count(';') for line in sample),
			',': sum(line.count(',') for line in sample),
			'\t': sum(line.count('\t') for line in sample),
		}
		delimiter = max(scores, key=scores.get)
		if scores[delimiter] == 0:
			return None

	try:
		reader = csv.reader(lines, delimiter=delimiter)
		rows = list(reader)
		if not rows:
			return None
		df = pd.DataFrame(rows)
		# si header_row es válido y hay suficientes filas, usarla como cabecera
		if 0 <= header_row < df.shape[0]:
			header = df.iloc[header_row].astype(str).str.strip().tolist()
			df = df[header_row+1:]
			df.columns = header
		df = df.reset_index(drop=True)
		return df
	except Exception:
		return None


def get_palette(name: str, n: int):
	"""Retorna una lista de colores según la paleta solicitada."""
	import plotly.express as px
	name = (name or 'Plotly')
	if name == 'Plotly':
		return px.colors.qualitative.Plotly[:n]
	if name == 'Viridis':
		return px.colors.sequential.Viridis
	if name == 'Cividis':
		return px.colors.sequential.Cividis
	if name == 'Inferno':
		return px.colors.sequential.Inferno
	if name == 'Blues':
		return px.colors.sequential.Blues
	if name == 'Pastel':
		return px.colors.qualitative.Pastel[:n]
	return px.colors.qualitative.Plotly[:n]


def main():
	st.set_page_config(page_title='Instituto Marcio Rene Espinal', layout='wide')
	st.markdown(
		"""
		<div style='text-align:center'>
			<h1 style='margin:0;padding:0'>Instituto Marcio Rene Espinal</h1>
			<p style='margin:0;padding:0;font-size:18px;color:#555;'>Análisis de datos de matrícula y calificaciones</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

	# Diseño: paleta y estilos
	st.sidebar.header('Ajustes de visual')
	default_palette = st.sidebar.selectbox('Paleta por defecto', options=['Plotly','Viridis','Cividis','Inferno','Blues','Pastel'], index=0)
	# tiny custom CSS to improve spacing and header
	st.markdown(
		"""
		<style>
		body {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;}
		h1 {font-size: 30px;}
		.section-card {padding: 12px; border-radius: 6px; background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.06);}
		</style>
		""",
		unsafe_allow_html=True
	)

	use_example = st.sidebar.checkbox('Usar archivo de ejemplo (`matricula_general.csv`)', value=False)
	uploaded_file = st.sidebar.file_uploader('Subir archivo (CSV/Excel/PDF)', type=['csv', 'xlsx', 'xls', 'pdf'])

	df = None
	if uploaded_file is not None:
		# If uploaded, try to read according to extension
		fname = uploaded_file.name.lower()
		if fname.endswith('.csv'):
			df = read_csv_flexible(uploaded_file)
		elif fname.endswith(('.xls', '.xlsx')):
			df = pd.read_excel(uploaded_file)
		elif fname.endswith('.pdf'):
			# Extract tables from PDF using pdfplumber
			try:
				uploaded_file.seek(0)
			except Exception:
				pass
			pdf_bytes = uploaded_file.read()
			tables = extract_tables_from_pdf(BytesIO(pdf_bytes))
			if not tables:
				st.info('No se encontraron tablas estructuradas en el PDF. Puedes intentar extraer texto y convertirlo manualmente en una tabla.')
				text = extract_text_from_pdf(BytesIO(pdf_bytes))
				if text:
					st.subheader('Texto extraído (preview)')
					st.text_area('Texto extraído del PDF', value=text[:10000], height=250)
					st.markdown('---')
					st.subheader('Intentar convertir texto a tabla')
					delim_choice = st.selectbox('Delimitador', options=['Detectar', ';', ',', '\\t'], index=0)
					header_row = st.number_input('Fila de cabecera (indexada en 0)', min_value=0, max_value=20, value=0)
					delim = None if delim_choice == 'Detectar' else ('\t' if delim_choice == '\\t' else delim_choice)
					if st.button('Previsualizar tabla desde texto'):
						parsed = parse_text_table(text, delimiter=delim, header_row=int(header_row))
						if parsed is None:
							st.warning('No se pudo convertir el texto a una tabla con esos parámetros. Intenta otro delimitador o ajusta la fila de cabecera.')
						else:
							st.write('Preview de tabla parseada:')
							st.dataframe(parsed.head(50))
							if st.button('Usar esta tabla para el análisis'):
								df = parsed.copy()
								st.success(f'Tabla parseada cargada (filas: {len(df)})')
				else:
					st.warning('No se pudo extraer texto del PDF.')
			else:
				st.sidebar.write(f'Tablas encontradas: {len(tables)}')
				idx = st.sidebar.number_input('Selecciona índice de tabla', min_value=0, max_value=max(0, len(tables)-1), value=0)
				df = tables[int(idx)]
				st.success(f'Tabla {idx} cargada desde PDF (filas: {len(df)})')
		else:
			st.warning('Tipo de archivo no soportado. Sube CSV, XLSX o PDF.')
	elif use_example:
		try:
			df = read_csv_flexible(EXAMPLE_CSV)
		except Exception as e:
			st.error(f'No se pudo leer el archivo de ejemplo: {e}')

	if df is None:
		st.info('Carga un archivo CSV/XLSX o activa el ejemplo para ver la demo.')
		return

	df = clean_matricula(df)
	# --- KPIs ---
	st.markdown('')
	try:
		total_students = len(df)
		periods = sorted(df['periodo_escolar_clean'].dropna().unique().tolist()) if 'periodo_escolar_clean' in df.columns else []
		period_text = f"{min(periods)} - {max(periods)}" if periods else 'N/A'
		avg_age = None
		if 'edad_numeric' in df.columns:
			avg_age = int(df['edad_numeric'].dropna().mean()) if not df['edad_numeric'].dropna().empty else None
		pct_female = None
		if 'genero' in df.columns:
			vc = df['genero'].value_counts(normalize=True)
			pct_female = float(vc.get('femenino', vc.get('f', 0)) * 100) if not vc.empty else None
	except Exception:
		total_students = len(df)
		period_text = 'N/A'
		avg_age = None
		pct_female = None

	col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
	col_kpi1.metric('Total registros', total_students)
	col_kpi2.metric('Rango periodos', period_text)
	col_kpi3.metric('Edad media', avg_age if avg_age is not None else 'N/A')
	col_kpi4.metric('% Femenino (aprox.)', f"{pct_female:.1f}%" if pct_female is not None else 'N/A')

	with st.expander('Datos (vistazo)', expanded=False):
		st.dataframe(df.head(200))

	# Allow download of cleaned data
	csv_bytes = to_csv_bytes(df)
	st.download_button('Descargar CSV limpio', data=csv_bytes, file_name='matricula_clean.csv', mime='text/csv')

	# Apply filters
	filtered = sidebar_filters(df)

	st.subheader('Resumen de datos de matricula')
	st.write(f'Total registros: {len(filtered)}')

	# Charts
	col1, col2 = st.columns(2)
	# Controls for left chart (periodo)
	with col1:
		if 'periodo_escolar_clean' in filtered.columns:
			tipo1 = st.selectbox('Tipo de gráfica (periodo)', options=['Histograma','Linea','Barras','Pastel','Donut','Pareto'], index=0)
			pal1 = st.selectbox('Paleta (periodo)', options=['Plotly','Viridis','Cividis','Inferno','Blues','Pastel'], index=0)
			# prepare counts and palette
			counts_p = filtered.groupby('periodo_escolar_clean').size().reset_index(name='count')
			counts_p = counts_p.sort_values('periodo_escolar_clean')
			pal_colors1 = get_palette(pal1, max(3, counts_p.shape[0]))
			if tipo1 == 'Histograma':
				fig1 = px.histogram(filtered, x='periodo_escolar_clean', title='Matricula por periodo escolar', color_discrete_sequence=pal_colors1)
			elif tipo1 == 'Linea':
				fig1 = px.line(counts_p, x='periodo_escolar_clean', y='count', title='Matricula por periodo escolar', color_discrete_sequence=pal_colors1)
				fig1.update_traces(mode='lines+markers')
			elif tipo1 == 'Barras':
				fig1 = px.bar(counts_p, x='periodo_escolar_clean', y='count', title='Matricula por periodo escolar', color='periodo_escolar_clean', color_discrete_sequence=pal_colors1)
			elif tipo1 in ('Pastel','Donut'):
				# pie/donut from counts
				fig1 = px.pie(counts_p, names='periodo_escolar_clean', values='count', title='Matricula por periodo escolar', color_discrete_sequence=pal_colors1)
				if tipo1 == 'Donut':
					fig1.update_traces(hole=0.4)
			else:  # Pareto
				# Pareto: bars (desc) + cumulative percent line
				pareto = counts_p.sort_values('count', ascending=False).reset_index(drop=True)
				pareto['cum_pct'] = pareto['count'].cumsum() / pareto['count'].sum() * 100
				fig1 = make_subplots(specs=[[{"secondary_y": True}]])
				fig1.add_trace(go.Bar(x=pareto['periodo_escolar_clean'].astype(str), y=pareto['count'], name='Count', marker_color=pal_colors1[:len(pareto)]), secondary_y=False)
				fig1.add_trace(go.Scatter(x=pareto['periodo_escolar_clean'].astype(str), y=pareto['cum_pct'], name='Cumulative %', mode='lines+markers', marker=dict(color='black')), secondary_y=True)
				fig1.update_yaxes(title_text='Count', secondary_y=False)
				fig1.update_yaxes(title_text='Cumulative %', secondary_y=True, range=[0,100])
				fig1.update_layout(title='Pareto - Matricula por periodo escolar')
			# common tweaks
			try:
				fig1.update_xaxes(title_text='periodo escolar')
			except Exception:
				pass
			st.plotly_chart(fig1, use_container_width=True)
	# Controls for right chart (carrera)
	with col2:
		if 'carrera' in filtered.columns:
			tipo2 = st.selectbox('Tipo de gráfica (carrera)', options=['Histograma','Linea','Barras','Pastel','Donut','Pareto'], index=0)
			pal2 = st.selectbox('Paleta (carrera)', options=['Plotly','Viridis','Cividis','Inferno','Blues','Pastel'], index=0)
			c_counts = filtered['carrera'].value_counts().reset_index()
			c_counts.columns = ['carrera', 'count']
			pal_colors2 = get_palette(pal2, max(3, c_counts.shape[0]))
			if tipo2 == 'Histograma':
				fig2 = px.histogram(filtered, x='carrera', title='Matriculados por carrera', color_discrete_sequence=pal_colors2)
			elif tipo2 == 'Linea':
				# line over ordered categories
				c_line = c_counts.sort_values('carrera').reset_index(drop=True)
				fig2 = px.line(c_line, x='carrera', y='count', title='Matriculados por carrera', color_discrete_sequence=pal_colors2)
				fig2.update_traces(mode='lines+markers')
			elif tipo2 == 'Barras':
				fig2 = px.bar(c_counts, x='carrera', y='count', title='Matriculados por carrera', color='carrera', color_discrete_sequence=pal_colors2)
			elif tipo2 in ('Pastel','Donut'):
				fig2 = px.pie(c_counts, names='carrera', values='count', title='Matriculados por carrera', color_discrete_sequence=pal_colors2)
				if tipo2 == 'Donut':
					fig2.update_traces(hole=0.4)
			else:  # Pareto
				pareto_c = c_counts.sort_values('count', ascending=False).reset_index(drop=True)
				pareto_c['cum_pct'] = pareto_c['count'].cumsum() / pareto_c['count'].sum() * 100
				fig2 = make_subplots(specs=[[{"secondary_y": True}]])
				fig2.add_trace(go.Bar(x=pareto_c['carrera'].astype(str), y=pareto_c['count'], name='Count', marker_color=pal_colors2[:len(pareto_c)]), secondary_y=False)
				fig2.add_trace(go.Scatter(x=pareto_c['carrera'].astype(str), y=pareto_c['cum_pct'], name='Cumulative %', mode='lines+markers', marker=dict(color='black')), secondary_y=True)
				fig2.update_yaxes(title_text='Count', secondary_y=False)
				fig2.update_yaxes(title_text='Cumulative %', secondary_y=True, range=[0,100])
				fig2.update_layout(title='Pareto - Matriculados por carrera')
			st.plotly_chart(fig2, use_container_width=True)

	# Distribuciones: género y edad colocadas lado a lado para armonía visual
	col_g, col_e = st.columns(2)
	with col_g:
		if 'genero' in filtered.columns:
			st.subheader('Distribución por género')
			tipo_g = st.selectbox('Tipo de gráfica (género)', options=['Histograma','Linea','Barras','Pastel','Donut','Pareto'], index=2)
			pal_g = st.selectbox('Paleta (género)', options=['Plotly','Viridis','Cividis','Inferno','Blues','Pastel'], index=0)
			gen_count = filtered['genero'].value_counts().reset_index()
			gen_count.columns = ['genero', 'count']
			gen_count['pct'] = (gen_count['count'] / gen_count['count'].sum() * 100).round(2)
			pal_colors_g = get_palette(pal_g, max(3, gen_count.shape[0]))
			if tipo_g == 'Histograma':
				fig3 = px.histogram(filtered, x='genero', title='Distribución por género', color_discrete_sequence=pal_colors_g)
			elif tipo_g == 'Linea':
				g_line = gen_count.sort_values('genero')
				fig3 = px.line(g_line, x='genero', y='count', title='Distribución por género', color_discrete_sequence=pal_colors_g)
				fig3.update_traces(mode='lines+markers')
			elif tipo_g == 'Barras':
				fig3 = px.bar(gen_count, x='genero', y='count', title='Distribución por género', color='genero', color_discrete_sequence=pal_colors_g)
			elif tipo_g in ('Pastel','Donut'):
				fig3 = px.pie(gen_count, names='genero', values='count', title='Distribución por género', color_discrete_sequence=pal_colors_g)
				if tipo_g == 'Donut':
					fig3.update_traces(hole=0.4)
			else:  # Pareto
				pareto_g = gen_count.sort_values('count', ascending=False).reset_index(drop=True)
				pareto_g['cum_pct'] = pareto_g['count'].cumsum() / pareto_g['count'].sum() * 100
				fig3 = make_subplots(specs=[[{"secondary_y": True}]])
				fig3.add_trace(go.Bar(x=pareto_g['genero'].astype(str), y=pareto_g['count'], name='Count', marker_color=pal_colors_g[:len(pareto_g)]), secondary_y=False)
				fig3.add_trace(go.Scatter(x=pareto_g['genero'].astype(str), y=pareto_g['cum_pct'], name='Cumulative %', mode='lines+markers', marker=dict(color='black')), secondary_y=True)
				fig3.update_yaxes(title_text='Count', secondary_y=False)
				fig3.update_yaxes(title_text='Cumulative %', secondary_y=True, range=[0,100])
				fig3.update_layout(title='Pareto - Distribución por género')
			st.plotly_chart(fig3, use_container_width=True)
	with col_e:
		if 'edad_numeric' in filtered.columns:
			st.subheader('Distribución de edad')
			tipo_e = st.selectbox('Tipo de gráfica (edad)', options=['Histograma','Linea','Barras','Pastel','Donut','Pareto'], index=0)
			pal_e = st.selectbox('Paleta (edad)', options=['Plotly','Viridis','Cividis','Inferno','Blues','Pastel'], index=0)
			pal_colors_e = get_palette(pal_e, 6)
			if tipo_e == 'Histograma':
				fig4 = px.histogram(filtered, x='edad_numeric', nbins=20, title='Distribución de edad', color_discrete_sequence=pal_colors_e)
			elif tipo_e == 'Linea':
				age_counts = filtered['edad_numeric'].dropna().astype(int).value_counts().reset_index()
				age_counts.columns = ['edad', 'count']
				age_counts = age_counts.sort_values('edad')
				fig4 = px.line(age_counts, x='edad', y='count', title='Distribución de edad', color_discrete_sequence=pal_colors_e)
				fig4.update_traces(mode='lines+markers')
			elif tipo_e == 'Barras':
				age_counts = filtered['edad_numeric'].dropna().astype(int).value_counts().reset_index()
				age_counts.columns = ['edad', 'count']
				age_counts = age_counts.sort_values('edad')
				fig4 = px.bar(age_counts, x='edad', y='count', title='Distribución de edad', color='edad', color_discrete_sequence=pal_colors_e)
			elif tipo_e in ('Pastel','Donut'):
				# use age buckets for pie
				age_bins = pd.cut(filtered['edad_numeric'].dropna(), bins=6)
				age_df = age_bins.value_counts().reset_index()
				age_df.columns = ['edad_range', 'count']
				# Convert Interval objects to strings so Plotly can serialize them
				age_df['edad_range'] = age_df['edad_range'].astype(str)
				fig4 = px.pie(age_df, names='edad_range', values='count', title='Distribución de edad (rangos)', color_discrete_sequence=pal_colors_e)
				if tipo_e == 'Donut':
					fig4.update_traces(hole=0.4)
			else:  # Pareto for age
				age_counts = filtered['edad_numeric'].dropna().astype(int).value_counts().reset_index()
				age_counts.columns = ['edad', 'count']
				pareto_a = age_counts.sort_values('count', ascending=False).reset_index(drop=True)
				pareto_a['cum_pct'] = pareto_a['count'].cumsum() / pareto_a['count'].sum() * 100
				fig4 = make_subplots(specs=[[{"secondary_y": True}]])
				fig4.add_trace(go.Bar(x=pareto_a['edad'].astype(str), y=pareto_a['count'], name='Count', marker_color=pal_colors_e[:len(pareto_a)]), secondary_y=False)
				fig4.add_trace(go.Scatter(x=pareto_a['edad'].astype(str), y=pareto_a['cum_pct'], name='Cumulative %', mode='lines+markers', marker=dict(color='black')), secondary_y=True)
				fig4.update_yaxes(title_text='Count', secondary_y=False)
				fig4.update_yaxes(title_text='Cumulative %', secondary_y=True, range=[0,100])
				fig4.update_layout(title='Pareto - Distribución de edad')
			st.plotly_chart(fig4, use_container_width=True)

	# --- Evolución y tendencias: Evolución de la matrícula ---
	st.markdown('---')
	st.subheader('Evolución de la matrícula')
	if 'periodo_escolar_clean' in filtered.columns:
		# opciones de agrupación
		group_opts = ['Ninguno'] + [c for c in ['carrera', 'grado', 'seccion', 'jornada', 'genero'] if c in filtered.columns]
		group_by = st.selectbox('Agrupar por', options=group_opts, index=0)
		show_pct_change = st.checkbox('Mostrar cambio año-a-año (%)', value=True)
		show_cumulative = st.checkbox('Mostrar acumulado', value=False)
		# calcular conteos
		if group_by == 'Ninguno':
			counts = filtered.groupby('periodo_escolar_clean').size().reset_index(name='count').sort_values('periodo_escolar_clean')
			fig_ev = px.line(counts, x='periodo_escolar_clean', y='count', title='Evolución de la matrícula (total)')
			st.plotly_chart(fig_ev, use_container_width=True)
			if show_pct_change:
				counts = counts.sort_values('periodo_escolar_clean')
				counts['pct_change'] = counts['count'].pct_change().fillna(0) * 100
				fig_pc = px.bar(counts, x='periodo_escolar_clean', y='pct_change', title='Cambio porcentual año-a-año (%)')
				st.plotly_chart(fig_pc, use_container_width=True)
			if show_cumulative:
				counts['cumulative'] = counts['count'].cumsum()
				fig_cum = px.line(counts, x='periodo_escolar_clean', y='cumulative', title='Evolución acumulada de la matrícula')
				st.plotly_chart(fig_cum, use_container_width=True)
		else:
			# agrupado por otra columna
			grp = filtered.groupby(['periodo_escolar_clean', group_by]).size().reset_index(name='count')
			grp = grp.sort_values('periodo_escolar_clean')
			fig_ev2 = px.line(grp, x='periodo_escolar_clean', y='count', color=group_by, title=f'Evolución por {group_by}')
			st.plotly_chart(fig_ev2, use_container_width=True)
			if show_pct_change:
				# calcular pct change por grupo
				pct_df = grp.copy()
				pct_df = pct_df.sort_values([group_by, 'periodo_escolar_clean'])
				pct_df['pct_change'] = pct_df.groupby(group_by)['count'].pct_change().fillna(0) * 100
				fig_pct = px.bar(pct_df, x='periodo_escolar_clean', y='pct_change', color=group_by, title='Cambio porcentual año-a-año por grupo')
				st.plotly_chart(fig_pct, use_container_width=True)
			if show_cumulative:
				cum_df = grp.copy()
				cum_df['cumulative'] = cum_df.groupby(group_by)['count'].cumsum()
				fig_cum2 = px.line(cum_df, x='periodo_escolar_clean', y='cumulative', color=group_by, title='Evolución acumulada por grupo')
				st.plotly_chart(fig_cum2, use_container_width=True)
	else:
		st.info('No hay columna `periodo_escolar_clean` para mostrar la evolución. Usa el limpiador para extraer el periodo.')

	# --- Tasa de finalización vs abandono ---
	st.markdown('---')
	st.subheader('Tasa de finalización vs abandono')
	# detectar columnas candidatas para inicio/termino
	col_options = list(filtered.columns)
	start_cands = [c for c in col_options if any(k in c.lower() for k in ['inicio', 'inscrip', 'ingreso', 'matricu', 'inscri'])]
	end_cands = [c for c in col_options if any(k in c.lower() for k in ['termino', 'final', 'finaliz', 'aband', 'egreso', 'retir'])]
	# Mostrar los selectboxes lado a lado en columnas más angostas para diseño compacto
	sc1, sc2 = st.columns([1,1])
	with sc1:
		start_col = st.selectbox('Inicio (columna)', options=(start_cands or col_options), index=0)
	with sc2:
		end_col = st.selectbox('Término (columna)', options=(end_cands or col_options), index=min(0, len(end_cands)-1) if end_cands else 0)
	# helper para interpretar valores "verdaderos"
	def interpret_truthy(s: pd.Series) -> pd.Series:
		# treat as True when value indicates yes/1/true or non-empty meaningful
		if s is None:
			return pd.Series([False]*0)
		ser = s.astype(str).str.strip().str.lower().replace({'nan':'', 'none':''})
		truthy = ser.isin(['1','1.0','si','sí','s','true','t','yes','y']) | (~ser.isin(['', '0', '0.0', 'no', 'false', 'n']))
		# If column seems numeric, treat >0 as True
		try:
			num = pd.to_numeric(s, errors='coerce')
			truthy = truthy | (num > 0)
		except Exception:
			pass
		return truthy

	# Guardas: columnas deben existir
	if start_col not in filtered.columns or end_col not in filtered.columns:
		st.warning('Selecciona columnas válidas para inicio y término.')
	else:
		status_df = filtered[[start_col, end_col] + (['periodo_escolar_clean'] if 'periodo_escolar_clean' in filtered.columns else [])].copy()
		status_df['start_bool'] = interpret_truthy(status_df[start_col])
		status_df['end_bool'] = interpret_truthy(status_df[end_col])
		# definir finished: start True and end True
		status_df['finished'] = status_df['start_bool'] & status_df['end_bool']
		status_df['not_finished'] = status_df['start_bool'] & (~status_df['end_bool'])
		# agrupaciones por periodo
		if 'periodo_escolar_clean' in status_df.columns:
			grp = status_df.groupby('periodo_escolar_clean').agg(started=('start_bool', 'sum'), finished=('finished', 'sum'), not_finished=('not_finished', 'sum'))
		else:
			grp = pd.DataFrame({'started':[status_df['start_bool'].sum()], 'finished':[status_df['finished'].sum()], 'not_finished':[status_df['not_finished'].sum()]}, index=['all'])
		grp = grp.reset_index()
		# porcentajes
		grp['pct_finished'] = (grp['finished'] / grp['started']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
		grp['pct_not_finished'] = (grp['not_finished'] / grp['started']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
		st.write('Tabla resumen por periodo:')
		st.dataframe(grp)
		# Graficas: lineas de started vs finished
		if 'periodo_escolar_clean' in grp.columns:
			fig_sf = go.Figure()
			fig_sf.add_trace(go.Scatter(x=grp['periodo_escolar_clean'], y=grp['started'], mode='lines+markers', name='Iniciaron'))
			fig_sf.add_trace(go.Scatter(x=grp['periodo_escolar_clean'], y=grp['finished'], mode='lines+markers', name='Terminaron'))
			fig_sf.update_layout(title='Iniciaron vs Terminaron por periodo', xaxis_title='Periodo escolar', yaxis_title='Cantidad')
			st.plotly_chart(fig_sf, use_container_width=True)
			# stacked percent bar
			pct_df = grp[['periodo_escolar_clean','pct_finished','pct_not_finished']].melt(id_vars=['periodo_escolar_clean'], value_vars=['pct_finished','pct_not_finished'], var_name='metric', value_name='pct')
			pct_df['metric'] = pct_df['metric'].map({'pct_finished':'Finalizaron (%)','pct_not_finished':'No finalizaron (%)'})
			fig_pct = px.bar(pct_df, x='periodo_escolar_clean', y='pct', color='metric', title='Porcentaje que finalizaron vs no finalizaron por periodo', barmode='stack')
			st.plotly_chart(fig_pct, use_container_width=True)
		# mostrar periodos con mayor tasa de abandono
		top_n_ab = st.number_input('Top N periodos por tasa de abandono', min_value=1, max_value=50, value=5)
		top_ab = grp.sort_values('pct_not_finished', ascending=False).head(int(top_n_ab))
		st.subheader('Periodos con mayor tasa de abandono')
		st.dataframe(top_ab)

	# --- Grado y Sección: cantidad de estudiantes ---
	with st.expander('Cantidad de estudiantes por grado y sección', expanded=False):
		st.write('Visualizaciones y resumen por grado y sección')
	# Opciones de visualización
	chart_choice = st.selectbox('Tipo de visualización', options=['Barra por grado', 'Barra por sección', 'Heatmap Grado x Sección', 'Barras apiladas (grado->sección)'], index=0)
	palette = st.selectbox('Paleta de colores', options=['Plotly', 'Viridis', 'Cividis', 'Inferno', 'Blues', 'Pastel'], index=0)
	top_n = st.number_input('Top N elementos para mostrar (0 = todos)', min_value=0, max_value=50, value=0)
	show_pct_gs = st.checkbox('Mostrar porcentajes en barras', value=False)

	# Usar la función `get_palette` definida a nivel de módulo

	# Compute grouped counts
	if 'grado' in filtered.columns and 'seccion' in filtered.columns:
		group = filtered.groupby(['grado', 'seccion']).size().reset_index(name='count')
		# total por grado y por seccion
		total_grado = group.groupby('grado', as_index=False)['count'].sum().sort_values('count', ascending=False)
		total_seccion = group.groupby('seccion', as_index=False)['count'].sum().sort_values('count', ascending=False)

		# Apply top_n if requested
		if top_n and top_n > 0:
			tg = total_grado.head(top_n)
			ts = total_seccion.head(top_n)
		else:
			tg = total_grado
			ts = total_seccion

		pal = get_palette(palette, max(len(tg), len(ts), 3))

		if chart_choice == 'Barra por grado':
			fig = px.bar(tg, x='grado', y='count', title='Cantidad por grado', color='grado', color_discrete_sequence=pal)
			if show_pct_gs:
				fig.update_traces(text=[f"{v} ({(v/tg['count'].sum()*100):.1f}%)" for v in tg['count']])
			st.plotly_chart(fig, use_container_width=True)
		elif chart_choice == 'Barra por sección':
			fig = px.bar(ts, x='seccion', y='count', title='Cantidad por sección', color='seccion', color_discrete_sequence=pal)
			if show_pct_gs:
				fig.update_traces(text=[f"{v} ({(v/ts['count'].sum()*100):.1f}%)" for v in ts['count']])
			st.plotly_chart(fig, use_container_width=True)
		elif chart_choice == 'Heatmap Grado x Sección':
			pivot = group.pivot_table(index='grado', columns='seccion', values='count', fill_value=0)
			fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(), colorscale=palette.lower() if palette in ['Viridis','Cividis','Inferno','Blues'] else 'Viridis'))
			fig.update_layout(title='Heatmap: Grado x Sección', xaxis_title='Sección', yaxis_title='Grado')
			st.plotly_chart(fig, use_container_width=True)
		else:
			# barras apiladas: grado en x, secciones apiladas
			stack_df = group.pivot_table(index='grado', columns='seccion', values='count', fill_value=0)
			stack_df = stack_df.reset_index()
			fig = go.Figure()
			colors = get_palette(palette, len(stack_df.columns)-1)
			for i, col in enumerate(stack_df.columns[1:]):
				fig.add_trace(go.Bar(x=stack_df['grado'], y=stack_df[col], name=str(col), marker_color=colors[i % len(colors)]))
			fig.update_layout(barmode='stack', title='Barras apiladas: Grado por Sección', xaxis_title='Grado')
			st.plotly_chart(fig, use_container_width=True)

		# Mostrar grados y secciones con mayor/menor cantidad
		st.markdown('---')
		st.subheader('Grados con mayor y menor cantidad de estudiantes')
		if not total_grado.empty:
			top_grade = total_grado.iloc[0]
			bottom_grade = total_grado.iloc[-1]
			st.write(f"Mayor: {top_grade['grado']} — {top_grade['count']} estudiantes")
			st.write(f"Menor: {bottom_grade['grado']} — {bottom_grade['count']} estudiantes")

		st.subheader('Secciones con mayor y menor cantidad de estudiantes')
		if not total_seccion.empty:
			top_section = total_seccion.iloc[0]
			bottom_section = total_seccion.iloc[-1]
			st.write(f"Mayor: {top_section['seccion']} — {top_section['count']} estudiantes")
			st.write(f"Menor: {bottom_section['seccion']} — {bottom_section['count']} estudiantes")
	else:
		st.info('No hay columnas `grado` y `seccion` suficientes para este análisis.')

	# --- Distribución por carrera ---
	st.markdown('---')
	with st.expander('Distribución por carrera', expanded=False):
		if 'carrera' in filtered.columns:
			c_chart_type = st.selectbox('Tipo de gráfico (carrera)', options=['Barras', 'Pastel', 'Barras apiladas por género', 'Barras apiladas por jornada'], index=0)
			c_palette = st.selectbox('Paleta (carrera)', options=['Plotly','Viridis','Cividis','Pastel','Blues'], index=0)
			top_n_c = st.number_input('Top N carreras (0 = todos)', min_value=0, max_value=100, value=0)

			c_counts = filtered['carrera'].value_counts().reset_index()
			c_counts.columns = ['carrera', 'count']
			if top_n_c and top_n_c > 0:
				c_counts = c_counts.head(top_n_c)

			pal_c = get_palette(c_palette, max(len(c_counts), 3))

			if c_chart_type == 'Barras':
				figc = px.bar(c_counts, x='carrera', y='count', title='Cantidad por carrera', color='carrera', color_discrete_sequence=pal_c)
				st.plotly_chart(figc, use_container_width=True)
			elif c_chart_type == 'Pastel':
				figc = go.Figure(data=[go.Pie(labels=c_counts['carrera'], values=c_counts['count'], textinfo='label+percent')])
				figc.update_layout(title='Distribución por carrera')
				st.plotly_chart(figc, use_container_width=True)
			elif c_chart_type == 'Barras apiladas por género' and 'genero' in filtered.columns:
				cross = filtered.groupby(['carrera','genero']).size().reset_index(name='count')
				pivot = cross.pivot(index='carrera', columns='genero', values='count').fillna(0)
				figc = go.Figure()
				colors = get_palette(c_palette, len(pivot.columns))
				for i, col in enumerate(pivot.columns):
					figc.add_trace(go.Bar(x=pivot.index, y=pivot[col], name=str(col), marker_color=colors[i % len(colors)]))
				figc.update_layout(barmode='stack', title='Carrera x Género', xaxis_title='Carrera')
				st.plotly_chart(figc, use_container_width=True)
			elif c_chart_type == 'Barras apiladas por jornada' and 'jornada' in filtered.columns:
				cross = filtered.groupby(['carrera','jornada']).size().reset_index(name='count')
				pivot = cross.pivot(index='carrera', columns='jornada', values='count').fillna(0)
				figc = go.Figure()
				colors = get_palette(c_palette, len(pivot.columns))
				for i, col in enumerate(pivot.columns):
					figc.add_trace(go.Bar(x=pivot.index, y=pivot[col], name=str(col), marker_color=colors[i % len(colors)]))
				figc.update_layout(barmode='stack', title='Carrera x Jornada', xaxis_title='Carrera')
				st.plotly_chart(figc, use_container_width=True)
		else:
			st.info('No hay columna `carrera` en el dataset filtrado.')

	# --- Distribución por zona/colonia ---
	st.markdown('---')
	# Trabajar sobre copia fuera del expander para evitar reasignaciones problemáticas
	filtered_for_zona = filtered.copy()
	with st.expander('Distribución por zona / colonia', expanded=False):
		# detectar columnas candidatas (usar lower para mayor robustez)
		z_col_candidates = [c for c in filtered_for_zona.columns if 'zona' in c.lower() or 'colonia' in c.lower() or 'sector' in c.lower()]
		# asegurar variable inicializada para evitar UnboundLocalError
		if not isinstance(z_col_candidates, list):
			z_col_candidates = []
		if z_col_candidates:
			zcol = st.selectbox('Columna de zona', options=z_col_candidates, index=0)
			z_counts = filtered_for_zona[zcol].value_counts().reset_index()
			z_counts.columns = [zcol, 'count']
			top_z = st.number_input('Top N zonas (0 = todos)', min_value=0, max_value=100, value=10)
			if top_z and top_z > 0:
				z_counts = z_counts.head(top_z)
			pal_z = get_palette('Plotly', max(len(z_counts), 3))
			figz = px.bar(z_counts, x=zcol, y='count', title=f'Cantidad por {zcol}', color=zcol, color_discrete_sequence=pal_z)
			st.plotly_chart(figz, use_container_width=True)

			# Mapa si existen lat/lon
			lat_cols = [c for c in filtered_for_zona.columns if 'lat' in c.lower()]
			lon_cols = [c for c in filtered_for_zona.columns if 'lon' in c.lower() or 'long' in c.lower()]
			if lat_cols and lon_cols:
				latc = lat_cols[0]
				lonc = lon_cols[0]
				map_df = filtered_for_zona.dropna(subset=[latc, lonc])
				if not map_df.empty:
					st.subheader('Mapa de procedencia (lat/lon)')
					figmap = px.scatter_mapbox(map_df, lat=latc, lon=lonc, hover_name=zcol, color=zcol, size_max=10, zoom=10)
					figmap.update_layout(mapbox_style='open-street-map')
					st.plotly_chart(figmap, use_container_width=True)
		else:
			st.info('No se detectó una columna de zona/colonia en el dataset.')


if __name__ == '__main__':
	main()


