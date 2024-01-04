# INSTRUÇÕES:

# Instale a biblioteca playwright no terminal com os comandos:
#1. pip install playwright
#2. playwright install

#Para executar o código, escreva no terminal:
# streamlit run carteira_aleatoria.py

#OBS: modelo assume risk-free = 0


### BIBILIOTECAS ###
import os
import time
import asyncio
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from playwright.async_api import async_playwright

### DASHBOARD ###
st.set_page_config(layout='wide')
st.title("Carteira Aleatória de Investimentos")

# Menu inicial
col1, col2, col3 = st.columns(3)

with col1:
    capital = st.number_input('◾ Qual será o capital inicial?',
                           min_value=100, max_value=1000000000, value=10000, step=100)
    capital_1 = float(capital)
    capital_2 = float(capital)


with col2:
    parametro = ('Índice Bovespa', 'Índice Valor', 'Índice Brasil 50')
    dropdown = st.selectbox('◾ Escolha seu índice para comparação: ', parametro)

    indice = []
    indice_yf = []
    if dropdown == 'Índice Bovespa':
        indice = 'ibov'
        indice_yf = '^BVSP'
    if dropdown == 'Índice Valor':
        indice = 'ivbx'
        indice_yf = '^IVBX'
    if dropdown == 'Índice Brasil 50':
        indice = 'ibxl'
        indice_yf = '^IBX50'


with col3:
    tempo = st.number_input('◾ Tempo de investimento (anos)', min_value=1, max_value=20, value=1, step=1)

if st.button("Carregar"):
    with st.spinner(text='Gerando carteiras'):
        time.sleep(5)

    ### DATAS ###
    # Data para a extração dos dados
    hoje = date.today()
    if hoje.isoweekday() == 6:
        data_dados = hoje + timedelta(days=2)
    elif hoje.isoweekday() == 7:
        data_dados = hoje + timedelta(days=1)
    else:
        data_dados = hoje

    # Data para as cotações
    d_inicio = hoje.replace(year=hoje.year - tempo)


    ### EXTRAÇÃO DOS DADOS ###
    arquivo = indice.upper() + f'DIA_' + data_dados.__format__('%d-%m-%y') + '.csv'

    if os.path.exists(arquivo):
        pass
    else:
        @st.cache_resource
        async def download(indice):
            async with async_playwright() as p:
                navegador = await p.chromium.launch()
                page = await navegador.new_page()
                await page.goto(f'https://sistemaswebb3-listados.b3.com.br/indexPage/day/{indice.upper()}?language=pt-br')

                async with page.expect_download() as download_info:
                    menu = await page.query_selector('#segment')
                    await menu.select_option('Setor de Atuação')
                    await page.locator(
                            'xpath=//*[@id="divContainerIframeB3"]/div/div[1]/form/div[2]/div/div[2]/div/div/div[1]/div[2]/p/a').click()

                download = await download_info.value
                path = await download.path()
                await download.save_as(download.suggested_filename)

                await navegador.close()

        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(download(indice))

    dados = pd.read_csv(arquivo, sep=';', decimal=',', encoding='windows-1252', engine='python')
    dados = dados.reset_index()
    tabela = dados[['Código', 'Ação', 'Setor']].iloc[:-2].set_index('Código')
    tabela = tabela.rename(columns={'Ação': 'Empresa'})
    tabela['Setor'] = tabela['Setor'].str.split('/').str.get(-1).sort_values(ascending=False)


    ### CARTEIRAS ALEATÓRIAS ###
    lista_ativos = []
    risco = []
    retorno = []
    peso = []
    sharpe = []
    numero_carteiras = 1000
    risk_free = 0

    # Seleção dos ativos
    while len(lista_ativos) < 5:
        ativo = np.random.choice(tabela.index)
        if ativo not in lista_ativos:
            lista_ativos.append(ativo)
            lista_ativos.sort()

    ativos = [papel + '.SA' for papel in lista_ativos]

    cotacao = yf.download(ativos, start=d_inicio)['Adj Close']

    for i in range(numero_carteiras):
        pesos_aleatorios = np.random.dirichlet(np.ones(len(lista_ativos)))
        pesos = np.round(pesos_aleatorios, 2)
        peso.append(pesos)

        carteira = np.log(cotacao / cotacao.shift(1)).fillna(0)
        média_carteira = carteira.mean()
        retorno_carteira = np.dot(média_carteira, pesos) * 252
        retorno.append(retorno_carteira)

        matriz_cov = carteira.cov()
        risco_portfolio = np.sqrt(np.dot(pesos, np.dot(matriz_cov, pesos))) * np.sqrt(252)
        risco.append(risco_portfolio)

        sharpe_ratio = (retorno_carteira - risk_free)/risco_portfolio
        sharpe.append(sharpe_ratio)

    portfolio_novo = pd.DataFrame()
    portfolio = {'Retorno': retorno,
                      'Risco': risco,
                      'Índice de Sharpe': sharpe,
                      'Pesos': peso
                      }
    portfolio_novo = pd.DataFrame(portfolio)

    # Maximização dos resultados
    # Índice de Sharpe
    max_sharpe = portfolio_novo['Índice de Sharpe'].max()
    carteira_sharpe = portfolio_novo[portfolio_novo['Índice de Sharpe'] == max_sharpe]

    # Retorno
    max_retorno = portfolio_novo['Retorno'].max()
    carteira_retorno = portfolio_novo[portfolio_novo['Retorno'] == max_retorno]


    ### CÁLCULO DO MONTANTE ###
    montante_sharpe = []
    montante_retorno = []

    preco = cotacao.iloc[0:]
    quantidade = round(capital/preco)

    # Montante carteira Índice de Sharpe
    quant_sharpe = quantidade * carteira_sharpe['Pesos'].iloc[0]
    PL_sharpe = preco * quant_sharpe
    total_sharpe = pd.DataFrame()
    total_sharpe['Retorno diário'] = (carteira * carteira_sharpe['Pesos'].iloc[0]).sum(axis=1)
    total_sharpe['Carteira'] = PL_sharpe.iloc[:].sum(axis=1).fillna(0)
    total_sharpe['Rendimento diário'] = total_sharpe['Carteira'] * total_sharpe['Retorno diário']
    for item in total_sharpe['Rendimento diário']:
        capital_1 += item
        montante_sharpe.append(capital_1)
    total_sharpe['Montante_Sharpe'] = montante_sharpe

    # Montante carteira Retorno
    quant_retorno = quantidade * carteira_retorno['Pesos'].iloc[0]
    PL_retorno = preco * quant_retorno
    total_retorno = pd.DataFrame()
    total_retorno['Retorno diário'] = (carteira * carteira_retorno['Pesos'].iloc[0]).sum(axis=1)
    total_retorno['Carteira'] = PL_retorno.iloc[:].sum(axis=1).fillna(0)
    total_retorno['Rendimento diário'] = total_retorno['Carteira'] * total_retorno['Retorno diário']
    for linha in total_retorno['Rendimento diário']:
        capital_2 += linha
        montante_retorno.append(capital_2)
    total_retorno['Montante_Retorno'] = montante_retorno


    ### BENCHMARK(ÍNDICE) ###
    cotacao_indice = yf.download(indice_yf, start=d_inicio)['Adj Close']
    retorno_indice = cotacao_indice.pct_change().fillna(0)


    ### COMPARAÇÃO DAS CARTEIRAS ###
    rentabilidade = pd.DataFrame()
    rentabilidade['Índice'] = (1 + retorno_indice).cumprod()
    rentabilidade['Carteira max Sharpe'] = (1 + total_sharpe['Retorno diário']).cumprod()
    rentabilidade['Carteira max Retorno'] = (1 + total_retorno['Retorno diário']).cumprod()


    ### MATRIZ DE CORRELAÇÃO ###
    matriz_correl = carteira.corr()


    ### DASHBOARD ###
    st.divider()
    st.subheader('Esta é a carteira selecionada para você:')
    st.write('**Ações escolhidas:**')
    st.table(tabela.loc[lista_ativos])
    col4, col5 = st.columns(2)
    st.write('')
    st.markdown('📊 **Backtesting das carteiras:**')
    with col4:
        st.write('💰 **Carteira 1: maximização do Índice de Sharpe**')
        st.table(carteira_sharpe)
    with col5:
        st.write('💰 **Carteira 2: maximização do retorno**')
        st.table(carteira_retorno)

    # Gráficos
    col6, col7 = st.columns(2)
    col8, col9 = st.columns(2)
    col10, col11 = st.columns(2)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=risco, y=retorno, mode='markers',
                                  marker=dict(color='cornflowerblue'),
                                  name='Carteiras',
                                  hovertemplate="Risco: %{x} <br> Retorno: %{y}</br>"))

    fig1.add_trace(go.Scatter(x=carteira_retorno['Risco'], y=carteira_retorno['Retorno'],
                                  mode='markers+text',
                                  marker=dict(color='black', symbol='pentagon-open', size=12),
                                  text=['Carteira max. retorno'],
                                  textposition="top center",
                                  name="Carteira max. retorno",
                                  hovertemplate="Risco: %{x} <br> Retorno: %{y}</br>"))

    fig1.add_trace(go.Scatter(x=carteira_sharpe['Risco'], y=carteira_sharpe['Retorno'],
                                  mode='markers+text',
                                  marker=dict(color='black', symbol='diamond-open', size=12),
                                  name="Carteira max. Sharpe",
                                  text=['Carteira max. Sharpe'],
                                  textposition="top center",
                                  hovertemplate="Risco: %{x} <br> Retorno: %{y}</br>"))

    fig1.update_layout(
                xaxis_title='Risco',
                yaxis_title='Retorno',
                title='Fronteira eficiente',
                showlegend=False)
    col6.plotly_chart(fig1)


    fig2 = px.line(rentabilidade,
                   title='Rentabilidade de cada carteira em comparação com o índice',
                   color_discrete_sequence=px.colors.qualitative.T10,
                   labels={'variable': 'Carteira', 'Date': 'Data', 'value': 'Retorno'})
    fig2.update_layout(legend=dict(yanchor='middle',
                                   y=0.9,
                                   xanchor='right',
                                   x=0.15))
    col7.plotly_chart(fig2)


    fig3 = px.line(total_sharpe, y='Montante_Sharpe', x=total_sharpe.index,
                   title='Evolução diário do Patrimônio, maximização do Sharpe',
                   color_discrete_sequence=px.colors.qualitative.Safe,
                   labels={'Date': 'Data', 'Montante_Sharpe': 'Capital'})
    col8.plotly_chart(fig3)


    fig4 = px.line(total_retorno, y='Montante_Retorno', x=total_retorno.index,
                   title='Evolução diária do Patrimônio, maximização do Retorno',
                   color_discrete_sequence=px.colors.qualitative.Pastel,
                   labels={'Date': 'Data', 'Montante_Retorno': 'Capital'})
    col9.plotly_chart(fig4)


    fig5 = px.imshow(matriz_correl, text_auto=True, color_continuous_scale='blues',
                     title="Correlação entre os ativos da carteira")
    col10.plotly_chart(fig5)


    ### DISCLAIMER ###
    st.divider()
    st.markdown('_**ATENÇÃO: Este projeto não constitui uma recomendação de investimento. '
                'As informações apresentadas são baseadas em fontes públicas, e podem não representar'
                ' todas as informações relevantes sobre o investimento. O mercado financeiro é um '
                'ambiente volátil e os retornos dos investimentos podem variar significativamente '
                'ao longo do tempo. Antes de investir em qualquer produto financeiro, é importante '
                'consultar um profissional de investimentos certificado.**_')
    st.write('**Produzido por:** Crislaine Balbino')
    st.write('**Versão:** 1.0.0')
else:
    st.write('')