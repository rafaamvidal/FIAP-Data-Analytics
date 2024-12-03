import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Configurando o menu lateral
menu = st.sidebar.radio("Menu", ["O Desafio", "Introdução", "Dados", "Modelos","Dashboard","Conclusão"])

# Exibe conteúdo com base na opção selecionada
if menu == "O Desafio":

    # Título da aplicação
    st.title("Análise do Mercado de Petróleo Brent: Uma Visão Estratégica")

    # Adiciona uma imagem a partir de uma URL
    st.image(
        "https://www.quimicabrasileira.com.br/wp-content/uploads/2024/06/worker-oil-rig-sunset-created-with-generative-ai-technology-scaled-thegem-blog-default.jpg",
        use_container_width=True,
    )
    st.subheader("Projeto: Pós Tech Alura/Fiap - Tech Challenge Fase 4")
    st.write("Data: Dez/2024")
    st.markdown('<h2 style="color:#e61859;">Integrantes do Projeto</h2>', unsafe_allow_html=True)
    st.write("**Rafael Morais Vidal RM 354846**")
    st.write("**Rafael Lopes Tanaka RM 356096**")
    st.write("**Rodrigo Kenji Rossetti Inonhe RM 354906**")
    st.write("**Lucas Morikawa Giovanini RM 355007**")

    st.markdown('<h2 style="color:#e61859;">O Desafio</h2>', unsafe_allow_html=True)
    st.write(
        """Um grande cliente do segmento pediu para que a consultoria
              desenvolvesse um dashboard interativo para gerar insights relevantes para
              tomada de decisão. Além disso, solicitaram que fosse desenvolvido um modelo
              de Machine Learning para fazer o forecasting do preço do petróleo."""
    )
    st.markdown('<h2 style="color:#e61859;">O Objetivo</h2>', unsafe_allow_html=True)
    st.write("""• Criar um dashboard interativo com ferramentas à sua escolha.""")
    st.write(
        """• Seu dashboard deve fazer parte de um storytelling que traga insights
                relevantes sobre a variação do preço do petróleo, como situações
                geopolíticas, crises econômicas, demanda global por energia e etc. Isso
                pode te ajudar com seu modelo. É obrigatório que você traga pelo menos
                4 (quatro) insights neste desafio."""
    )
    st.write(
        """• Criar um modelo de Machine Learning que faça a previsão do preço do
                petróleo diariamente (lembre-se de time series). Esse modelo deve estar
                contemplado em seu storytelling e deve conter o código que você
                trabalhou, analisando as performances do modelo."""
    )
    st.write(
        """• Criar um plano para fazer o deploy em produção do modelo, com as
                ferramentas que são necessárias."""
    )
    st.write("""• Fazer um MVP do modelo em produção utilizando o Streamlit.""")


elif menu == "Introdução":
  

  # Título da aplicação
  st.markdown('<h2 style="color:#e61859;">Introdução</h2>', unsafe_allow_html=True)

  # Adiciona uma imagem a partir de uma URL
  st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ6Y-Kf4yHN_KBKMfZejnLY9Hj6t0fWovzDEEYJim4drD65m_fbTySHwE8TaAMkumEYVmw&usqp=CAU",use_container_width=True,)
  
  st.write("""Imagine o mercado de petróleo Brent, cuja cotação se tornou um dos principais indicadores econômicos globais, 
           com seus altos e baixos sendo influenciados por fatores como conflitos geopolíticos, oscilações na oferta e demanda, 
           e políticas energéticas globais. Este mercado representa mais do que um simples recurso energético, ele é o termômetro da economia mundial, 
           influenciando desde o custo do transporte até as cadeias produtivas.""")

  st.write("""Nos últimos anos, a volatilidade do mercado de petróleo aumentou, exigindo análises mais profundas para a tomada de decisão. 
           O petróleo Brent, produzido no Mar do Norte, é amplamente utilizado como benchmark para o preço internacional do petróleo e, por isso, 
           compreender suas oscilações é crucial para investidores, indústrias e formuladores de políticas públicas.""")

  st.write("""Este estudo busca não apenas explorar os dados históricos de preços do petróleo Brent disponíveis, mas também responder perguntas-chave:""")
  
  st.write("""• Quais são os padrões históricos do mercado?""")
  
  st.write("""• O que impulsiona as flutuações de preço""")
   
  st.write("""• E como é possível prever as tendências futuras?""")
  
  st.write("""Vamos mergulhar nos dados e trazer à tona insights que guiarão estratégias para os próximos anos""")

  # Título da aplicação
  st.markdown('<h2 style="color:#e61859;">Importância dos Dados</h2>', unsafe_allow_html=True)

  # Adiciona uma imagem a partir de uma URL
  st.image("https://www.leighpartnership.com/wp-content/uploads/2018/09/Data-is-the-new-oil-300x225.gif",use_container_width=True,)

  st.write("""Imagine um executivo de uma grande empresa de logística planejando suas operações. Ele depende de previsões confiáveis do preço do 
              petróleo para determinar custos futuros e mitigar riscos. No passado, decisões desse tipo eram baseadas em experiências passadas, 
              mas no mundo atual, a análise de dados é a principal aliada.""")  

  st.write("""Os dados do mercado de petróleo Brent permitem análises detalhadas das flutuações de preço ao longo de décadas. 
              Isso possibilita identificar padrões sazonais, correlações com eventos geopolíticos e até prever futuras oscilações. 
              Ao utilizar informações históricas organizadas e interpretadas, é possível tomar decisões mais precisas, mitigando riscos financeiros e otimizando operações.""")  

  st.write("""Essa abordagem é essencial em um mundo onde o petróleo continua sendo um dos recursos mais estratégicos e disputados. 
              Dados bem analisados não são apenas uma ferramenta; eles são o diferencial que define o sucesso em um mercado volátil e competitivo.""")                
      
  # Título da aplicação
  st.markdown('<h2 style="color:#e61859;">Metodologia</h2>', unsafe_allow_html=True)

  # Adiciona uma imagem a partir de uma URL
  st.image("https://media.licdn.com/dms/image/v2/D4D12AQH-r-cL6Rj5Ww/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1681912120078?e=2147483647&v=beta&t=XTu5UvOHWOSZyjExEIFeYJKxao0UDmSsuQqb4-adr_Y",use_container_width=True,) 

  st.write("""Para atingir os objetivos deste estudo, adotamos uma abordagem analítica abrangente, combinando ferramentas exploratórias e preditivas. 
              Nosso processo foi estruturado da seguinte forma:""")  

  st.markdown('<h3 style="color:#e61859;">Fonte de Dados Primária</h3>', unsafe_allow_html=True)

  st.write("""Coletamos os dados de preço do petróleo Brent do site do IPEA, com as seguintes características:""")      

  st.write("""- **Variável**: Preço por barril do petróleo bruto Brent (FOB) (**EIA366_PBRENT366**).""")                
  st.write("""- **Frequência**: Diária (de 04/01/1986 até a data mais atual que estiver disponível).""")
  st.write("""- **Unidade**: Dólares (US$).""")          
  st.write("""- **Fonte**: **Energy Information Administration (EIA)** """)    
  st.write("""- **Descrição**: O Brent, produzido no Mar do Norte (Europa), é um benchmark internacional para o preço do petróleo bruto. 
           Os valores são calculados no modelo **FOB (free on board)**, que exclui despesas de frete e seguro.""")
  st.markdown(
      """
      <a href="https://www.eia.gov/dnav/pet/TblDefs/pet_pri_spt_tbldef2.asp" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Para mais informações clique aqui</a>
      """,
      unsafe_allow_html=True
  )
  
  st.markdown('<h3 style="color:#e61859;">Fontes Secundárias</h3>', unsafe_allow_html=True)

  st.write(""" Além dos dados primários, utilizamos também fontes de análise de conteúdo, incluindo:""")  
  st.write("""    - Mídias sociais, blogs e artigos de notícias para capturar percepções de mercado e eventos que possam ter impactado o preço do petróleo.""")   

  st.markdown('<h3 style="color:#e61859;">Análise Descritivas</h3>', unsafe_allow_html=True)
  st.write("""Investigamos os dados históricos do preço do petróleo Brent para identificar tendências gerais, períodos de maior volatilidade e padrões sazonais.""")   

  st.markdown('<h3 style="color:#e61859;">Análise de Séries Temporais</h3>', unsafe_allow_html=True)
  st.write("""Exploramos sazonalidades, tendências de longo prazo e pontos de inflexão para compreender melhor o comportamento dos preços ao longo do tempo.""")   

  st.markdown('<h3 style="color:#e61859;">Modelagem Preditiva (Machine Learning)</h3>', unsafe_allow_html=True)
  st.write("""Desenvolvemos modelos preditivos utilizando algoritmos de aprendizado de máquina para prever os preços futuros do petróleo Brent, oferecendo suporte à tomada de decisões estratégicas.""")   
  
  st.markdown('<h3 style="color:#e61859;">Dashboard Interativo</h3>', unsafe_allow_html=True)
  st.write("""Criamos uma plataforma interativa para visualização dos insights, permitindo uma análise dinâmica e acessível para diferentes stakeholders.""")   

elif menu == "Dados":
  # Título da aplicação
  st.markdown('<h2 style="color:#e61859;">Tratamento dos dados</h2>', unsafe_allow_html=True)

  st.image("https://www.unite.ai/wp-content/uploads/2022/11/ETL.png",use_container_width=True,) 

  st.write("""Baixamos a tabela do site do IPEA""")  
  st.markdown(
      """
      <a href="http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Clique aqui para acessar o site com a tabela</a>
      """,
      unsafe_allow_html=True)  

#################################################################################################################################################

  import pandas as pd
  import streamlit as st

  # URL do IPEADATA com a tabela do petróleo Brent
  url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"

  # Código usado para leitura e exibição da tabela
  codigo = """
  # URL do IPEADATA com a tabela do petróleo Brent
  url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"

  # Tente carregar e exibir a tabela
  try:
      # Lê todas as tabelas disponíveis na URL
      tabelas = pd.read_html(url)
      
      # Seleciona a primeira tabela
      dfb = tabelas[2]

      # Renomeia as colunas
      dfb.columns = ["Data", "Preço"]

      # Exibe o DataFrame na aplicação
      dfb.head()
  """

  # Exibe o código na aplicação
  st.subheader("Código para leitura da tabela do IPEADATA")
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:
      # Lê todas as tabelas disponíveis na URL
      tabelas = pd.read_html(url)
      
      # Seleciona a primeira tabela
      dfb = tabelas[2]

      # Renomeia as colunas
      dfb.columns = ["Data", "Preço"]

      # Exibe o DataFrame na aplicação
      st.subheader("Tabela de Preços do Petróleo Brent")
      st.write(dfb.head())
      
  except Exception as e:
      st.error(f"Erro ao carregar a tabela: {e}")

#################################################################################################################################################

  import io  # Para capturar a saída de df.info()

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Informações do Dataframe
  dfb.info()
  """

  # Capturando a saída de df.info()
  buffer = io.StringIO()
  dfb.info(buf=buffer)  # Redireciona a saída para o buffer
  info_str = buffer.getvalue()  # Obtém o conteúdo do buffer

  # Mostrando as informações no Streamlit
  st.subheader("Informações do DataFrame")
  st.code(codigo, language='python')
  st.text(info_str)  # Exibe como texto simples

#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # 1. Remover a primeira linha
  dfb = dfb.iloc[1:]

  dfb.head()
  """

  # Exibe o código na aplicação
  st.subheader("Iniciando o tratamento dos dados para dar início à análise:")
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:
      # 1. Remover a primeira linha
      dfb = dfb.iloc[1:]

      dfb.head()

      # Exibe o DataFrame na aplicação
      st.write(dfb.head())
      
  except Exception as e:
      st.error(f"Erro ao carregar a tabela: {e}")

#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Converter a coluna 'Data' para datetime
  dfb['Data'] = pd.to_datetime(dfb['Data'], format='%d/%m/%Y', errors='coerce')
  """

  # Exibe o código na aplicação
  st.code(codigo, language='python')


#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Ajustando os valores para ficar de acordo com a tabela original
  # Dividir por 100 para ajustar os valores
  dfb['Preço'] = pd.to_numeric(dfb['Preço'], errors='coerce') / 100

  dfb.head()
  """

  # Exibe o código na aplicação
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:
    # Ajustando os valores para ficar de acordo com a tabela original
    # Dividir por 100 para ajustar os valores
    dfb['Preço'] = pd.to_numeric(dfb['Preço'], errors='coerce') / 100

    dfb.head()

    # Exibe o DataFrame na aplicação
    st.write(dfb.head())
      
  except Exception as e:
      st.error(f"Erro ao carregar a tabela: {e}")


#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Verficando o tamanho da tabela
  dfb.shape
  """

  # Exibe o código na aplicação
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:

    # Exibe o DataFrame na aplicação
    st.write(dfb.shape)
      
  except Exception as e:
      st.error(f"Erro ao carregar a tabela: {e}")

#################################################################################################################################################

  st.write("**Nos dados, esses valores de preço sao em dólares**")
  st.write("**Agora vamos partir para a análise exporatório dos dados**")


  # Título da aplicação
  st.markdown('<h2 style="color:#e61859;">Análise Exploratória de Dados</h2>', unsafe_allow_html=True)
  st.image("https://www.dataimd.com/scripts/explorando-os-segredos-dos-dados-uma-jornada-de-descobertas-atrav-s-da-an-lise-explorat-ria/featured_hub3b9602acdbd07ff9c7c825b05a2b1d2_6051932_720x2500_fit_q75_h2_lanczos.webp",use_container_width=True,) 
  
  # Código usado para leitura e exibição da tabela
  codigo = """
  # Iniciando a analise do preco
  dfb['Preço'].describe()
  """

  # Exibe o código na aplicação
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:
    # Iniciando a analise do preco
    st.write(dfb['Preço'].describe())
      
  except Exception as e:
      st.error(f"Erro ao carregar a tabela: {e}")

  st.write("""O conjunto de dados possui 11.292 registros, com uma média de preço de 53,31 dólares e um desvio padrão de 33,16 dólares.""") 
  st.write("""Os valores variam entre um mínimo de 9,10 dólares e um máximo de 143,95 dólares. A mediana foi de 48,89 dólares, enquanto 25% dos valores estão 
            abaixo de 20,61 dólares e 75% estão abaixo de 76,81 dólares, indicando que a maioria dos preços se concentra em uma faixa mais elevada.""") 
  st.write("""Essa distribuição pode sugerir uma tendência de aumento ao longo do tempo, com variações menos significativas em direção a preços mais baixos.""") 


#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Vamos verficar se existem valores nulos
  dfb.isnull().sum()
  """

  # Exibe o código na aplicação
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:
    
    st.write(dfb.isnull().sum())
      
  except Exception as e:
      st.error(f"Erro ao carregar a tabela: {e}")

  st.write("""Os dados não possuem valores nulos.""") 

#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Verificando duplicatas
  dfb.duplicated().sum()
  """

  # Exibe o código na aplicação
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:
    
    st.write(dfb.duplicated().sum())
      
  except Exception as e:
      st.error(f"Erro ao carregar a tabela: {e}")

  st.write("""Não há valores repetidos na tabela.""") 

#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  dfb.head(30)
  """

  # Exibe o código na aplicação
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:
    
    st.write(dfb.head(30))
      
  except Exception as e:
      st.error(f"Erro ao carregar a tabela: {e}")

  st.write("""Observa-se que, assim como ocorre em valores de bolsa, as datas excluem os finais de semana.""") 


#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  #codigo = """
  ## Para facilitar ja vamos deixa-la ajustada para ser usada no prophet mais adiante
  #dfb.rename(columns={'Data': 'ds', 'Preço': 'y'}, inplace=True)

  # Definir a coluna 'ds' como índice
  #dfb.set_index('ds', inplace=True)
  #
  #dfb.head(20)
  #"""

  # Exibe o código na aplicação
  #st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  #try:

    # Definir a coluna 'ds' como índice
  #  dfb.set_index('Data', inplace=True)

  #  st.write(dfb.head(20))
      
  #except Exception as e:
  #    st.error(f"Erro ao carregar a tabela: {e}")

#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Importando as libs
  import matplotlib.pyplot as plt
  import seaborn as sns

  # Criando os gráficos de boxplot e violino
  fig, axes = plt.subplots(1, 2, figsize=(15, 6))
  sns.boxplot(data=dfb, ax=axes[0])
  sns.violinplot(data=dfb, ax=axes[1])
  plt.show()
  """

  # Exibe o código na aplicação
  st.code(codigo, language='python')
  st.subheader("Iniciando a visualização dos dados")
  

  # Implementação do código para carregar e exibir a tabela
  try:

    # Importando as libs
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd  # Certifique-se de que pandas está importado

    # Criando os gráficos de boxplot e violino
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(data=dfb, ax=axes[0])
    sns.violinplot(data=dfb, ax=axes[1])

    # Mostrando o gráfico na aplicação Streamlit
    st.pyplot(fig)

  except Exception as e:
      st.error(f"Erro ao gerar gráficos: {e}")

  # Descrição dos gráficos
  st.write("""No gráfico, identificamos as concentrações dos valores. A média, conforme constatado, permanece em 48,88, como indicado pelo boxplot. 
              Não há presença evidente de outliers em uma análise geral.""")
  st.write("""No gráfico de violino, observa-se que a maior parte dos valores está concentrada até 40, com uma segunda concentração significativa entre 40 e 100. 
              Apenas uma pequena parcela dos dados atinge valores próximos a 140.""")      


#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Plotando um histograma do dfb
  plt.figure(figsize=(15, 6))
  sns.histplot(data=dfb, kde=True)
  plt.show()
  """

  # Exibe o código na aplicação
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:

    # Importando as bibliotecas
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd  # Certifique-se de que pandas está importado
    
    # Plotando um histograma do dfb
    plt.figure(figsize=(15, 6))
    sns.histplot(data=dfb, kde=True)
    
    # Mostrando o gráfico na aplicação Streamlit
    st.pyplot(plt)

  except Exception as e:
      st.error(f"Erro ao gerar gráficos: {e}")

  # Descrição dos gráficos
  st.write("""Este gráfico confirma que a grande maioria dos valores da tabela está concentrada entre 10 e 40.""")



#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Vamos visulizar como se comporta pelo tempo
  plt.figure(figsize=(15, 6))
  sns.lineplot(data=dfb)
  plt.show()
  """

  # Exibe o código na aplicação
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:
    # Importando as bibliotecas
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Convertendo a coluna de data para o tipo de data do Pandas
    dfb['Data'] = pd.to_datetime(dfb['Data'], dayfirst=True)

    # Ordenando o DataFrame pela coluna de data
    dfb = dfb.sort_values('Data')
    
    # Plotando o gráfico e armazenando a figura
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=dfb, x=dfb['Data'], y='Preço')
    fig = plt.gcf()

    # Mostrando o gráfico na aplicação Streamlit
    st.pyplot(fig)

  except Exception as e:
    st.error(f"Erro ao gerar gráficos: {e}")

  # Descrição dos gráficos
  st.write("""Identificamos um pico moderado entre os anos de 1990 e 1995, que, embora significativo, foi menor em comparação ao pico expressivo registrado entre 2005 e 2010. 
            Observou-se uma recuperação gradual de 2010 a 2015, seguida de uma nova queda. Entre 2015 e 2020, houve grande volatilidade, culminando em uma das maiores quedas do gráfico em 2020, 
            possivelmente devido à pandemia. Contudo, até 2024, ocorreu uma rápida recuperação, alcançando os níveis mais recentes registrados na tabela.""")

#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Importando o plotly
  import plotly.express as px
  
  # Plotando o mesmo grafico usando o plotly
  fig = px.line(dfb, x=dfb.index, y='y')
  fig.show()
  """

  # Exibe o código na aplicação
  st.subheader("Vamos aprofundar a análise para identificar as datas específicas em que esses eventos ocorreram:")
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:
    # Importando o plotly
    import plotly.express as px
    
    # Convertendo a coluna de data para o tipo de data do Pandas
    dfb['Data'] = pd.to_datetime(dfb['Data'], dayfirst=True)

    # Ordenando o DataFrame pela coluna de data
    dfb = dfb.sort_values('Data')

    # Plotando o mesmo grafico usando o plotly
    fig = px.line(dfb, x=dfb['Data'], y='Preço')

    # Mostrando o gráfico na aplicação Streamlit
    st.plotly_chart(fig)

  except Exception as e:
    st.error(f"Erro ao gerar gráficos: {e}")

  # Descrição dos gráficos
  st.write("""Neste gráfico, podemos observar as datas com maior precisão:""")
  st.write("""- O primeiro pico ocorreu em setembro de 1990.""")  
  st.write("""- O segundo pico, o maior registrado no gráfico, aconteceu em julho de 2008, seguido de uma queda acentuada até dezembro de 2008.""")    
  st.write("""- Após esse período, os valores se recuperaram gradualmente até maio de 2011.""")    
  st.write("""- Em 2012, houve um aumento moderado, seguido de uma queda em junho do mesmo ano.""")    
  st.write("""- Entre julho de 2014 e janeiro de 2015, ocorreu outra queda acentuada.""")    
  st.write("""- No período subsequente, observou-se flutuações até janeiro de 2020, quando houve novamente uma queda brusca, possivelmente relacionada à COVID-19, que se estendeu até abril de 2020.""")    
  st.write("""- Após essa queda, houve uma rápida recuperação, com um aumento registrado em março de 2022, seguido de uma queda em setembro de 2022.""")  
  st.write("""- Desde então, os valores apresentaram flutuações, mas demonstram uma estabilidade maior em comparação a outros períodos.""")  

#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Criando um scatter plot
  plt.figure(figsize=(15, 6))
  sns.scatterplot(data=dfb)
  plt.show()
  """

  # Exibe o código na aplicação
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir a tabela
  try:
    # Convertendo a coluna de data para o tipo de data do Pandas
    dfb['Data'] = pd.to_datetime(dfb['Data'], dayfirst=True)

    # Ordenando o DataFrame pela coluna de data
    dfb = dfb.sort_values('Data')

    # Plotando o mesmo grafico usando o plotly
    plt.figure(figsize=(15, 6))
    sns.scatterplot(data=dfb, x=dfb['Data'], y='Preço')
    fig = plt.gcf()

    # Mostrando o gráfico na aplicação Streamlit
    st.pyplot(fig)    

  except Exception as e:
    st.error(f"Erro ao gerar gráficos: {e}")

  # Descrição dos gráficos
  st.write("""No início do gráfico, observa-se uma maior concentração de pontos, indicando menor volatilidade nos valores, exceto em 1990, que apresentou flutuações mais significativas.""")
  st.write("""Já no final de 2008, o espaçamento entre os pontos revela uma queda rápida nos valores, possivelmente associada a crises. 
            Situações semelhantes podem ser observadas em 2014 e, novamente, em 2020, quando o valor mais baixo registrado (9,12 dólares) foi alcançado. 
            O maior espaçamento nesse período sugere uma queda acelerada, provavelmente a mais rápida em todo o gráfico.""")  

#################################################################################################################################################

  st.markdown('<h2 style="color:#e61859;">Analisando os eventos que ocorreram nesses períodos</h2>', unsafe_allow_html=True)
  
  st.markdown('<h4>Análise do Aumento do Preço do Petróleo em 1990</h4>', unsafe_allow_html=True)

  st.markdown('<h5>Contexto Histórico</h5>', unsafe_allow_html=True)
  st.write("""Em 1990, o mercado global de petróleo sofreu uma grande perturbação devido a eventos geopolíticos significativos na região do Golfo Pérsico. 
            O evento mais marcante foi a **invasão do Kuwait pelo Iraque** em 2 de agosto de 1990, desencadeando o que ficou conhecido como a **Guerra do Golfo**.""")

  st.markdown('<h5>Impacto no Mercado de Petróleo</h5>', unsafe_allow_html=True)
  st.write("""1. **Interrupção da Produção**:""")
  st.write("""- O Kuwait, um dos principais produtores de petróleo da OPEP, teve sua produção completamente interrompida pela invasão.""")
  st.write("""- O Iraque também enfrentou sanções econômicas impostas pela comunidade internacional, limitando sua capacidade de exportar petróleo.""")

  st.write("""2. **Ameaça à Estabilidade Regional**:""")
  st.write("""- O Golfo Pérsico é uma das regiões mais estratégicas para a produção e exportação de petróleo.""")
  st.write("""- A possibilidade de escalada do conflito e o risco de interrupções em outras rotas e países produtores (como Arábia Saudita) causaram pânico nos mercados.""")

  st.write("""3. **Aumento nos Preços**:""")
  st.write("""- Antes da invasão, o preço médio mensal do petróleo Brent estava em torno de **US17 por barril** (julho de 1990).""")
  st.write("""- Em outubro de 1990, o preço atingiu **US41 por barril**, um aumento de mais de 100% em apenas três meses.""")

  st.markdown('<h5>Fatores Contribuintes</h5>', unsafe_allow_html=True)
  st.write("""1. **Volatilidade nos Estoques e Produção**:""")
  st.write("""- A Organização dos Países Exportadores de Petróleo (OPEP) enfrentou dificuldades para coordenar respostas eficazes à crise.""")
  st.write("""- Apesar de tentativas de aumentar a produção em outros países membros, as incertezas políticas mantiveram os preços altos.""")

  st.write("""2. **Sentimento do Mercado**:""")
  st.write("""- A guerra trouxe uma incerteza global significativa, afetando tanto países consumidores quanto produtores.""")
  st.write("""- A especulação sobre a continuidade do fornecimento de petróleo aumentou a volatilidade dos preços.""")

  st.write("""**Fontes**""")
  st.markdown(
      """
      <a href="https://en.wikipedia.org/wiki/1990_oil_price_shock" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Wikipedia: 1990 oil price shock</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://en.wikipedia.org/wiki/1990%E2%80%931999_world_oil_market_chronology" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Wikipedia: 1990–1999 world oil market chronology</a>
      """,
      unsafe_allow_html=True)  

#################################################################################################################################################

  st.markdown('<h4>Análise do Aumento do Preço do Petróleo em 2008</h4>', unsafe_allow_html=True)

  st.markdown('<h5>Contexto Histórico</h5>', unsafe_allow_html=True)
  st.write("""Em 2008, o mercado global de petróleo experimentou uma volatilidade sem precedentes, com os preços atingindo níveis recordes no meio do ano e 
            caindo drasticamente nos meses subsequentes. Essas flutuações foram influenciadas por uma combinação de fatores econômicos, geopolíticos e especulativos.""")

  st.markdown('<h5>Impacto no Mercado de Petróleo</h5>', unsafe_allow_html=True)
  st.write("""1. **Pico dos Preços**:""")  
  st.write("""- Em julho de 2008, o preço do barril de petróleo Brent atingiu aproximadamente **US$ 144**, o valor mais alto registrado até então.""")               

  st.write("""2. **Queda Abrupta**:""")  
  st.write("""- Após o pico, os preços começaram a cair rapidamente, chegando a cerca de **US$ 34** por barril em dezembro de 2008.""")      

  st.markdown('<h5>Fatores Contribuintes</h5>', unsafe_allow_html=True)
  st.write("""1. **Aumento da Demanda Global**:""")  
  st.write("""- Países em rápido desenvolvimento, como China e Índia, apresentaram um crescimento econômico acelerado, aumentando significativamente a demanda por petróleo.""")   

  st.write("""2. **Especulação no Mercado**:""")  
  st.write("""- Investidores financeiros aumentaram suas participações em contratos futuros de petróleo, contribuindo para a elevação dos preços.""") 

  st.write("""3. **Crise Financeira Global**:""")  
  st.write("""- A crise financeira de 2008, desencadeada pelo colapso do Lehman Brothers em setembro, resultou em uma recessão global que reduziu drasticamente a demanda por petróleo, levando à queda acentuada dos preços.""")  

  st.write("""4. **Flutuações Cambiais**:""")  
  st.write("""- A desvalorização do dólar americano no início de 2008 tornou o petróleo mais barato para detentores de outras moedas, estimulando a demanda e elevando os preços.""")                     

  st.write("""**Fontes**""")
  st.markdown(
      """
      <a href="https://en.wikipedia.org/wiki/2000s_energy_crisis" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Wikipedia: 2000s energy crisis</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://en.wikipedia.org/wiki/2000s_commodities_boom" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Wikipedia: 2000s commodities boom</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://en.wikipedia.org/wiki/Price_of_oil" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Price of oil</a>
      """,
      unsafe_allow_html=True) 
  st.markdown(
      """
      <a href="https://en.wikipedia.org/wiki/World_oil_market_chronology_from_2003" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      World oil market chronology from 2003</a>
      """,
      unsafe_allow_html=True)        

#################################################################################################################################################

  st.markdown('<h4>Análise da Queda do Preço do Petróleo em 2014</h4>', unsafe_allow_html=True)

  st.markdown('<h5>Contexto Histórico</h5>', unsafe_allow_html=True)
  st.write("""Em 2014, o mercado global de petróleo registrou uma queda dramática nos preços, com o barril do Brent passando de aproximadamente 
            **US 115 em junho** para cerca de **US 46 em janeiro de 2015**. Essa queda foi causada por uma combinação de fatores estruturais e 
            estratégicos que resultaram em um excesso de oferta no mercado global.""")

  st.markdown('<h5>Impacto no Mercado de Petróleo</h5>', unsafe_allow_html=True)
  st.write("""1. **Queda Acentuada nos Preços**:""")  
  st.write("""- Em seis meses, o preço do petróleo Brent caiu mais de **60%**, marcando uma das quedas mais rápidas e significativas da história recente.""")      

  st.write("""2. **Desequilíbrio entre Oferta e Demanda**:""")  
  st.write("""- O aumento da produção global coincidiu com uma desaceleração na demanda por petróleo, especialmente em economias emergentes.""")        

  st.markdown('<h5>Fatores Contribuintes</h5>', unsafe_allow_html=True)
  st.write("""1. **Produção Aumentada de Petróleo de Xisto nos Estados Unidos**:""")  
  st.write("""- Avanços em tecnologias como o fraturamento hidráulico (fracking) permitiram aos Estados Unidos aumentar significativamente a produção de petróleo de xisto.""")      
  st.write("""- O aumento da produção americana reduziu a dependência de importações e adicionou uma grande quantidade de petróleo ao mercado global, criando um excedente de oferta.""") 

  st.write("""2. **Decisão Estratégica da OPEP**:""")  
  st.write("""- Em novembro de 2014, a Organização dos Países Exportadores de Petróleo (OPEP), liderada pela Arábia Saudita, decidiu manter os níveis de produção, apesar do excesso de oferta no mercado.""")      
  st.write("""- Essa decisão foi vista como uma tentativa de pressionar produtores de alto custo, como os de petróleo de xisto, a sair do mercado, preservando a participação da OPEP a longo prazo.""")    

  st.write("""3. **Redução da Demanda Global**:""")  
  st.write("""- A desaceleração econômica em economias emergentes, particularmente na China, resultou em uma demanda menor por petróleo, agravando o desequilíbrio no mercado.""")  

  st.write("""4. **Fortalecimento do Dólar Americano**:""")  
  st.write("""- O fortalecimento do dólar americano tornou o petróleo, cotado nessa moeda, mais caro para compradores estrangeiros, reduzindo a demanda.""")  

  st.markdown('<h5>Consequências</h5>', unsafe_allow_html=True)
  st.write("""1. **Impacto nos Países Produtores**:""")  
  st.write("""- Economias altamente dependentes da exportação de petróleo, como Venezuela, Nigéria e Rússia, sofreram grandes perdas de receita, enfrentando desafios econômicos severos.""")      
  st.write("""- Esses países foram forçados a adotar medidas de austeridade ou buscar alternativas econômicas.""") 

  st.write("""2. **Benefícios para Países Importadores**:""")  
  st.write("""- Países importadores de petróleo, como Estados Unidos e membros da União Europeia, se beneficiaram da queda nos preços, resultando em custos de energia mais baixos, maior consumo e redução da inflação.""")      


  st.write("""**Fontes**""")
  st.markdown(
      """
      <a href="https://g1.globo.com/economia/noticia/2015/01/entenda-queda-do-preco-do-petroleo-e-seus-efeitos.html" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      G1: Entenda a queda do preço do petróleo e seus efeitos</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://www.dw.com/pt-002/8-causas-da-queda-livre-do-pre%C3%A7o-do-petr%C3%B3leo/a-19027834" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      DW: 8 causas da queda livre do preço do petróleo</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://economia.uol.com.br/noticias/bbc/2014/10/17/quem-ganha-e-quem-perde-com-a-queda-do-preco-do-petroleo.htm" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      UOL Economia: Quem ganha e quem perde com a queda do preço do petróleo</a>
      """,
      unsafe_allow_html=True) 


#################################################################################################################################################

  st.markdown('<h2 style="color:#e61859;">Retornando à análise</h2>', unsafe_allow_html=True)

  st.write("""Vamos analisar o comportamento dos valores ao longo dos anos e meses.""")   

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Criando um boxplot por mes e ano

  df_boxplot = dfb.copy()
  df_boxplot.reset_index(inplace=True)
  df_boxplot['ds'] = pd.to_datetime(df_boxplot['ds'])
  df_boxplot['year'] = df_boxplot['ds'].dt.year
  df_boxplot['month'] = df_boxplot['ds'].dt.month

  # Gráfico de Boxplot por Ano
  plt.figure(figsize=(15, 6))
  sns.boxplot(x='year', y='y', data=df_boxplot)
  plt.title('Distribuição dos Precos por Ano')
  plt.xlabel('Ano')
  plt.ylabel('Preco')
  plt.xticks(rotation=45)
  plt.show()
  """

  # Exibe o código na aplicação
  st.subheader("Análise por ano:")
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir o gráfico
  try:
      # Importando as bibliotecas
      import streamlit as st
      import matplotlib.pyplot as plt
      import seaborn as sns
      import pandas as pd

      # Convertendo a coluna 'Data' para o tipo datetime
      dfb['Data'] = pd.to_datetime(dfb['Data'])

      # Criando as colunas 'year' e 'month'
      df_boxplot = dfb.copy()
      df_boxplot['year'] = df_boxplot['Data'].dt.year

      # Criando o gráfico
      plt.figure(figsize=(15, 6))
      sns.boxplot(x='year', y='Preço', data=df_boxplot)
      plt.title('Distribuição dos Preços por Ano')
      plt.xlabel('Ano')
      plt.ylabel('Preço')
      plt.xticks(rotation=45)

      # Armazenando a figura e exibindo no Streamlit
      fig = plt.gcf()
      st.pyplot(fig)

  except Exception as e:
      st.error(f"Erro ao gerar gráficos: {e}")


  # Descrição dos gráficos
  st.write("""Neste gráfico, é possível visualizar o comportamento do preço do petróleo ao longo dos anos:""")
  st.write("""- **1987**: Ano de início da tabela, marcado por muita flutuação e presença significativa de outliers.""")  
  st.write("""- **1991**: Apresenta muitos outliers, provavelmente decorrentes do aumento em 1990 seguido de uma queda acentuada.""")  
  st.write("""- **1997**: Embora tenha apresentado muitos outliers, as flutuações foram menores que as de 1990, possivelmente refletindo picos de aumento nos preços ao longo do ano.""")    
  st.write("""- **2002**: Demonstrou flutuações tanto para preços altos quanto para baixos, indicando um ano instável.""")    
  st.write("""- **2008**: Um ano de grandes variações, com um dos preços mais altos registrados, seguido de um declínio acentuado, evidenciado por outliers no espectro de valores baixos.""")  
  st.write("""- **2011 e 2012**: Períodos de instabilidade, com outliers voltados para preços mais baixos.""") 
  st.write("""- **2014-2015**: Em 2014, iniciou-se um declínio acentuado e acelerado nos preços, que se estabilizou em 2015, tornando-se um ano mais estável em comparação.""") 
  st.write("""- **2017**: Registrou um crescimento nos preços do petróleo.""") 
  st.write("""- **2018**: Observou-se outra queda acelerada nos preços.""") 
  st.write("""- **2020**: Ano mais incerto do gráfico, com outliers tanto para valores altos quanto baixos. Este foi o período de maior instabilidade, com variações extremas e o menor valor registrado em toda a série.""") 
  st.write("""- **2022**: Ano de recuperação, com aumento nos valores do petróleo.""") 
  st.write("""- **2023 e 2024**: Caracterizam-se por menor volatilidade, com preços mais estáveis em comparação aos anos anteriores.""") 

#################################################################################################################################################

  st.markdown('<h4>Análise da Queda do Preço do Petróleo em 2020</h4>', unsafe_allow_html=True)

  st.markdown('<h5>Contexto Histórico</h5>', unsafe_allow_html=True)
  st.write("""Em 2020, o mercado global de petróleo enfrentou uma das maiores quedas de preços já registradas, resultando de uma combinação de fatores sem precedentes que afetaram tanto a oferta quanto a demanda.""")

  st.markdown('<h5>Impacto no Mercado de Petróleo</h5>', unsafe_allow_html=True)
  st.write("""1. **Queda Acentuada nos Preços**:""")  
  st.write("""- Em abril de 2020, o preço do barril de petróleo Brent caiu para menos de **US$ 10**, níveis não vistos desde 2002.""")      

  st.write("""2. **Preços Negativos do WTI**:""")  
  st.write("""- Em 20 de abril de 2020, os contratos futuros do petróleo WTI nos EUA atingiram valores negativos pela primeira vez na história, chegando a **-US$ 37,63** por barril.""")   

  st.markdown('<h5>Fatores Contribuintes</h5>', unsafe_allow_html=True)
  st.write("""1. **Pandemia de COVID-19**:""")  
  st.write("""- **Redução da Demanda**: Medidas de confinamento e restrições de viagem globalmente implementadas reduziram drasticamente a demanda por combustíveis fósseis.""")    
  st.write("""- **Interrupção Econômica**: A desaceleração econômica global diminuiu a atividade industrial, reduzindo ainda mais a demanda por energia.""")    

  st.write("""2. **Guerra de Preços entre Arábia Saudita e Rússia**:""")  
  st.write("""- **Colapso das Negociações da OPEP+**: Em março de 2020, a OPEP e aliados, incluindo a Rússia, não conseguiram chegar a um acordo sobre cortes de produção.""")    
  st.write("""- **Aumento da Produção**: A Arábia Saudita respondeu aumentando sua produção e oferecendo descontos significativos, iniciando uma guerra de preços que exacerbou a queda dos preços.""")   

  st.write("""3. **Capacidade de Armazenamento Limitada**:""")  
  st.write("""- Com a demanda em declínio e a produção permanecendo alta, os estoques de petróleo atingiram a capacidade máxima, especialmente nos EUA, levando a situações onde os produtores pagavam para que outros assumissem o petróleo excedente.""")   

  st.markdown('<h5>Consequências</h5>', unsafe_allow_html=True)
  st.write("""1. **Impacto nos Países Produtores**:""")  
  st.write("""- Países dependentes da exportação de petróleo, como Rússia, Venezuela e países do Oriente Médio, enfrentaram quedas significativas nas receitas, afetando suas economias.""")    

  st.write("""2. **Benefícios para Países Importadores**:""")  
  st.write("""- Nações importadoras de petróleo experimentaram custos de energia mais baixos, o que poderia estimular a recuperação econômica pós-pandemia.""")    

  st.write("""3. **Impacto na Indústria de Energia**:""")  
  st.write("""- Empresas de petróleo e gás enfrentaram desafios financeiros, levando a cortes de empregos, redução de investimentos e, em alguns casos, falências.""")    

  st.write("""A queda do preço do petróleo em 2020 foi resultado de uma tempestade perfeita de fatores: uma pandemia global que reduziu drasticamente a demanda, 
            uma guerra de preços entre grandes produtores que aumentou a oferta e limitações na capacidade de armazenamento. Este evento destacou a vulnerabilidade 
            do mercado de petróleo a choques simultâneos de oferta e demanda e teve implicações econômicas profundas em todo o mundo.""")    

  st.write("""**Fontes**""")
  st.markdown(
      """
      <a href="https://www.dw.com/pt-br/por-que-o-pre%C3%A7o-do-petr%C3%B3leo-despencou/a-53202121" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Por que o preço do petróleo despencou? – DW – 21/04/2020</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://g1.globo.com/economia/noticia/2020/03/09/o-que-explica-o-tombo-do-preco-do-petroleo-e-quais-os-seus-efeitos.ghtml" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      O que explica o tombo do preço do petróleo e quais os seus efeitos - G1</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://www.ibp.org.br/observatorio-do-setor/analises/covid-19-e-os-impactos-sobre-o-mercado-de-petroleo/" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      COVID-19 e os impactos sobre o mercado de petróleo - IBP</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://economia.uol.com.br/noticias/afp/2020/12/31/o-preco-do-petroleo-fecha-2020-com-uma-queda-de-mais-de-20-devido-a-covid-19.htm" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      O preço do petróleo fecha 2020 com uma queda de mais de 20% devido à ...</a>
      """,
      unsafe_allow_html=True)  

#################################################################################################################################################

  st.markdown('<h4>Análise da Recuperação do Preço do Petróleo em 2020-2021</h4>', unsafe_allow_html=True)

  st.markdown('<h5>Contexto Histórico</h5>', unsafe_allow_html=True)
  st.write("""Após a queda histórica nos preços do petróleo em 2020, causada pela pandemia de COVID-19 e uma guerra de preços entre grandes produtores, 
            o mercado iniciou uma recuperação significativa no final de 2020, que se intensificou ao longo de 2021. 
            Essa recuperação foi impulsionada por fatores econômicos, geopolíticos e estratégicos que reequilibraram a oferta e a demanda globais.""")

  st.markdown('<h5>Impacto no Mercado de Petróleo</h5>', unsafe_allow_html=True)
  st.write("""1. **Aumento Gradual dos Preços**:""")  
  st.write("""- Em outubro de 2020, o preço do barril de petróleo Brent estava em torno de **US$ 40**.""")  
  st.write("""- Em junho de 2021, os preços já haviam atingido cerca de **US$ 74**, representando um aumento de mais de 80%.""")      

  st.write("""2. **Estabilização do Mercado**:""")  
  st.write("""- O mercado se reequilibrou com a retomada da demanda e ajustes na oferta.""")   

  st.markdown('<h5>Fatores Contribuintes</h5>', unsafe_allow_html=True)
  st.write("""1. **Retomada da Demanda Global**:""")  
  st.write("""- **Adoção de Vacinas**: A implementação de campanhas de vacinação em massa reduziu as restrições relacionadas à pandemia, permitindo a retomada da mobilidade e da atividade econômica.""")  
  st.write("""- **Aumento no Consumo**: A recuperação da demanda foi impulsionada por setores como transporte e indústria, que começaram a operar em níveis próximos aos pré-pandemia.""")   

  st.write("""2. **Acordos de Produção da OPEP+**:""")  
  st.write("""- Após a guerra de preços de 2020, os membros da OPEP+ adotaram cortes de produção rigorosos para reduzir o excesso de oferta.""")  
  st.write("""- A produção foi ajustada gradualmente para atender à demanda crescente, ajudando a sustentar a recuperação dos preços.""")   

  st.write("""3. **Restrições na Oferta**:""")  
  st.write("""- A produção em algumas regiões enfrentou limitações devido a fatores técnicos e geopolíticos, restringindo ainda mais a oferta e pressionando os preços.""")  

  st.write("""4. **Condições Macroeconômicas**:""")  
  st.write("""- O enfraquecimento do dólar americano durante parte do período tornou o petróleo mais barato para compradores internacionais, estimulando ainda mais a demanda.""")  

  st.markdown('<h5>Consequências</h5>', unsafe_allow_html=True)
  st.write("""1. **Impacto Positivo nos Países Produtores**:""")  
  st.write("""- Economias dependentes de petróleo, como Arábia Saudita e Rússia, experimentaram uma recuperação nas receitas, ajudando a estabilizar suas economias após o choque de 2020.""")  

  st.write("""2. **Pressão sobre Países Importadores**:""")  
  st.write("""- Apesar de alguns benefícios econômicos, os preços mais altos do petróleo começaram a gerar inflação em países importadores, pressionando custos de energia e produção industrial.""")  

  st.markdown('<h5>Conclusão</h5>', unsafe_allow_html=True)
  st.write("""A recuperação dos preços do petróleo entre 2020 e 2021 foi resultado de uma combinação de fatores, incluindo o aumento da demanda devido à retomada econômica, 
            os cortes estratégicos de produção da OPEP+ e restrições na oferta. Esse período demonstrou a capacidade do mercado de se ajustar rapidamente a choques extremos,
            refletindo a interação complexa entre fatores geopolíticos, econômicos e técnicos.""")  

  st.write("""**Fontes**""")
  st.markdown(
      """
      <a href="https://g1.globo.com/economia/noticia/2021/10/17/por-que-o-preco-do-petroleo-esta-disparando-no-mundo-todo.ghtml" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Por que o preço do petróleo está disparando no mundo todo - G1</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://www.ibp.org.br/observatorio-do-setor/snapshots/evolucao-dos-precos-internacionais-do-petroleo-e-projecoes-para-2025/" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Evolução dos preços internacionais do petróleo - IBP</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://www.poder360.com.br/brasil/preco-do-petroleo-sobe-41-em-2021-e-atinge-recorde-em-2-anos/" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Preço do petróleo sobe 41% em 2021 - Poder360</a>
      """,
      unsafe_allow_html=True)       


#################################################################################################################################################

  st.markdown('<h4>Análise das Flutuações do Preço do Petróleo de 2022 até Novembro de 2024</h4>', unsafe_allow_html=True)

  st.markdown('<h5>Contexto Geral</h5>', unsafe_allow_html=True)
  st.write("""Entre 2022 e 2024, o mercado global de petróleo passou por flutuações significativas nos preços, influenciadas por uma combinação de fatores geopolíticos, econômicos e de oferta e demanda.""")

  st.markdown('<h5>Principais Eventos e Impactos nos Preços</h5>', unsafe_allow_html=True)
  st.write("""1. **Invasão da Ucrânia pela Rússia (Fevereiro de 2022)**""")
  st.write("""- **Impacto**: A invasão russa à Ucrânia em fevereiro de 2022 gerou preocupações sobre a estabilidade do fornecimento de petróleo, uma vez que a Rússia é um dos maiores produtores mundiais. 
            Como resultado, os preços do petróleo Brent ultrapassaram US$ 120 por barril em março de 2022.""")  

  st.write("""2. **Acordos da OPEP+ e Ajustes na Produção (2022-2023)**""")
  st.write("""- **Impacto**: A Organização dos Países Exportadores de Petróleo e aliados (OPEP+) implementou cortes de produção para equilibrar o mercado diante da volatilidade. 
            Essas medidas ajudaram a estabilizar os preços em torno de US$ 80 a US$ 90 por barril durante 2023.""")  

  st.write("""3. **Recuperação Econômica Pós-Pandemia e Demanda Global (2022-2023)**""")
  st.write("""- **Impacto**: A retomada econômica global após a pandemia de COVID-19 aumentou a demanda por petróleo, contribuindo para a elevação dos preços. 
            No entanto, preocupações com a inflação e políticas monetárias restritivas em várias economias limitaram aumentos mais acentuados.""")       

  st.write("""4. **Aumento da Produção dos EUA e Outros Produtores Não-OPEP (2023-2024)**""")
  st.write("""- **Impacto**: A produção de petróleo nos Estados Unidos atingiu níveis recordes, com uma média de 13,23 milhões de barris por dia em 2023 e projeções de 13,53 milhões para 2024. Essa oferta adicional exerceu pressão descendente sobre os preços.""")   

  st.write("""5. **Tensões Geopolíticas no Oriente Médio (Outubro de 2024)**""")
  st.write("""- **Impacto**: Conflitos na região, incluindo ataques a instalações petrolíferas, elevaram os preços temporariamente devido a preocupações com interrupções no fornecimento. No entanto, a capacidade de resposta rápida de outros produtores ajudou a mitigar aumentos prolongados.""")   

  st.write("""6. **Preocupações com a Demanda e Perspectivas Econômicas (2024)**""")
  st.write("""- **Impacto**: Previsões de crescimento econômico mais lento, especialmente na China, e expectativas de aumento da oferta global levaram a uma queda nos preços do petróleo, com o Brent caindo para cerca de US$ 71 por barril em novembro de 2024.""")   

  st.markdown('<h5>Conclusão</h5>', unsafe_allow_html=True)
  st.write("""De 2022 até novembro de 2024, o mercado de petróleo foi marcado por volatilidade significativa, influenciada por eventos geopolíticos, decisões estratégicas de produção e dinâmicas de oferta e demanda globais. 
            A capacidade de adaptação dos produtores e as respostas políticas desempenharam papéis cruciais na determinação das tendências de preços durante esse período.""")  

  st.write("""**Fontes**""")
  st.markdown(
      """
      <a href="https://www.reuters.com/es/negocio/6T32DZDS7VOLTKN7AXYUQ46EN4-2024-11-13/" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      EIA eleva ligeramente las previsiones de producción de petróleo en EEUU y en el mundo</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://pt.tradingeconomics.com/commodity/brent-crude-oil" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Brent - petróleo - Contrato Futuro - Preços | 1970-2024 Dados | 2025-2026 Previsão</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://www.ibp.org.br/observatorio-do-setor/snapshots/evolucao-dos-precos-internacionais-do-petroleo-e-projecoes-para-2025/" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Evolução dos preços internacionais do petróleo e projeções para 2025</a>
      """,
      unsafe_allow_html=True)      
  st.markdown(
      """
      <a href="https://valorinveste.globo.com/mercados/internacional-e-commodities/noticia/2023/11/15/petroleo-recua-com-expectativa-de-aumento-de-oferta-e-excedente-em-2024.ghtml" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Petróleo recua com expectativa de aumento de oferta e excedente em 2024</a>
      """,
      unsafe_allow_html=True)       
  st.markdown(
      """
      <a href="https://elpais.com/internacional/2024-10-05/el-jefe-de-la-agencia-internacional-de-la-energia-el-mercado-petrolero-esta-en-riesgo-y-puede-ponerse-peor.html" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      El jefe de la Agencia Internacional de la Energía: "El mercado petrolero está en riesgo, y puede ponerse peor</a>
      """,
      unsafe_allow_html=True)     


#################################################################################################################################################

  st.markdown('<h4>Tendências Futuras do Mercado de Petróleo</h4>', unsafe_allow_html=True)

  st.markdown('<h5>1. Aumento da Produção Global</h5>', unsafe_allow_html=True)
  st.write("""- **Projeções de Produção**: A Administração de Informação de Energia dos EUA (EIA) projeta que a produção mundial de petróleo atingirá **104,7 milhões de barris por dia em 2025**, superando as previsões anteriores.""")
  st.write("""- **Principais Contribuintes**: Espera-se que países como Estados Unidos, Canadá, Guiana e Argentina liderem esse aumento na produção.""")

  st.markdown('<h5>2. Crescimento Moderado da Demanda</h5>', unsafe_allow_html=True)
  st.write("""- **Revisões da OPEP**: A Organização dos Países Exportadores de Petróleo (OPEP) revisou para baixo suas previsões de crescimento da demanda global para 2024 e 2025, destacando a fraqueza em economias como China e Índia.""")
  st.write("""- **Fatores Contribuintes**: A desaceleração econômica e o aumento do uso de energias renováveis estão entre os principais motivos para a redução na demanda projetada.""")                   

  st.markdown('<h5>3. Pressão Descendente nos Preços</h5>', unsafe_allow_html=True)
  st.write("""- **Análises de Mercado**: Analistas do Citi preveem uma queda de até **20% nos preços do petróleo**, situando-o em torno de **US$ 60 por barril**, devido ao aumento da oferta e políticas fiscais mais favoráveis à indústria nos EUA.""")
  st.write("""- **Impacto das Políticas Energéticas dos EUA**: A vitória eleitoral de Donald Trump em 2024 trouxe expectativas de mudanças na política energética dos EUA, incluindo a liberação de grandes reservas de petróleo e a redução de impostos 
            para produtores de combustíveis fósseis, o que pode aumentar a oferta e pressionar os preços para baixo.""")            

  st.markdown('<h5>4. Riscos Geopolíticos e Econômicos</h5>', unsafe_allow_html=True)
  st.write("""- **Conflitos Regionais**: Tensões no Oriente Médio e outras regiões produtoras podem causar interrupções no fornecimento, afetando os preços.""")
  st.write("""- **Flutuações Cambiais**: A valorização ou desvalorização do dólar americano pode influenciar os preços do petróleo, tornando-o mais caro ou mais barato para países com outras moedas.""")            

  st.markdown('<h5>Conclusão</h5>', unsafe_allow_html=True)
  st.write("""As tendências futuras do mercado de petróleo indicam um cenário de aumento na produção global, crescimento moderado da demanda e possíveis pressões descendentes nos preços, influenciados por 
            políticas energéticas e fatores geopolíticos. É essencial que os stakeholders do setor permaneçam atentos a essas dinâmicas para adaptar suas estratégias conforme as mudanças no mercado.""")  

#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Gráfico de Boxplot por Mês
  plt.figure(figsize=(15, 6))
  sns.boxplot(x='month', y='y', data=df_boxplot)
  plt.title('Distribuição dos Precos por Mês')
  plt.xlabel('Mês')
  plt.ylabel('Preco')
  plt.show()
  """

  # Exibe o código na aplicação
  st.subheader("Análise por Mês:")
  st.code(codigo, language='python')

  # Implementação do código para carregar e exibir o gráfico
  try:
      # Importando as bibliotecas
      import streamlit as st
      import matplotlib.pyplot as plt
      import seaborn as sns
      import pandas as pd

      # Convertendo a coluna 'Data' para o tipo datetime
      dfb['Data'] = pd.to_datetime(dfb['Data'])

      # Criando as colunas 'year' e 'month'
      df_boxplot = dfb.copy()
      df_boxplot['month'] = df_boxplot['Data'].dt.month      
      
      # Criando o gráfico
      plt.figure(figsize=(15, 6))
      sns.boxplot(x='month', y='Preço', data=df_boxplot)
      plt.title('Distribuição dos Precos por Mês')
      plt.xlabel('Mês')
      plt.ylabel('Preço')
      plt.xticks(rotation=0)

      # Armazenando a figura e exibindo no Streamlit
      fig = plt.gcf()
      st.pyplot(fig)

  except Exception as e:
      st.error(f"Erro ao gerar gráficos: {e}")

  # Descrição dos gráficos
  st.write("""Neste gráfico, é possível observar o comportamento dos preços do petróleo ao longo dos meses:""")
  st.write("""- O mês de **julho (7)** se destaca por apresentar grandes variações em relação aos outros meses.""")  
  st.write("""- **Novembro, dezembro e janeiro** são os meses que tendem a registrar os menores preços ao longo dos anos, o que pode ser analisado mais a fundo para entender as possíveis causas.""")  
  st.write("""- No entanto, em média, os preços não apresentam variações significativas entre os meses, especialmente quando comparados ao comportamento geral ao longo de todos os anos.""")  

  st.write("""Vamos analisar o comportamento dos preços a partir de 2021 para compreender como eles se encontram atualmente""")  

#################################################################################################################################################

  st.markdown('<h4>Análise das Flutuações Mensais do Preço do Petróleo</h4>', unsafe_allow_html=True)

  st.markdown('<h5>Contexto Geral</h5>', unsafe_allow_html=True)
  st.write("""A análise dos dados históricos de preços do petróleo ao longo de várias décadas revelou tendências sazonais marcantes. 
            Os meses de novembro, dezembro e janeiro apresentam os menores preços médios, enquanto julho demonstra as maiores variações de preço. 
            Essa dinâmica é influenciada por fatores sazonais, eventos geopolíticos e mudanças na demanda e oferta globais.""")

  st.markdown('<h5>Análise dos Menores Preços em Novembro, Dezembro e Janeiro</h5>', unsafe_allow_html=True)
  st.write("""1. **Redução Sazonal da Demanda**""")
  st.write("""- Durante o inverno no hemisfério norte, a demanda por combustíveis de transporte diminui devido às condições climáticas adversas e menor atividade econômica.""")
  st.write("""- A redução do consumo impacta diretamente a pressão sobre os preços.""")

  st.write("""2. **Aumento dos Estoques**""")
  st.write("""- Refinarias frequentemente acumulam estoques antes do inverno para atender à demanda de aquecimento. Estoques elevados em períodos de baixa demanda contribuem para a queda nos preços.""")

  st.write("""3. **Decisões Estratégicas da OPEP**""")
  st.write("""- A Organização dos Países Exportadores de Petróleo (OPEP) frequentemente realiza reuniões em novembro e dezembro. Dependendo das decisões sobre níveis de produção, os preços podem ser ajustados para refletir a oferta futura.""")

  st.write("""**Exemplo Histórico:** Em dezembro de 2020, após um ano de baixa demanda devido à pandemia de COVID-19, os preços do petróleo permaneceram reduzidos devido aos altos estoques acumulados.""")

  st.markdown('<h5>Análise das Maiores Variações de Preço em Julho/h5>', unsafe_allow_html=True)
  st.write("""1. **Alta Demanda Sazonal**""")
  st.write("""- Julho coincide com o verão no hemisfério norte, onde há maior mobilidade e, consequentemente, aumento na demanda por combustíveis de transporte. Esse aumento cria maior volatilidade no mercado.""")

  st.write("""2. **Manutenções e Interrupções na Produção**""")
  st.write("""- Este período é frequentemente escolhido para manutenções planejadas em refinarias e instalações de produção, reduzindo temporariamente a oferta e elevando as flutuações nos preços.""")

  st.write("""3. **Eventos Geopolíticos e Econômicos**""")
  st.write("""- Conflitos regionais, tensões no Oriente Médio e incertezas econômicas frequentemente atingem o pico no meio do ano, exacerbando a volatilidade.""")
  st.write("""- A temporada de furacões no Golfo do México também contribui para interrupções na produção, influenciando significativamente os preços.""")

  st.write("""**Exemplo Histórico:** Em julho de 2008, os preços do petróleo atingiram picos históricos devido às tensões geopolíticas no Oriente Médio e especulações sobre a oferta global.""")

  st.markdown('<h5>Conclusão</h5>', unsafe_allow_html=True)
  st.write("""A análise das flutuações mensais dos preços do petróleo demonstra uma relação clara entre fatores sazonais e eventos externos. Os menores preços em novembro, dezembro e janeiro 
              refletem uma demanda reduzida e decisões estratégicas de produção, enquanto as variações em julho são impulsionadas por aumento de demanda, interrupções na produção e fatores geopolíticos.""")  
  st.write("""Compreender esses padrões é essencial para antecipar movimentos no mercado e orientar estratégias econômicas e industriais. Este conhecimento fornece uma base sólida para análises preditivas e tomadas de decisão fundamentadas no setor energético.""")  


#################################################################################################################################################

  # Código usado para leitura e exibição da tabela
  codigo = """
  # Filtrando os dados para os anos depois de 2020
  anos_desejados = [2021, 2022, 2023, 2024]
  df_boxplot_filtrado = df_boxplot[df_boxplot['year'].isin(anos_desejados)].copy()

  # Ajustando o mês para ser uma categoria ordenada
  df_boxplot_filtrado['month'] = df_boxplot_filtrado['month'].astype(str)
  meses_ordenados = [str(i) for i in range(1, 13)]  # Lista de meses em ordem crescente
  df_boxplot_filtrado['month'] = pd.Categorical(df_boxplot_filtrado['month'], categories=meses_ordenados, ordered=True)

  # Criando um boxplot por mês, agrupando os anos
  plt.figure(figsize=(15, 6))
  sns.boxplot(x='month', y='y', data=df_boxplot_filtrado)
  plt.title('Distribuição dos Preços por Mês (2021-2024)')
  plt.xlabel('Mês')
  plt.ylabel('Preço')
  plt.xticks(rotation=0)
  plt.show()
  """

  # Exibe o código na aplicação
  st.subheader("Observando os anos mais recentes:")
  st.code(codigo, language='python')

  try:
      # Importando as bibliotecas
      import streamlit as st
      import matplotlib.pyplot as plt
      import seaborn as sns
      import pandas as pd

      # Criando o DataFrame inicial com a coluna de data
      df_boxplot = dfb.copy()
      df_boxplot.reset_index(inplace=True)
      df_boxplot['Data'] = pd.to_datetime(df_boxplot['Data'])
      df_boxplot['year'] = df_boxplot['Data'].dt.year  # Adicionando a coluna 'year'
      df_boxplot['month'] = df_boxplot['Data'].dt.month  # Adicionando a coluna 'month'

      # Filtrando os dados para os anos desejados
      anos_desejados = [2021, 2022, 2023, 2024]
      df_boxplot_filtrado = df_boxplot[df_boxplot['year'].isin(anos_desejados)].copy()

      # Ajustando o mês para ser uma categoria ordenada
      df_boxplot_filtrado['month'] = df_boxplot_filtrado['month'].astype(str)
      meses_ordenados = [str(i) for i in range(1, 13)]  # Lista de meses em ordem crescente
      df_boxplot_filtrado['month'] = pd.Categorical(df_boxplot_filtrado['month'], categories=meses_ordenados, ordered=True)

      # Criando o gráfico
      plt.figure(figsize=(15, 6))
      sns.boxplot(x='month', y='Preço', data=df_boxplot_filtrado)
      plt.title('Distribuição dos Preços por Mês (2021-2024)')
      plt.xlabel('Mês')
      plt.ylabel('Preço')
      plt.xticks(rotation=0)

      # Armazenando a figura e exibindo no Streamlit
      fig = plt.gcf()
      st.pyplot(fig)
 

  except Exception as e:
      st.error(f"Erro ao gerar gráficos: {e}")

  # Descrição dos gráficos
  st.write("""Observamos que, em anos mais recentes, o comportamento dos preços apresenta algumas mudanças:""")
  st.write("""- **Dezembro** é o mês com os menores valores para o petróleo, além de mostrar poucas variações.""")  
  st.write("""- **Janeiro** se destaca por apresentar quedas significativas, o que pode torná-lo um mês interessante para compra, embora seja caracterizado por baixa estabilidade.""")  
  st.write("""- **Fevereiro** é o mês mais instável desse período, com variações acentuadas nos preços.""")  
  st.write("""- **Junho e julho** são meses marcados por aumentos expressivos nos valores do petróleo.""")    
  st.write("""- **Novembro** apresenta certa instabilidade, mas em menor grau quando comparado a fevereiro.""")    


#################################################################################################################################################

  st.write("""A análise dos preços do petróleo entre 2021 e 2024 revela padrões sazonais distintos em comparação com períodos anteriores. 
            Essas variações podem ser atribuídas a uma combinação de fatores sazonais, econômicos e geopolíticos que influenciaram a oferta e a demanda global de petróleo.""")

  st.markdown('<h5>1. Dezembro: Menores Valores e Baixa Volatilidade</h5>', unsafe_allow_html=True)
  st.write("""- **Redução Sazonal da Demanda**: Durante dezembro, especialmente no hemisfério norte, há uma diminuição na demanda por combustíveis devido às 
            férias de fim de ano e menor atividade industrial. Essa redução na demanda contribui para a queda nos preços e menor volatilidade.""")
  st.write("""- **Aumento dos Estoques**: As refinarias costumam acumular estoques antes do inverno para atender à demanda de aquecimento. No entanto, se a demanda for menor que o esperado, os estoques elevados podem levar a uma queda nos preços.""")

  st.markdown('<h5>2. Janeiro: Quedas Significativas e Baixa Estabilidade</h5>', unsafe_allow_html=True)
  st.write("""- **Ajustes Pós-Feriados**: Após o aumento de consumo durante as festas de fim de ano, janeiro geralmente apresenta uma correção na demanda, resultando em quedas nos preços.""")
  st.write("""- **Incertezas Econômicas**: O início do ano fiscal pode trazer incertezas econômicas e ajustes em políticas energéticas, contribuindo para a instabilidade dos preços.""")

  st.markdown('<h5>3. Fevereiro: Alta Instabilidade</h5>', unsafe_allow_html=True)
  st.write("""- **Clima Severo**: No hemisfério norte, fevereiro é marcado por condições climáticas extremas que podem afetar a produção e o transporte de petróleo, causando flutuações nos preços.""")
  st.write("""- **Eventos Geopolíticos**: Historicamente, tensões geopolíticas, como conflitos no Oriente Médio, têm ocorrido nesse período, influenciando a volatilidade dos preços.""")

  st.markdown('<h5>4. Junho e Julho: Aumentos Significativos nos Preços</h5>', unsafe_allow_html=True)
  st.write("""- **Alta Demanda Sazonal**: Esses meses correspondem ao verão no hemisfério norte, período de aumento significativo nas viagens e, consequentemente, na demanda por combustíveis. Essa elevação na demanda pode pressionar os preços para cima.""")
  st.write("""- **Manutenções e Interrupções na Produção**: Muitas instalações de produção e refino programam manutenções para os meses de verão, o que pode reduzir temporariamente a oferta e impactar os preços.""")

  st.markdown('<h5>5. Novembro: Instabilidade Moderada</h5>', unsafe_allow_html=True)
  st.write("""- **Preparativos para o Inverno**: As refinarias ajustam a produção para atender à demanda de aquecimento, o que pode causar flutuações nos preços.""")
  st.write("""- **Decisões da OPEP**: A Organização dos Países Exportadores de Petróleo (OPEP) realiza reuniões em novembro para definir políticas de produção. Dependendo das decisões tomadas, como aumentos de produção, os preços podem ser impactados negativamente.""")

  st.markdown('<h5>Conclusão</h5>', unsafe_allow_html=True)
  st.write("""As variações mensais nos preços do petróleo entre 2021 e 2024 refletem uma complexa interação de fatores sazonais, econômicos e geopolíticos. Compreender esses padrões é essencial para análises precisas e tomadas de decisão informadas no setor energético.""")

  st.write("""**Fontes**""")
  st.markdown(
      """
      <a href="https://www.gov.br/anp/pt-br/centrais-de-conteudo/publicacoes/anuario-estatistico/anuario-estatistico-brasileiro-do-petroleo-gas-natural-e-biocombustiveis-2024" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Anuário Estatístico Brasileiro do Petróleo, Gás Natural e Biocombustíveis 2024</a>
      """,
      unsafe_allow_html=True)  
  st.markdown(
      """
      <a href="https://www.ibp.org.br/observatorio-do-setor/evolucao-dos-precos-internacionais-do-petroleo-e-projecoes-para-2025/" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Evolução dos preços internacionais do petróleo e projeções para 2025</a>
      """,
      unsafe_allow_html=True)        
  st.markdown(
      """
      <a href="https://pt.tradingeconomics.com/commodity/brent-crude-oil" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      Brent - petróleo - Contrato Futuro - Preços | 1970-2024 Dados | 2025-2026 Previsão</a>
      """,
      unsafe_allow_html=True)        


  st.markdown('<h4>Dando continuidade, seguiremos com os modelos de machine learning</h4>', unsafe_allow_html=True)

elif menu == "Modelos":

  st.markdown('<h2 style="color:#e61859;">Modelos</h2>', unsafe_allow_html=True)
  st.image(
        "https://www.deeplearningbook.com.br/wp-content/uploads/2022/04/Machine-Learning-Guia-Definitivo.jpeg",
        use_container_width=True,
  )

  st.markdown('<h4>Selecione o modelo</h4>', unsafe_allow_html=True)
  # Abas para cada página
  tab1, tab2, tab3, tab4 = st.tabs(["Prophet", "LSTM", "RNN Regressor","Conclusão"])

  # Conteúdo de cada página
  with tab1:

    st.markdown('<h2 style="color:#e61859;">Prophet</h2>', unsafe_allow_html=True)

    # Código usado para leitura e exibição da tabela
    codigo = """
    dfp = dfb.sort_index()
    dfp = dfp.reset_index()  
  
    # Aplicando o modelo Prophet
    model_prophet = Prophet()
    model_prophet.fit(dfp)
    future = model_prophet.make_future_dataframe(periods=90, freq='B') # Vamos usar um periodo de 90 dias para frente, com uma frequencia B de dias uteis
    forecast = model_prophet.predict(future) # Criando as previsoes.
    """

    # Exibe o código na aplicação
    st.code(codigo, language='python')


#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error # Importando a lib para termos os parametros de como foi o forecast
    """

    # Exibe o código na aplicação
    st.code(codigo, language='python')


#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    model_prophet.plot(forecast)

    """

    # Exibe o código na aplicação
    st.code(codigo, language='python')

    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "plot1_prophet.png"  # Substitua pelo nome ou caminho completo da imagem


    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)
    

#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    model_prophet.plot_components(forecast)
    """

    # Exibe o código na aplicação
    st.code(codigo, language='python')

    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "plot2_prophet.png"  # Substitua pelo nome ou caminho completo da imagem


    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)



    st.write("""Como observado na análise exploratória de dados (EDA), a aplicação do modelo Prophet reforça os insights 
            previamente identificados sobre a análise mensal: dezembro e janeiro apresentam uma tendência a valores mais baixos, 
            conforme já explicado anteriormente.""")

    st.write("""O modelo também oferece uma nova perspectiva ao analisar o comportamento semanal. Os preços apresentam uma 
            tendência inicial de queda desde o início da semana, atingindo o menor valor médio às quartas-feiras. 
            Após esse ponto, os preços começam a subir gradualmente, alcançando os valores mais altos. 
            Essa análise reflete a dinâmica semanal do mercado, considerando que não há cotações realizadas durante os finais de semana.""")            

#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    import numpy as np
    # Divisão dos dados em treino e teste
    train = dfp.iloc[:-30]  # Treino: Dados até os últimos 30 pontos
    test = dfp.iloc[-30:]   # Teste: Últimos 30 pontos

    # Ajustando o modelo Prophet com o conjunto de treino
    model_normal = Prophet()
    model_normal.fit(train)

    # Criando datas futuras para prever o conjunto de teste
    future = model_normal.make_future_dataframe(periods=30, freq='D')  # Previsão para o período de teste
    forecast = model_normal.predict(future)

    # Filtrar apenas as previsões correspondentes ao conjunto de teste
    forecast_test = forecast.iloc[-30:]

    # Extraindo as previsões e valores reais
    y_true = test['y'].values  # Valores reais no conjunto de teste
    y_pred = forecast_test['yhat'].values  # Valores previstos no mesmo período

    # Calculando métricas
    prophet_mae = mean_absolute_error(y_true, y_pred)
    prophet_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    prophet_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"Prophet MAE: {prophet_mae}, RMSE: {prophet_rmse}, MAPE: {prophet_mape}")  
   """

    # Exibe o código na aplicação
    st.code(codigo, language='python')

    try:
      st.write(f"Prophet MAE: 15.80, RMSE: 16.02, MAPE: 21.16%")  

    except Exception as e:
      st.error(f"Erro ao gerar gráficos: {e}")


#################################################################################################################################################

    st.markdown('<h4>Interpretação das Métricas do Modelo Prophet</h4>', unsafe_allow_html=True)
    st.write("""A análise das métricas de desempenho do modelo Prophet, com base nos valores fornecidos, é apresentada abaixo:""")

    st.markdown('<h5>1. MAE (Mean Absolute Error): 15,80</h5>', unsafe_allow_html=True)
    st.write("""- **Definição**:""")
    st.write("""- Representa o erro médio absoluto nas previsões, indicando que, em média, o modelo apresenta um desvio de **15,80 unidades** em relação aos valores reais.""")

    st.write("""- **Análise**:""")
    st.write("""- Esse valor pode ser considerado aceitável, uma vez que o preço atual do petróleo está em torno de 75,00 dólares por barril, o que representa um erro médio relativo moderado dentro desse contexto.""")

    st.markdown('<h5>2. RMSE (Root Mean Squared Error): 16,02</h5>', unsafe_allow_html=True)
    st.write("""- **Definição**:""")
    st.write("""- Mede a raiz quadrada da média dos erros ao quadrado, penalizando mais os erros maiores.""")
    st.write("""- O RMSE próximo ao MAE sugere que não há outliers significativos no conjunto de dados.""")

    st.write("""- **Análise**:""")
    st.write("""- O modelo apresenta um desempenho consistente, com um erro médio de aproximadamente **16,02 unidades**, o que está alinhado com o valor do MAE.""")
    st.write("""- A proximidade entre RMSE e MAE é um bom indicativo de que os erros estão distribuídos de forma uniforme.""")

    st.markdown('<h5>3. MAPE (Mean Absolute Percentage Error): 21,16%</h5>', unsafe_allow_html=True)
    st.write("""- **Definição**:""")
    st.write("""- Reflete o erro médio percentual em relação aos valores reais, indicando que o modelo erra, em média, **21,16%** do valor real ao realizar as previsões.""")

    st.write("""- **Análise**:""")
    st.write("""- O desempenho é aceitável, mas há margem para melhorias, especialmente considerando critérios gerais para MAPE:""")
    st.write("""- **MAPE < 10%**: Excelente precisão.""")
    st.write("""- **10% ≤ MAPE ≤ 20%**: Precisão aceitável.""")
    st.write("""- **MAPE > 20%**: Necessidade de ajustes no modelo.""")
    st.write("""- O valor de **21,16%** posiciona o modelo de uma forma que precise de ajustes.""")

    st.markdown('<h4>Análise Geral</h4>', unsafe_allow_html=True)
    st.write("""- **Consistência entre MAE e RMSE**:""")
    st.write("""- A pequena diferença entre essas métricas indica uma boa estabilidade no modelo, sem a presença de erros extremos que poderiam distorcer os resultados.""")

    st.write("""- **Impacto Geral do Erro**:""")
    st.write("""- Em termos absolutos (MAE) e percentuais (MAPE), o desempenho do modelo pode se aceitável, considerando a complexidade inerente à previsão de preços de commodities como o petróleo.""")
    st.write("""- Contudo, se as previsões forem usadas para decisões de alta sensibilidade, pode ser interessante explorar ajustes no modelo ou incorporar mais variáveis explicativas.""")


#################################################################################################################################################

    st.markdown('<h3>Validacao Cruzada Prophet</h3>', unsafe_allow_html=True)

    # Código usado para leitura e exibição da tabela
    codigo = """
    import plotly.graph_objects as go # Importando a Lib para criacao das vizualizacoes

    # Ajustar o modelo Prophet com sazonalidades personalizadas e remoção de sazonalidades padrão
    model_custom = Prophet(
      yearly_seasonality=False,
      weekly_seasonality=False,
      daily_seasonality=False,
      interval_width=0.8,  # Ajustando o intervalo de confiança para 80%
      changepoint_prior_scale=0.05 # Tornar o modelo mais sensível a mudanças recentes
    )

    # Adicionar sazonalidades específicas (anual e semanal, como exemplo)
    model_custom.add_seasonality(name='yearly', period=365.25, fourier_order=10)
    model_custom.add_seasonality(name='monthly', period=30.5, fourier_order=3)

    # Ajustar o modelo nos dados limpos
    model_custom.fit(dfp)
    """

    # Exibe o código na aplicação
    st.code(codigo, language='python')


#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    from prophet.diagnostics import cross_validation, performance_metrics

    # Fazer validação cruzada para avaliar o modelo ajustado
    df_cv = cross_validation(model_custom, initial='730 days', period='90 days', horizon='30 days')

    # Obter métricas de desempenho
    dfcvp = performance_metrics(df_cv)
    """

    # Exibe o código na aplicação
    st.code(codigo, language='python')
  

#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Visualizar métricas
    print(dfcvp)
    """

    # Exibe o código na aplicação
    st.code(codigo, language='python')

    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "performance_metrics.jpg"  # Substitua pelo nome ou caminho completo da imagem

    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)

#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Visualizar métricas
    df_cv.head()
    """

    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')

    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "df_cv.jpg"  # Substitua pelo nome ou caminho completo da imagem

    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)

#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Gráfico do MAPE
    mape = dfcvp['mape']
    fig_mape = go.Figure()
    fig_mape.add_trace(go.Scatter(x=mape.index, y=mape, mode='lines', name='MAPE'))
    fig_mape.update_layout(title='MAPE da Validação Cruzada do Prophet',
                        xaxis_title='Index',
                        yaxis_title='MAPE')
    fig_mape.show()
    """

    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')


    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "newplot.png"  # Substitua pelo nome ou caminho completo da imagem

    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)

    st.write("""Obtivemos um MAPE (Erro Absoluto Percentual Médio) variando entre 21% e 24% nos primeiros 30 dias de previsão. 
            O gráfico do MAPE apresenta flutuações consistentes ao longo dos dias, refletindo a dinâmica dos dados analisados. 
            Embora o valor possa ser considerado elevado em alguns contextos, dentro do escopo e do comportamento dos dados deste projeto, 
            ele se mostra adequado para atender aos objetivos da análise, especialmente considerando as limitações do modelo e o foco nas tendências gerais.""")

#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Criar gráfico interativo das previsões ajustadas
    fig_adjusted = go.Figure()

    # Adicionar valores reais
    fig_adjusted.add_trace(go.Scatter(
      x=dfp['ds'], y=dfp['y'],
      mode='lines', name='Valores Reais',
      line=dict(color='blue')
    ))

    # Adicionar valores previstos
    fig_adjusted.add_trace(go.Scatter(
      x=df_cv['ds'], y=df_cv['yhat'],
      mode='lines', name='Previsão (yhat)',
      line=dict(color='orange')
    ))

    # Adicionar intervalo de confiança
    fig_adjusted.add_trace(go.Scatter(
      x=pd.concat([df_cv['ds'], df_cv['ds'][::-1]]),
      y=pd.concat([df_cv['yhat_upper'], df_cv['yhat_lower'][::-1]]),
      fill='toself', fillcolor='rgba(255, 165, 0, 0.2)',
      line=dict(color='rgba(255, 165, 0, 0)'),
      name='Intervalo de Confiança'
    ))

    # Personalizar layout
    fig_adjusted.update_layout(
      title='Previsão Ajustada do Modelo Prophet',
      xaxis_title='Data',
      yaxis_title='Preço do Petróleo Brent (US$)',
      template='plotly_white'
    )

    fig_adjusted.show()
    """

    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')


    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "newplot2.png"  # Substitua pelo nome ou caminho completo da imagem

    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)

    st.write("""Neste gráfico, podemos observar, em um contexto geral, o comportamento do forecast e do intervalo de confiança ao longo do tempo com base nos dados analisados. 
            Devido à presença de numerosos outliers e flutuações significativas em determinados períodos, o forecast apresenta desvios consideráveis, tornando-se, 
            em alguns casos, distante do que seria esperado para uma previsão confiável. Essa análise levanta questionamentos sobre a adequação do modelo para 
            previsões de curto prazo, indicando a necessidade de ajustes ou complementação com outras abordagens para melhorar sua precisão.""")

#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Gerar datas futuras para 90 dias após a última data do conjunto original
    future_90 = model_custom.make_future_dataframe(periods=90, freq='D')

    # Fazer previsões
    forecast_90 = model_custom.predict(future_90)

    # Filtrar previsões apenas para datas futuras (após a última data no conjunto original)
    forecast_90_future = forecast_90[forecast_90['ds'] > dfp['ds'].max()]

    # Criar o gráfico interativo
    fig_90_days = go.Figure()

    # Adicionar linha para a previsão central (yhat)
    fig_90_days.add_trace(go.Scatter(
      x=forecast_90_future['ds'], y=forecast_90_future['yhat'],
      mode='lines', name='Previsão (yhat)',
      line=dict(color='orange')
    ))

    # Adicionar intervalo de confiança (yhat_upper e yhat_lower)
    fig_90_days.add_trace(go.Scatter(
      x=pd.concat([forecast_90_future['ds'], forecast_90_future['ds'][::-1]]),
      y=pd.concat([forecast_90_future['yhat_upper'], forecast_90_future['yhat_lower'][::-1]]),
      fill='toself', fillcolor='rgba(255, 165, 0, 0.2)',
      line=dict(color='rgba(255, 165, 0, 0)'),
      name='Intervalo de Confiança'
    ))

    # Personalizar o layout do gráfico
    fig_90_days.update_layout(
      title='Previsão de 90 Dias Após a Última Data da Tabela',
      xaxis_title='Data',
      yaxis_title='Preço do Petróleo Brent (US$)',
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
      template='plotly_white'
    )

    # Exibir o gráfico
    fig_90_days.show()
    """

    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')


    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "newplot3.png"  # Substitua pelo nome ou caminho completo da imagem

    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)

    st.write("""Esses são os valores projetados para um período de 90 dias após o último registro disponível em nossa base de dados. 
            A projeção indica uma tendência de queda até dezembro, seguida por uma estabilização e um aumento gradativo a partir da segunda quinzena de janeiro.""")

    st.write("""**Análise:**""")
    st.write("""De forma geral, o Prophet com validação cruzada, embora mais robusto que sua versão padrão, mostrou limitações significativas para previsões de curto prazo. 
            Isso ocorre porque, ao priorizar tendências, o modelo não lida bem com variações abruptas decorrentes de fatores políticos e econômicos que influenciam diretamente a curva. 
            No entanto, sua capacidade de capturar padrões gerais o torna uma ferramenta confiável como referência para projeções de longo prazo.""")  


#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Ajustar o modelo Prophet
    model_custom = Prophet(
      yearly_seasonality=False,  # Desativar sazonalidade anual padrão
      weekly_seasonality=False,  # Sem sazonalidade semanal
      daily_seasonality=False,   # Sem sazonalidade diária
      interval_width=0.8,        # Intervalo de confiança em 80%
      changepoint_prior_scale=0.05  # Tornar o modelo mais sensível a mudanças recentes
    )

    # Adicionar sazonalidades específicas
    model_custom.add_seasonality(name='yearly', period=365.25, fourier_order=10)  # Sazonalidade anual
    model_custom.add_seasonality(name='monthly', period=30.5, fourier_order=3)   # Sazonalidade mensal (Fourier order ajustado)

    # Ajustar o modelo usando apenas os dados recentes (opcional, se necessário)
    df_recent = dfp[dfp['ds'] > '2020-01-01']  # Ajustar a data limite conforme necessário
    model_custom.fit(df_recent)

    # Fazer validação cruzada com novos parâmetros
    df_cv2 = cross_validation(
      model_custom,
      initial='730 days',  # Usar 2 anos como base inicial para o treino
      period='30 days',    # Mover o ponto de corte mensalmente
      horizon='90 days'    # Previsões para 90 dias à frente
    )

    # Obter métricas de desempenho
    dfcvp2 = performance_metrics(df_cv2)
    """

    # Exibe o código na aplicação
    st.subheader("Vamos ajustar o código para avaliar se ele consegue capturar com maior precisão os valores mais recentes:")
    st.code(codigo, language='python')

#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Exibir as primeiras linhas das métricas
    print(dfcvp2)
    """

    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')


    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "dfcvp2.jpg"  # Substitua pelo nome ou caminho completo da imagem

    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)


#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Gráfico do MAPE
    mape = dfcvp2['mape']
    fig_mape = go.Figure()
    fig_mape.add_trace(go.Scatter(x=mape.index, y=mape, mode='lines', name='MAPE'))
    fig_mape.update_layout(title='MAPE da Validação Cruzada do Prophet',
                        xaxis_title='Index',
                        yaxis_title='MAPE')
    fig_mape.show()
    """

    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')


    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "newplot4.png"  # Substitua pelo nome ou caminho completo da imagem

    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)

    st.write("""Com a alteração dos parâmetros no Prophet, obtivemos um MAPE variando entre 7% e 16% ao longo de 90 dias. Esse resultado representa uma melhoria significativa, tornando o modelo mais adequado para previsões de curto prazo.""")


#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Gerar datas futuras para previsão de 90 dias
    future_90 = model_custom.make_future_dataframe(periods=90, freq='D')

    # Fazer previsões
    forecast_90 = model_custom.predict(future_90)

    # Filtrar previsões apenas para datas futuras (após a última data no conjunto original)
    forecast_90_future = forecast_90[forecast_90['ds'] > df_recent['ds'].max()]

    # Criar o gráfico interativo
    fig_90_days = go.Figure()

    # Adicionar linha para a previsão central (yhat)
    fig_90_days.add_trace(go.Scatter(
      x=forecast_90_future['ds'], y=forecast_90_future['yhat'],
      mode='lines', name='Previsão (yhat)',
      line=dict(color='orange')
    ))

    # Adicionar intervalo de confiança (yhat_upper e yhat_lower)
    fig_90_days.add_trace(go.Scatter(
      x=pd.concat([forecast_90_future['ds'], forecast_90_future['ds'][::-1]]),
      y=pd.concat([forecast_90_future['yhat_upper'], forecast_90_future['yhat_lower'][::-1]]),
      fill='toself', fillcolor='rgba(255, 165, 0, 0.2)',
      line=dict(color='rgba(255, 165, 0, 0)'),
      name='Intervalo de Confiança'
    ))

    # Personalizar o layout do gráfico
    fig_90_days.update_layout(
      title='Previsão de 90 Dias Após a Última Data',
      xaxis_title='Data',
      yaxis_title='Preço do Petróleo Brent (US$)',
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
      template='plotly_white'
    )

    # Exibir o gráfico
    fig_90_days.show()
    """

    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')


    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "newplot5.png"  # Substitua pelo nome ou caminho completo da imagem

    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)

    st.write("""Neste gráfico, com os ajustes realizados para refinar o Prophet, é possível observar que a previsão apresenta valores mais suaves e 
            próximos da realidade no curto prazo. O gráfico inicia com um valor de 73,25 dólares, indicando uma queda até a metade de dezembro, 
            seguida por uma recuperação gradual até o final de janeiro. Essa abordagem se mostra mais assertiva para cenários que demandam maior 
            precisão no curto prazo, em contraste com análises focadas em tendências de longo prazo.""")

    st.write("""**Análise:**""")
    st.write("""Para obter previsões com maior precisão no curto prazo, é mais eficaz utilizar esses parâmetros ajustados, que permitem compreender 
            melhor as variações no preço do petróleo nos próximos meses. No entanto, devido à limitação de dados disponíveis para o treinamento e ao 
            foco do Prophet em capturar tendências, mesmo com os ajustes realizados, o modelo pode apresentar erros consideráveis em projeções de longo prazo.""")  

#################################################################################################################################################

    st.markdown('<h2 style="color:#e61859;">Conclusão da Análise: Prophet e a Previsão do Preço do Petróleo Brent</h2>', unsafe_allow_html=True)

    st.write("""A análise realizada utilizando o modelo **Prophet** demonstrou que é possível obter resultados robustos para previsões de longo prazo e 
            previsões aceitáveis para horizontes de curto prazo, desde que configurado adequadamente. Os resultados foram baseados em uma abordagem sistemática, ,
            com destaque para a utilização de **validação cruzada**, que garantiu maior confiabilidade nos resultados. A seguir, detalhamos as conclusões principais:""")

    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "resultados_prophet.jpg"  # Substitua pelo nome ou caminho completo da imagem

    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)

    st.write("""**Contexto Geral da Previsão**""")
    st.write("""O preço do petróleo Brent é influenciado por diversos fatores externos, como eventos geopolíticos, decisões da OPEP+ e a transição energética. 
            Modelos de previsão baseados apenas em séries temporais, como o Prophet, podem não capturar completamente a complexidade desses fatores, 
            mas são eficazes para identificar padrões históricos e projetar tendências.""")

    st.write("""1. **Previsão de Longo Prazo**:""")
    st.write("""- A abordagem com **validação cruzada** provou-se adequada para projeções de longo prazo, entregando resultados robustos e confiáveis, especialmente na identificação de tendências estruturais.""")
    st.write("""- Esse comportamento é particularmente útil em contextos onde o foco está em decisões estratégicas de planejamento.""")

    st.write("""2. **Previsão de Curto Prazo**:""")
    st.write("""- Apesar da volatilidade intrínseca do mercado, os resultados do Prophet após ajustes, alcançaram um **MAPE (Mean Absolute Percentage Error)** aceitável, sugerindo que o modelo é útil para projeções operacionais.""")
    st.write("""- No entanto, a precisão é limitada em cenários de alta instabilidade, como crises geopolíticas ou choques econômicos abruptos.""")


    st.write("""**Desafios e Limitações Encontradas**""")
    st.write("""Apesar dos resultados, os seguintes desafios foram identificados:""")
    st.write("""- **Ausência de Variáveis Exógenas**: O modelo Prophet, por natureza, trabalha com dados da série temporal histórica e 
            não incorpora fatores exógenos diretamente, como decisões políticas ou climáticas. Esses fatores podem causar discrepâncias em mercados altamente sensíveis, como o de petróleo.""")
    st.write("""- **Volatilidade de Curto Prazo**: Eventos imprevisíveis, como pandemias ou mudanças bruscas na demanda, ainda representam um desafio para modelos puramente baseados em séries temporais.""")
    st.write("""- **Mudanças Estruturais**: A transição energética e mudanças no comportamento de consumo são fatores que o Prophet pode não capturar adequadamente sem ajustes específicos ou integração com outros modelos.""")


    st.write("""**Recomendações**""")
    st.write("""1. **Incorporar Variáveis Exógenas**:""")
    st.write("""- Para melhorar a previsão em contextos de alta volatilidade, recomenda-se a inclusão de variáveis externas relevantes (ex.: produção da OPEP, eventos climáticos, estoques globais de petróleo).""")

    st.write("""2. **Híbridos de Modelos**:""")
    st.write("""- A combinação do Prophet com algoritmos de aprendizado de máquina pode aumentar a capacidade preditiva, especialmente para capturar padrões mais complexos e não lineares.""")

    st.write("""3. **Foco em Análises Sazonais e de Longo Prazo**:""")
    st.write("""- O Prophet demonstrou forte capacidade em capturar tendências gerais e padrões sazonais, sendo ideal para análises de planejamento e estratégias de médio a longo prazo.""")

    st.write("""No entanto, esse nível de detalhamento vai além do escopo desta análise, cujo objetivo principal é direcionar e contextualizar a situação e as tendências do preço do petróleo de forma objetiva e alinhada ao propósito do projeto.""")


  with tab2:
    
    st.markdown('<h2 style="color:#e61859;">LSTM</h2>', unsafe_allow_html=True)

    # Código usado para leitura e exibição da tabela
    codigo = """
    #Criando um novo dataframe usando apenas a coluna de fechamento
    data = dfp.filter(['y'])

    #Convertendo o dataframe para um array numpy
    dataset = data.values

    training_data_len = int(np.ceil( len(dataset) * .95 ))

    training_data_len
    """

    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')

    try:
      # Importando as bibliotecas
      st.write('10728')

      
    except Exception as e:
        st.error(f"Erro ao gerar gráficos: {e}")

#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    scaled_data
    """
    
    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')

    try:
      st.write("""array([[0.07067112],
       [0.0693363 ],
       [0.07007786],
       ...,
       [0.47571376],
       [0.4785317 ],
       [0.48787542]])""")

      
    except Exception as e:
        st.error(f"Erro ao gerar gráficos: {e}")        

#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    train_data = scaled_data[0:int(training_data_len), :]

    #Dividindo os dados em conjuntos de dados x_train e y_train
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            print(x_train)
            print(y_train)
            print()

    #Convertendo x_train e y_train em matrizes numpy
    x_train, y_train = np.array(x_train), np.array(y_train)

    #Remodelando os dados
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # x_train.shape
    """
    
    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')

    try:
      st.write("""[array([0.07067112, 0.0693363 , 0.07007786, 0.07044865, 0.07067112,
       0.07044865, 0.07044865, 0.07030033, 0.07081943, 0.0710419 ,
       0.07156099, 0.07178346, 0.07081943, 0.07156099, 0.07178346,
       0.07178346, 0.0710419 , 0.07178346, 0.07267334, 0.07363737,
       0.07378569, 0.07378569, 0.07415647, 0.07267334, 0.07156099,
       0.07119021, 0.07400816, 0.07452725, 0.07400816, 0.07326659,
       0.07526882, 0.07586207, 0.07697442, 0.07712273, 0.07697442,
       0.07845755, 0.07882833, 0.07956989, 0.07994067, 0.08290693,
       0.08379681, 0.08550241, 0.08490916, 0.08342603, 0.08327772,
       0.0819429 , 0.07771598, 0.0756396 , 0.07919911, 0.08068224,
       0.08231368, 0.08105302, 0.08787542, 0.08565072, 0.07934742,
       0.07897664, 0.07823508, 0.07660363, 0.07675195, 0.07712273])]
       [0.07638116425658138]""")

      st.write("""[array([0.07067112, 0.0693363 , 0.07007786, 0.07044865, 0.07067112,
       0.07044865, 0.07044865, 0.07030033, 0.07081943, 0.0710419 ,
       0.07156099, 0.07178346, 0.07081943, 0.07156099, 0.07178346,
       0.07178346, 0.0710419 , 0.07178346, 0.07267334, 0.07363737,
       0.07378569, 0.07378569, 0.07415647, 0.07267334, 0.07156099,
       0.07119021, 0.07400816, 0.07452725, 0.07400816, 0.07326659,
       0.07526882, 0.07586207, 0.07697442, 0.07712273, 0.07697442,
       0.07845755, 0.07882833, 0.07956989, 0.07994067, 0.08290693,
       0.08379681, 0.08550241, 0.08490916, 0.08342603, 0.08327772,
       0.0819429 , 0.07771598, 0.0756396 , 0.07919911, 0.08068224,
       0.08231368, 0.08105302, 0.08787542, 0.08565072, 0.07934742,
       0.07897664, 0.07823508, 0.07660363, 0.07675195, 0.07712273]),""")     

      st.write("""array([0.0693363 , 0.07007786, 0.07044865, 0.07067112, 0.07044865,
       0.07044865, 0.07030033, 0.07081943, 0.0710419 , 0.07156099,
       0.07178346, 0.07081943, 0.07156099, 0.07178346, 0.07178346,
       0.0710419 , 0.07178346, 0.07267334, 0.07363737, 0.07378569,
       0.07378569, 0.07415647, 0.07267334, 0.07156099, 0.07119021,
       0.07400816, 0.07452725, 0.07400816, 0.07326659, 0.07526882,
       0.07586207, 0.07697442, 0.07712273, 0.07697442, 0.07845755,
       0.07882833, 0.07956989, 0.07994067, 0.08290693, 0.08379681,
       0.08550241, 0.08490916, 0.08342603, 0.08327772, 0.0819429 ,
       0.07771598, 0.0756396 , 0.07919911, 0.08068224, 0.08231368,
       0.08105302, 0.08787542, 0.08565072, 0.07934742, 0.07897664,
       0.07823508, 0.07660363, 0.07675195, 0.07712273, 0.07638116])]
       [0.07638116425658138, 0.07526881720430108]""")     

      
    except Exception as e:
        st.error(f"Erro ao gerar gráficos: {e}")    

#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    #Compilando o modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    #Treinando o modelo
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    """
    
    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')


#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    #Criar o conjunto de dados de teste
    #Criar o conjunto de dados de teste. Criar uma nova matriz contendo valores escalonados do índice 1543 a 2002
    test_data = scaled_data[training_data_len - 60: , :]

    #Criar os conjuntos de dados x_test e y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    #Converter os dados em um array numpy
    x_test = np.array(x_test)

    #Remodelar os dados
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    #Obter os valores de preços previstos dos modelos
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    #Obter a raiz do erro quadrático médio (RMSE)
    mape = mean_absolute_percentage_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(f"MAPE: {mape:.2f}%")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    """
    
    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')

    try:
      st.write("""MAPE: 0.02%""")
      st.write("""MAE: 1.43""")
      st.write("""RMSE: 1.88""")
                  
      
    except Exception as e:
        st.error(f"Erro ao gerar gráficos: {e}")        



#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Plot
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    #Visualize
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Valor de fechamento em USD ($)', fontsize=18)
    plt.plot(train['y'])
    plt.plot(valid[['y', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    """
    
    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')

    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "plot1_lstm.png"  # Substitua pelo nome ou caminho completo da imagem

    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Stre
    st.image(imagem)

    st.markdown('<h2 style="color:#e61859;">Conclusão da Análise: LSTM</h2>', unsafe_allow_html=True)

    st.write("""O modelo LSTM apresentou um desempenho satisfatório na previsão dos preços do petróleo Brent, conforme evidenciado pelas seguintes métricas de avaliação:""")
    st.write("""- **RMSE (Erro Quadrático Médio):** 1,88 – O valor relativamente baixo do RMSE indica que o modelo possui um erro médio 
             pequeno em relação aos preços reais, o que demonstra uma boa capacidade preditiva, especialmente em relação a variações absolutas no preço.""")

    st.write("""- **MAE (Erro Médio Absoluto):** 1,43 – O MAE reforça a precisão do modelo ao indicar que, em média, a diferença entre os valores previstos e os 
              reais é de apenas 1,43 unidades monetárias (USD). Isso representa uma boa precisão para séries temporais financeiras.""")

    st.write("""- **MAPE (Erro Percentual Médio Absoluto):** 2% – Com um MAPE de apenas 2%, o modelo apresenta um nível de erro percentual bastante baixo, 
              o que significa que, em média, as previsões diferem apenas 2% dos valores reais. Isso é um indicativo de que o modelo consegue capturar bem a dinâmica dos preços com um alto grau de precisão.""")

    st.write("""**Pontos positivos:**""")
    st.write("""1. **Baixo erro absoluto e percentual**:""")
    st.write("""- As métricas de RMSE, MAE e MAPE indicam que o modelo consegue prever o preço do petróleo Brent com precisão, especialmente em cenários de estabilidade e pequenas variações.""")

    st.write("""2. **Consistência em diferentes períodos**:""")
    st.write("""- A curva de previsão segue de forma estável as variações da série histórica, mostrando que o modelo consegue generalizar bem tanto para períodos de alta quanto de baixa volatilidade.""")

    st.write("""**Limitações observadas**""")
    st.write("""1. **Erro em momentos de alta volatilidade**:""")
    st.write("""- Apesar das métricas gerais serem boas, o modelo ainda pode apresentar erros ligeiramente maiores em períodos de alta oscilação dos preços, conforme observado visualmente.""")

    st.write("""2. **Dependência exclusiva da série histórica**:""")
    st.write("""- O modelo utiliza apenas dados históricos do preço do petróleo, o que pode limitar sua precisão em eventos inesperados ou choques externos.""")      

    st.write("""De forma geral, o modelo LSTM demonstrou ser uma ferramenta eficaz para a previsão dos preços do petróleo Brent, com métricas que indicam alta precisão.""")    

  

#################################################################################################################################################
  
  with tab3:
    st.markdown('<h2 style="color:#e61859;">RNN Regressor</h2>', unsafe_allow_html=True)  

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Preparação dos dados
    y = dfp['y'].values

    # Definindo uma janela de tempo (número de steps passados usados como entrada)
    window_size = 10

    # Criando os conjuntos de entrada (X) e saída (y) com base na janela de tempo
    X, y_rnn = [], []
    for i in range(len(y) - window_size):
        X.append(y[i:i+window_size])
        y_rnn.append(y[i+window_size])

    X = np.array(X)
    y_rnn = np.array(y_rnn)

    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y_rnn, test_size=0.2, shuffle=False)

    # Ajustando os dados para a entrada da RNN (adicionando uma dimensão)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Construção do modelo RNN
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)  # Saída de um único valor (regressão)
    ])

    # Compilando o modelo
    model.compile(optimizer='adam', loss='mse')

    # Treinando o modelo
    history = model.fit(X_train, y_train, epochs=1, batch_size=1, validation_split=0.2, verbose=1)

    # Fazendo previsões
    y_pred = model.predict(X_test)
    """

    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')


#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    # Calculando as métricas de erro
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Exibindo as métricas
    print(f"MAPE: {mape:.2f}%")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    """

    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')

    try:
      # Importando as bibliotecas
      st.write('MAPE: 0.02%')
      st.write('MAE: 1.46')
      st.write('RMSE: 2.05')

      
    except Exception as e:
        st.error(f"Erro ao gerar gráficos: {e}")    


#################################################################################################################################################

    # Código usado para leitura e exibição da tabela
    codigo = """
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, label='Valores Reais', color='blue')
    plt.plot(range(len(y_pred)), y_pred, label='Previsões', color='orange', linestyle='dashed')
    plt.title('Comparação entre Valores Reais e Previsões (RNN)')
    plt.xlabel('Amostras (Índice do Teste)')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    """
    
    # Exibe o código na aplicação
    #st.subheader("Vamos usar a validacao cruzada para tentarmos obter valores melhores das metricas do modelo:")
    st.code(codigo, language='python')

    from PIL import Image

    # Caminho da imagem no computador
    caminho_imagem = "plot_rnn.png"  # Substitua pelo nome ou caminho completo da imagem

    # Abrir a imagem com o Pillow
    imagem = Image.open(caminho_imagem)

    # Exibir a imagem no Streamlit
    st.image(imagem)


    st.markdown('<h2 style="color:#e61859;">Conclusão da Análise: RNN Regressor</h2>', unsafe_allow_html=True)

    st.write("""O modelo RNN Regressor apresentou resultados consistentes na previsão dos preços do petróleo Brent, conforme indicado pelas métricas de desempenho:""")
    st.write("""- **RMSE (Erro Quadrático Médio):** 2,05 – Embora seja ligeiramente superior ao do modelo LSTM, esse valor ainda indica uma boa precisão geral nas previsões, com erros moderados em relação aos valores reais.""")

    st.write("""- **MAE (Erro Médio Absoluto):** 1,46 – O erro absoluto médio indica que a diferença média entre os valores reais e previstos foi de aproximadamente 1,46 unidades monetárias (USD), o que demonstra uma boa precisão no contexto de variações de preço do petróleo.""")

    st.write("""- **MAPE (Erro Percentual Médio Absoluto):** 2% – Assim como o modelo LSTM, o RNN apresenta um erro percentual baixo, mostrando que as previsões diferem, em média, apenas 2% dos valores reais.""")

    st.write("""**Pontos positivos:**""")
    st.write("""1. **Acompanhamento fiel da série histórica**:""")
    st.write("""- Conforme observado no gráfico, o modelo RNN consegue seguir com precisão a tendência dos valores reais, especialmente em períodos de estabilidade e variação moderada.""")

    st.write("""2. **Baixa variação entre valores reais e previstos**:""")
    st.write("""- As linhas de previsão estão bem próximas dos valores reais, o que indica que o modelo é eficiente para prever pequenas oscilações no preço.""")

    st.write("""**Limitações observadas**""")
    st.write("""1. **Sensibilidade a picos extremos**:""")
    st.write("""- Embora o modelo consiga capturar bem as tendências gerais, ele pode apresentar dificuldades em prever com precisão picos de alta volatilidade, como mostrado em pontos de maior variação.""")

    st.write("""2. **Erro ligeiramente superior ao LSTM**:""")
    st.write("""- O RMSE do modelo RNN é um pouco maior, indicando que ele pode ser menos preciso em relação a certas variações rápidas dos preços.""")      

    st.write("""O modelo **RNN Regressor** demonstrou ser eficaz para prever o preço do petróleo Brent, com desempenho competitivo em relação ao **LSTM** e uma precisão que o torna adequado para aplicações de previsão de preços em curto e médio prazo.""")    

  with tab4:
    st.markdown('<h2 style="color:#e61859;">Conclusão Comparativa: Prophet, LSTM e RNN Regressor para Previsão do Preço do Petróleo Brent</h2>', unsafe_allow_html=True)

    st.write("""Com base na análise dos três modelos – **Prophet, LSTM e RNN Regressor** – e considerando suas métricas de desempenho, vantagens e limitações, podemos determinar o modelo mais adequado para prever o preço do petróleo Brent.""")

    st.markdown('<h2 style="color:#e61859;">Modelo Prophet</h2>', unsafe_allow_html=True)    
    st.write("""**Vantagens:**""")
    st.write("""- Excelente para capturar tendências de longo prazo, especialmente em mercados com sazonalidade clara.""")
    st.write("""- Fácil de interpretar, o que facilita a análise de componentes como tendência, sazonalidade e feriados.""")
    st.write("""- A abordagem de validação cruzada demonstrou robustez para projeções estratégicas de longo prazo.""")

    st.write("""**Limitações:**""")
    st.write("""- Menor capacidade de capturar mudanças abruptas ou eventos externos não recorrentes, como crises geopolíticas.""")
    st.write("""- Menor precisão para previsões de curto prazo em cenários de alta volatilidade.""")

    st.write("""**Desempenho:**""")    
    st.write("""- Indicado para projeções de longo prazo e análises estratégicas em ambientes estáveis.""")


    st.markdown('<h2 style="color:#e61859;">Modelo LSTM (Long Short-Term Memory)</h2>', unsafe_allow_html=True)   
    st.write("""**Vantagens:**""")
    st.write("""- Forte capacidade de capturar padrões complexos e não lineares em séries temporais, sendo mais adequado para mercados voláteis.""")
    st.write("""- Excelente desempenho em previsões de curto prazo devido à sua habilidade em lidar com dependências de longo prazo na sequência de dados.""")

    st.write("""**Limitações:**""")
    st.write("""- Requer maior poder computacional e tempo de treinamento em comparação com o Prophet.""")
    st.write("""- Pode ser menos interpretável, o que dificulta a explicação de previsões para decisões estratégicas.""")

    st.write("""**Desempenho:**""")    
    st.write("""- Melhor desempenho geral nas previsões de curto prazo e cenários instáveis.""")


    st.markdown('<h2 style="color:#e61859;">Modelo RNN Regressor</h2>', unsafe_allow_html=True)   
    st.write("""**Vantagens:**""")
    st.write("""- Também capaz de capturar padrões não lineares em séries temporais.""")
    st.write("""- Mais simples e leve que o LSTM, com boa precisão para previsões em ambientes moderadamente voláteis.""")

    st.write("""**Limitações:**""")
    st.write("""- Menor capacidade de capturar dependências de longo prazo em comparação ao LSTM.""")
    st.write("""- Menor precisão em ambientes altamente voláteis ou com eventos externos significativos.""")

    st.write("""**Desempenho:**""")    
    st.write("""- Adequado para previsões de curto prazo em cenários menos voláteis, mas inferior ao LSTM.""")


    st.markdown('<h2 style="color:#e61859;">Conclusão Final</h2>', unsafe_allow_html=True)   

    st.write("""Com base nos resultados e nas características dos modelos, o **LSTM** é o mais adequado para a previsão do preço do petróleo Brent. Isso se deve à sua:""")
    st.write("""1. **Capacidade de capturar padrões não lineares complexos**, comuns no mercado de commodities.""")
    st.write("""2. **Desempenho superior em métricas quantitativas (RMSE, MAPE e MAE)**, indicando maior precisão em comparação aos outros modelos.""")
    st.write("""3. **Eficácia em cenários de curto prazo e alta volatilidade**, o que é essencial para um mercado tão sensível a fatores externos como o petróleo.""")

    st.write("""Embora o **Prophet** seja uma excelente escolha para projeções de longo prazo e decisões estratégicas, sua precisão limitada em ambientes voláteis o torna menos eficaz para previsões 
              operacionais no curto prazo. Já o **RNN Regressor**, apesar de apresentar bons resultados, fica atrás do **LSTM** em termos de precisão e capacidade de capturar padrões complexos.""")

    st.write("""Portanto, para previsões mais precisas e operacionais do preço do petróleo Brent, o **LSTM é a melhor escolha.**""")


#################################################################################################################################################

elif menu == "Dashboard":

  # Título da aplicação
  st.markdown('<h2 style="color:#e61859;">Dashboard</h2>', unsafe_allow_html=True)
  
  # URL do relatório do Power BI (substitua pelo link do seu relatório)
  power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiNDBjMDBlYjUtMzRmOC00YzkyLTg5NGUtYTg5MWNhN2JiMDI0IiwidCI6ImVmYTU1OWEyLTJmOTctNGRkNi1hMmFlLThhYjAyZDliMzMyOSJ9"

  # Incorporando o Power BI com iframe
  st.components.v1.iframe(power_bi_url, width=800, height=600)

  st.write("Para acessar o dashboard em uma nova página, clique abaixo")
  st.markdown(
      """
      <a href="https://app.powerbi.com/view?r=eyJrIjoiNDBjMDBlYjUtMzRmOC00YzkyLTg5NGUtYTg5MWNhN2JiMDI0IiwidCI6ImVmYTU1OWEyLTJmOTctNGRkNi1hMmFlLThhYjAyZDliMzMyOSJ9" target="_blank" style="text-decoration:none; color:#e61859; font-size:18px;">
      DASHBOARD</a>
      """,
      unsafe_allow_html=True)    

elif menu == "Conclusão":

  # Título da aplicação
  st.markdown('<h2 style="color:#e61859;">Conclusão do Projeto de Análise e Previsão do Preço do Petróleo Brent</h2>', unsafe_allow_html=True)

  st.write("""O projeto de análise e previsão do preço do petróleo Brent revelou a complexidade inerente a um mercado globalmente 
            estratégico e altamente volátil. Desde as oscilações históricas impulsionadas por eventos geopolíticos até flutuações sazonais e transições econômicas, 
            nosso estudo trouxe à tona insights valiosos sobre padrões passados, fatores determinantes e possibilidades futuras, respondendo às principais perguntas que nortearam o trabalho.""")

  st.markdown('<h3 style="color:#e61859;">Respostas às Perguntas do Estudo</h3>', unsafe_allow_html=True)

  st.write("""1. **Quais são os padrões históricos do mercado?**""")
  st.write("""- Identificamos tendências sazonais claras, como os preços mais baixos em dezembro e janeiro, associados à baixa demanda e estoques elevados, e aumentos significativos em junho e julho devido à alta demanda no verão do hemisfério norte.""")
  st.write("""- Eventos como a Guerra do Golfo (1990), o pico histórico de 2008 e a queda de 2020, causada pela pandemia de COVID-19, são exemplos das flutuações abruptas que caracterizam esse mercado volátil.""")

  st.write("""2. **O que impulsiona as flutuações de preço?**""")
  st.write("""- **Fatores geopolíticos**: Conflitos no Oriente Médio e decisões da OPEP.""")
  st.write("""- **Fatores econômicos**: Crises financeiras e transições para fontes renováveis desafiam padrões tradicionais.""")
  st.write("""- **Fatores sazonais**: Estações do ano alteram a demanda, enquanto estoques e produção influenciam preços.""")
  st.write("""- **Eventos climáticos**: Furacões e interrupções inesperadas na produção adicionam um elemento imprevisível.""")

  st.write("""3. **Como é possível prever as tendências futuras?**""")
  st.write("""- Utilizando modelos robustos, como o LSTM, que lidam com padrões complexos e não lineares em cenários voláteis de curto prazo.""")
  st.write("""- O Prophet mostrou-se eficaz para análises de longo prazo em cenários estáveis, reforçando a importância de usar modelos complementares.""")

  st.markdown('<h3 style="color:#e61859;">Modelo Selecionado</h3>', unsafe_allow_html=True)
  st.write("""Após comparar Prophet, LSTM e RNN Regressor, o LSTM destacou-se como o mais adequado para previsões operacionais de curto prazo devido à sua 
            capacidade de capturar padrões não lineares e seu desempenho superior em métricas como MAPE, RMSE e MAE. Enquanto o Prophet continua valioso para análises 
            estratégicas de longo prazo, o LSTM atende às exigências de precisão e agilidade de um mercado sensível a mudanças rápidas.""")

  st.markdown('<h3 style="color:#e61859;">Desafios e Relevância</h3>', unsafe_allow_html=True)
  st.write("""O estudo ressaltou desafios como a imprevisibilidade de eventos geopolíticos, a transição para energias renováveis e a influência de fatores climáticos. 
            Esses elementos reforçam a necessidade de modelos preditivos que considerem variáveis exógenas para maior precisão.""")
  st.write("""A análise revelou a resiliência do mercado, evidenciada por padrões de recuperação após choques e pela influência das decisões estratégicas da OPEP. 
            Esses insights são essenciais para investidores, indústrias e formuladores de políticas públicas, ajudando na gestão de riscos e na tomada de decisões estratégicas.""")

  st.markdown('<h3 style="color:#e61859;">Conclusão Final: O Futuro à Luz do Passado</h3>', unsafe_allow_html=True)
  st.write("""Assim como o Brent reflete o pulso da economia global, este projeto ilustra o poder transformador da análise de dados.
            Compreender os padrões históricos e antecipar tendências não é apenas um diferencial competitivo, mas uma necessidade estratégica. 
            Ao integrar ciência de dados, conhecimento do mercado e modelos avançados, criamos não apenas previsões, mas um mapa para navegar as complexidades de um mundo em constante transformação.""")
  st.write("""Este estudo demonstrou que a previsão de preços do petróleo Brent exige uma abordagem que equilibre análises quantitativas robustas e considerações qualitativas sobre fatores externos. 
            O LSTM, com sua capacidade de lidar com padrões complexos, provou ser a melhor escolha para previsões operacionais de curto prazo. No entanto, 
            o Prophet, com validação cruzada, permanece uma ferramenta valiosa para análises estratégicas de longo prazo.""")
  st.write("""Compreender padrões históricos e antecipar tendências futuras não é apenas uma vantagem competitiva, mas uma necessidade em um mercado global tão estratégico. 
            Este estudo reforça a importância de unir ciência de dados, tecnologia e expertise de mercado para enfrentar os desafios e aproveitar as oportunidades de um mundo em constante transformação. 
            O Brent continuará sendo o termômetro da economia mundial, e, agora, estamos mais preparados para interpretar o que ele nos revela.""")
