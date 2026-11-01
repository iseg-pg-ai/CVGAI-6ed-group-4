# Plano de Projeto: Leitor Inteligente de Documentos Visuais
Pós-Graduação em Inteligência Artificial — Computer Vision & IA Generativa
Ano Letivo 2026/2027 — Proposta 03

---

## 1. Resumo Executivo e Enquadramento

O presente projeto tem por base a **Proposta 03 — Leitor Inteligente de Documentos Visuais**, cujo objetivo central consiste em converter informação contida em imagens de documentos (especificamente talões de compra e faturas simplificadas portuguesas) em dados estruturados e respostas a perguntas de linguagem natural com suporte visual verificável.

O sistema responde diretamente a todos os requisitos do enunciado:
1. **Pré-processamento com OpenCV**: Correção de rotação, recorte da região de interesse, normalização de iluminação em vincos e melhoramento de contraste térmico.
2. **Extração Estruturada de 5 Campos**: Nome da Entidade/Comerciante, NIF do Comerciante, NIF do Cliente (ou Consumidor Final), Data e Total pago.
3. **Perguntas e Respostas Verificáveis**: Capacidade de responder a questões em linguagem natural, distinguindo claramente dados presentes de dados ausentes, acompanhados da região da imagem (bounding box) que sustenta a resposta.
4. **Interface Gráfica Interativa (Gradio)**: Demonstração visual do pipeline passo a passo e exploração dos resultados em tempo real.
5. **Arquitetura Híbrida de Inteligência (Local vs. Cloud)**: Suporte primário a inferência local com `llama.cpp` (privacidade de dados fiscais e independência de rede) e suporte secundário a modelos na cloud (Google Gemini e OpenAI), replicando o código lecionado nas aulas.
6. **Avaliação Quantitativa e Robustez Física**: Medição do rácio de acerto nos 5 campos sobre um conjunto de 15 a 20 documentos reais — incluindo documentos propositadamente amarrotados e desdobrados — com mapeamento detalhado dos erros mais frequentes.

---

## 2. Tipo de Documento Escolhido e Desafios Reais

Delimitação: **Talões de Compra e Faturas Simplificadas do Retalho e Restauração em Portugal**.

Porquê esta escolha e que desafios coloca ao sistema:
- **Caso de uso real**: Automatização de notas de despesa, conferência contabilística e integração com ERPs.
- **Desafios geométricos e óticos**:
  - Fotografias tiradas com smartphone com rotações de 90 graus (horário ou anti-horário).
  - Papel térmico amarrotado e desdobrado: introduz micro-sombras lineares ao longo dos vincos, reflexos especulares na camada térmica e linhas de texto onduladas (em vez de estritamente horizontais).
  - Tinta térmica desvanecida e caracteres partidos pelas dobras do papel.
  - Conflito semântico de valores: coexistência de múltiplos números semelhantes no mesmo papel (subtotais com diferentes taxas de IVA vs. total final a pagar; NIF do emitente vs. NIF do cliente).
- **Vantagem Diferenciadora no Projeto**: A maioria das faturas simplificadas portuguesas inclui um código QR da Autoridade Tributária (AT). O nosso pipeline utilizará a descodificação desse QR Code via OpenCV como mecanismo automático de verificação da exatidão dos dados extraídos pelo OCR e pelos modelos de linguagem.

---

## 3. Arquitetura da Solução (Pipeline em 4 Fases)

### Fase 1: Pré-processamento e Legibilidade (OpenCV)
- **Normalização de Orientação**: Deteção e correção de rotação (0 graus, 90 graus horário/anti-horário, 180 graus). Como as fotos de smartphone vêm frequentemente na horizontal, corrigir a orientação é o passo crítico para que o OCR funcione.
- **Recorte Automático do Talão**: Segmentação por contornos para separar o papel do fundo (ex.: mesa de madeira) e focar a resolução nos caracteres.
- **Remoção de Sombras de Dobras e Vincos**: Normalização de iluminação de fundo (divisão morfológica ou binarização adaptativa Gaussiana) para mitigar o impacto de sombras causadas por papel que foi amarrotado e desdobrado, evitando que o limiar transforme vincos em manchas pretas.
- **Realce Térmico (CLAHE)**: Equalização adaptativa de histograma limitada por contraste (`cv2.createCLAHE`) para recuperar texto com tinta desvanecida nas zonas de vale das dobras.

### Fase 2: Deteção e Extração de Texto com Localização Espacial (OCR)
- **Motor Principal (Padrão Recomendado)**: **EasyOCR** (modelo de deteção profunda CRAFT + reconhecimento BiLSTM, altamente resiliente a linhas de texto onduladas provocadas por papel amarrotado e fontes térmicas degradadas).
- **Motor Secundário (Comparativo)**: **Tesseract** (motor clássico com segmentação em modo `--psm 4/6`, rápido mas mais sensível a dobras e texto curvo).
- **Estratégia Operacional**: Não executamos ambos em simultâneo a cada pedido para não duplicar o tempo de resposta. O motor vencedor do benchmark preliminar (Nível 1 da Secção 4) é adotado como padrão de produção, mantendo-se o outro acessível por um seletor na interface, apenas para fins de demonstração académica.
- Para cada bloco ou linha de texto identificada, o sistema extrai o texto e preserva as coordenadas exatas da sua caixa delimitadora (*bounding box* em píxeis: `[x1, y1, x2, y2]`).
- Este mapeamento espacial é o alicerce para cumprir o requisito de fornecer evidência visual sem depender de coordenadas imprecisas geradas por modelos de linguagem.

### Fase 3: Extração dos 5 Campos Predefinidos
O sistema estrutura a informação nos seguintes atributos normalizados:
- `entidade`: Designação social do comerciante.
- `nif_comerciante`: Número de identificação fiscal do emitente (9 dígitos com validação de checksum módulo 11).
- `nif_cliente`: NIF associado à fatura ou indicação de Consumidor Final.
- `data`: Data e hora da transação no formato ISO (`YYYY-MM-DD`).
- `total`: Valor total a pagar em euros.

### Fase 4: Assistente de Perguntas e Respostas Verificável (Local vs. Cloud)
O módulo de inteligência é implementado sobre uma interface unificada (`llm_client.py`), permitindo alternar de fornecedor com um único clique na aplicação:
- **Fornecedor Local (Padrão de Privacidade)**: `llama.cpp` (`llama-server`) em `http://127.0.0.1:8080/v1` com modelo leve (ex.: `Qwen2.5-3B-Instruct`), 100% offline e em conformidade com a proteção de dados.
- **Fornecedores Cloud (Demonstração da Matéria)**:
  - **Google Gemini** via `google-genai` (ex.: `gemini-2.5-flash`), replicando `class-code/genAI/gemini_image.py`.
  - **OpenAI** via `openai` (ex.: `gpt-4o-mini`), replicando `class-code/genAI/openai_image.py`.
  - As chaves são lidas de um ficheiro `.env` (nunca versionado), à semelhança do código das aulas.
- **Ponto de Arquitetura Relevante**: Como o `llama-server` expõe uma API compatível com a especificação OpenAI (`/v1/chat/completions`), o mesmo cliente Python serve o modelo local e os modelos cloud, alterando apenas `base_url`, `api_key` e o nome do modelo.
- **Funcionamento**: O modelo selecionado recebe o texto indexado por linha (`[01] O DAS BIFANAS`, `[02] NIF: 518396894`, ...). Em talões amarrotados, o modelo deduz semanticamente erros pontuais de OCR (ex.: `TOT4L` -> `total`).
- Ao responder a uma pergunta do utilizador, o modelo devolve:
  - Resposta em linguagem natural.
  - Classificação de presença: `ENCONTRADO` ou `AUSENTE NO DOCUMENTO`.
  - Índice da linha de suporte.
- O OpenCV utiliza esse índice para desenhar um destaque a cores sobre a imagem original e recortar a região comprovativa para apresentação na interface.
- **Failover**: Se a rede ou a quota da API falharem durante a demonstração, a aplicação recua automaticamente para o modelo local, garantindo que a apresentação nunca quebra.

---

## 4. O que Vamos Comparar (Experiências e Benchmark)

Para garantir um relatório técnico estruturado e métricas concretas para a apresentação, dividimos as experiências em três níveis independentes.

### Nível 1: Duelo de Motores OCR (EasyOCR vs. Tesseract)
- **Objetivo**: Comparar diretamente os dois motores sugeridos no enunciado sobre o mesmo conjunto de faturas, antes de introduzir qualquer modelo de linguagem.
- **Critérios**: Taxa de deteção de caracteres/palavras em papel térmico, precisão das caixas delimitadoras, resiliência a texto ondulado em papel amarrotado e tempo médio de inferência por imagem.
- **Resultado Esperado**: O EasyOCR deverá revelar-se superior na leitura de caracteres degradados e no seguimento de linhas deformadas por vincos; o Tesseract deverá ser mais rápido, mas suscetível a fragmentação de linhas sob papel amassado. O vencedor é eleito o motor padrão do projeto, mantendo-se o outro acessível via seletor na interface.

### Nível 2: Duelo de Inteligência (Regex vs. LLM Híbrido vs. VLM Multimodal)
Com o texto e as caixas delimitadoras extraídos pelo OCR vencedor, comparamos três abordagens de extração e resposta:

1. **Abordagem A (Baseline — Regras Heurísticas / Regex)**:
   - Extração dos 5 campos por expressões regulares manuais e validação algorítmica do NIF.
   - Ponto de discussão no relatório: ilustra a fragilidade das regras estáticas quando o formato do recibo varia, quando caracteres sofrem erros de OCR devido a vincos, ou quando coexistem múltiplos totais (subtotal vs. total líquido vs. total a pagar).

2. **Abordagem B (Proposta Principal — Híbrida: OCR Espacial + LLM)**:
   - O OCR fornece coordenadas exatas e texto; o modelo de linguagem compreende o contexto semântico, tolera ruído tipográfico provocado por vincos, extrai os campos e responde em JSON estrito.
   - Ponto de discussão no relatório: combina a exatidão espacial do OCR clássico com a flexibilidade semântica do LLM e a garantia de privacidade quando executado localmente.

3. **Abordagem C (Exploratória — VLM Multimodal Direto)**:
   - Envio direto da imagem tratada para um modelo de visão-linguagem (ex.: `Qwen2.5-VL` ou `Gemma-3`) sem passar por OCR prévio.
   - Ponto de discussão no relatório: avaliação do compromisso entre carga computacional/VRAM e capacidade de leitura de tabelas complexas face à abordagem híbrida.

### Nível 3: Duelo de Paradigmas (LLM Local vs. LLM Cloud)
- **Objetivo**: Comparar o modelo local em `llama.cpp` (`Qwen2.5-3B`) com os modelos comerciais da cloud (`Gemini 2.5 Flash` e `GPT-4o-mini`).
- **Dimensões de Comparação**:
  - **Privacidade e RGPD**: Tratamento integralmente local, sem fuga de dados fiscais sensíveis, vs. transmissão de imagens e texto para servidores externos.
  - **Latência de Resposta**: Tempo de cálculo local em CPU/GPU vs. tempo de rede somado à latência da API remota.
  - **Custo e Dependência**: Custo zero marginal e funcionamento offline vs. custo por token e dependência de conectividade.
  - **Taxa de Acerto e Resiliência a Ruído**: Avaliar se um modelo compacto de 3B parâmetros atinge paridade na extração e na deteção de informação ausente face a modelos de fronteira.

---

## 5. Estrutura da Interface Gráfica (Gradio)

A aplicação será organizada em três separadores claros para facilitar quer a avaliação quer a apresentação ao vivo.

### Separador 1: Leitor e Assistente de Documentos (Demonstração Principal)
- **Painel Esquerdo (Entrada e Controlos de Imagem)**:
  - Carregamento de imagem (Upload) ou seleção rápida a partir de galeria de exemplos (incluindo talões planos e talões amarrotados).
  - Controlos interativos de OpenCV: seletor de rotação (0 graus, 90 graus, 180 graus, 270 graus), botão de recorte automático, slider de intensidade de contraste CLAHE e alternador de atenuação de vincos.
  - Seletor do motor OCR (`EasyOCR [Padrão]` vs. `Tesseract`).
  - Seletor do fornecedor de inteligência:
    - `Regras Python (Regex Baseline)`
    - `LLM Local via llama.cpp (Qwen2.5 3B) [Padrão Privado]`
    - `LLM Cloud - Google Gemini (gemini-2.5-flash)`
    - `LLM Cloud - OpenAI (gpt-4o-mini)`
  - Botão de execução do processamento.
- **Painel Central (Evidência Visual)**:
  - Imagem do documento anotada com retângulos delimitadores coloridos sobre o texto identificado.
  - Caixa de detalhe visual com o recorte ampliado da zona do documento que comprova a resposta ativa.
- **Painel Direito (Dados Estruturados e Assistente)**:
  - Tabela com os 5 campos extraídos, respetivo estado de validação e valor correspondente no QR Code da AT (quando disponível).
  - Caixa de conversação para perguntas em linguagem natural (com botões de atalho: *"Qual o NIF do cliente?"*, *"Qual o método de pagamento?"*, *"Qual o email da loja?"* para demonstrar dados ausentes).
  - Indicador categórico: `ENCONTRADO` vs. `AUSENTE NO DOCUMENTO`.

### Separador 2: Pipeline Passo a Passo (Didático / Slides)
- Apresentação lado a lado das quatro etapas de visão computacional:
  1. Imagem Bruta
  2. Imagem Alinhada e Recortada
  3. Imagem Filtrada e Otimizada (CLAHE e atenuação de vincos)
  4. Mapa de Caixas Delimitadoras do OCR
- Permite capturar ecrãs diretos para o relatório e explicar claramente ao professor o contributo do OpenCV em documentos desafiadores.

### Separador 3: Validação e Benchmark (15 a 20 Documentos)
- Apresentação de tabela e gráficos de precisão:
  - Percentagem de sucesso na extração por campo (`Entidade`, `NIF Comerciante`, `NIF Cliente`, `Data`, `Total`).
  - Desagregação de desempenho: **Grupo A (Talões Planos)** vs. **Grupo B (Talões Amarrotados/Desdobrados)**.
  - Comparação de tempo médio de execução e de precisão entre **Local (`llama.cpp`)** e **Cloud (`Gemini` / `OpenAI`)**.
  - Matriz de erros frequentes observados no dataset de teste (ex.: corte de caracteres em vincos, confusão entre NIF emitente/cliente pelo Regex, recuperação bem-sucedida pelo LLM).

---

## 6. Divisão de Trabalho da Equipa

### Responsabilidades da Parte Técnica (Desenvolvimento do Código)
- Implementação dos módulos de pré-processamento OpenCV (`preprocessing.py`) com filtros de contraste e atenuação de sombras de vincos.
- Integração dos motores de OCR e do cliente unificado de linguagem (`llm_client.py`), suportando `llama.cpp` local e APIs cloud (Gemini e OpenAI) com chaves em ficheiro `.env`.
- Construção da interface gráfica em Gradio (`app.py`).
- Criação do script de avaliação automatizada sobre a pasta de documentos (`evaluate.py`).

### Responsabilidades dos Colegas (Dados, Validação, Relatório e Apresentação)
1. **Recolha e Catalogação do Dataset (Dias 1 a 3)**:
   - Obter 7 a 11 fotos adicionais de talões/faturas de lojas variadas, combinando com as 9 já recolhidas para totalizar 16 a 20 imagens.
   - **Composição recomendada**:
     - ~10 a 12 talões normais/planos.
     - ~5 a 8 talões propositadamente amarrotados e desdobrados (para testar os limites do sistema e enriquecer a secção de análise de erros).
     - Pelo menos 2 a 3 talões com dados ausentes (ex.: sem NIF de cliente ou sem contacto).
   - Preencher o ficheiro de validação (`ground_truth.csv`) com os valores reais corretos de cada talão nas colunas: `ficheiro`, `grupo` (plano/amarrotado), `entidade`, `nif_comerciante`, `nif_cliente`, `data`, `total`.

2. **Estruturação do Relatório Escrito (Dias 4 a 10)**:
   - Secção 1: Introdução, problema de negócio e contextualização legal do documento fiscal português.
   - Secção 2: Pipeline de Visão Computacional (OpenCV): justificação das técnicas de rotação, binarização adaptativa e CLAHE (com base nas imagens do Separador 2 da app).
   - Secção 3: Avaliação Comparativa de Motores OCR (EasyOCR vs. Tesseract).
   - Secção 4: Comparação de Inteligência e Paradigmas (Regex vs. LLM Local vs. LLM Cloud): discussão de privacidade (RGPD), custos de infraestrutura e latência.
   - Secção 5: Resultados Experimentais e Análise de Erros: comparação direta do desempenho em talões planos vs. amarrotados.
   - Secção 6: Conclusão e desenvolvimentos futuros.

3. **Construção do PowerPoint de Apresentação (Dias 10 a 13)**:
   - Alinhamento de 10 a 12 slides com narrativa limpa e visual.
   - Roteiro da apresentação: Enquadramento (1 min) -> Arquitetura CV e LLMs Local vs. Cloud (2 min) -> Demonstração ao vivo no Gradio (3 min) -> Resultados experimentais e análise do caso dos talões amarrotados (2 min) -> Conclusão (1 min).

---

## 7. Estratégia de Hardware e Execução

- **Ambiente de Desenvolvimento Principal**: O projeto corre integralmente em computador portátil com CPU padrão e placa gráfica ligeira, utilizando o pipeline híbrido com modelos de 3B parâmetros (ex.: `Qwen2.5-3B-Instruct`), com tempos de resposta de 3 a 5 segundos por documento.
- **Flexibilidade com a Cloud**: A inclusão opcional de Google Gemini e OpenAI permite testar e demonstrar modelos de fronteira sem sobrecarregar o hardware local, bastando configurar as chaves no ficheiro `.env`.
- **Ambiente de Aceleração Opcional**: Modelos de visão mais pesados (7B a 14B) podem ser executados numa máquina com maior capacidade gráfica, servindo a API pela rede local sem alterar o código da interface.
- **Segurança da Apresentação**: Toda a demonstração possui contingência 100% offline via `llama.cpp`, garantindo que eventuais quebras de rede ou quotas de API durante a aula não afetam a apresentação.

---

## 8. Cronograma e Próximos Marcos

| Fase | Duração Estimada | Entregável Principal |
| :--- | :--- | :--- |
| **Marco 1: Reunião e Alinhamento** | Dia 1 | Aprovação da arquitetura e início da recolha de dados |
| **Marco 2: Dataset e Ground Truth** | Dias 2 a 4 | Pasta `data/receipts/` com 16 a 20 fotos (planas e amarrotadas) e `ground_truth.csv` |
| **Marco 3: Código Funcional e Gradio** | Dias 5 a 8 | Interface operacional com pré-processamento, OCR, extração e Q&A |
| **Marco 4: Avaliação e Benchmark** | Dias 9 a 10 | Métricas, gráficos e tabelas de análise de erros planos vs. amarrotados e Local vs. Cloud |
| **Marco 5: Relatório e Slides** | Dias 11 a 13 | Documento final e apresentação PowerPoint finalizados |
| **Marco 6: Ensaio de Demonstração** | Dia 14 | Teste geral da apresentação e submissão |
