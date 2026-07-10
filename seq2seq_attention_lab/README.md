# Comparador Seq2Seq com GRU vs Transformer encoder-decoder

Este projeto compara duas arquiteturas Sequence-to-Sequence resolvendo o mesmo desafio: inverter sequencias numericas.

Exemplo:

```text
Entrada:  [3, 3, 1, 9, 0, 5, 6]
Esperado: [6, 5, 0, 9, 1, 3, 3]
```

O objetivo e permitir a manipulacao dos hiperparametros de cada modelo, treinar os modelos no mesmo ambiente e comparar:

- acuracia token a token;
- taxa de sequencia exata;
- tokens extras e tokens faltantes;
- loss final;
- tempo de treinamento;
- tempo medio de inferencia;
- quantidade de parametros;
- memoria de pico em CUDA, quando disponivel.

## Arquiteturas comparadas

### 1. Seq2Seq com GRU

A arquitetura recorrente usa um encoder GRU para ler a sequencia de entrada e produzir um hidden state final. O decoder GRU usa esse estado para gerar a saida token por token.

Esse modelo ajuda a visualizar uma limitacao comum em arquiteturas recorrentes: mesmo aprendendo a logica principal da tarefa, o decoder pode ter dificuldade para prever o token de parada `<EOS>` no momento correto.

### 2. Transformer encoder-decoder

O Transformer tambem segue o principio Sequence-to-Sequence, mas usa mecanismos de atencao:

- self-attention no encoder;
- self-attention causal no decoder;
- cross-attention para conectar decoder e encoder.

Com isso, a relacao entre tokens da entrada e da saida passa a ser representada de forma mais direta.

## Estrutura do projeto

```text
seq2seq_attention_lab/
├── app.py
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── data.py
    ├── metrics.py
    ├── training.py
    ├── utils.py
    └── models/
        ├── __init__.py
        ├── gru_seq2seq.py
        └── transformer_seq2seq.py
```

## Instalacao

Crie e ative um ambiente virtual, se desejar:

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

## Executando a interface

Na pasta do projeto, rode:

```bash
python -m streamlit run app.py
```

Se o comando `streamlit` nao for reconhecido no Windows, prefira sempre:

```bash
python -m streamlit run app.py
```

## Como usar

1. Ajuste os parametros gerais na barra lateral.
2. Informe as sequencias de teste, uma por linha.
3. Ajuste os hiperparametros da GRU e do Transformer.
4. Clique em **Treinar e comparar**.
5. Analise as metricas, a tabela de resultados por sequencia e a curva de loss.

## Principais metricas

### Acuracia token a token

Formula:

```text
acuracia = tokens_corretos / total_de_tokens_esperados
```

Essa metrica mede quantos tokens foram previstos corretamente na posicao esperada.

### Sequencia exata

Retorna verdadeiro somente quando a saida completa do modelo e exatamente igual ao esperado.

Essa metrica e importante porque um modelo pode acertar todos os tokens esperados e ainda gerar tokens extras no final.

Exemplo:

```text
Esperado: [5, 0, 4, 1, 1, 3]
Predito:  [5, 0, 4, 1, 1, 3, 3]
```

A acuracia token a token pode ser alta, mas a sequencia exata e falsa.

## Observacao sobre custo de processamento

No Seq2Seq com GRU, o custo cresce de forma mais recorrente e aproximadamente linear com o comprimento da sequencia.

No Transformer, a self-attention tem custo mais sensivel ao comprimento, porque compara tokens entre si. Para sequencias curtas, isso ainda e controlado; para sequencias longas, o custo cresce mais rapidamente.

Por isso, a interface mostra tanto metricas de qualidade quanto metricas de custo.
