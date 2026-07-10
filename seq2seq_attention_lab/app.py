from __future__ import annotations

import pandas as pd
import streamlit as st
import torch

from src.data import parse_test_sequences
from src.training import (
    GRUTrainConfig,
    TransformerTrainConfig,
    evaluate_gru_model,
    evaluate_transformer_model,
    train_gru,
    train_transformer,
)
from src.utils import get_device


DEFAULT_TESTS = """3,3,1,9,0,5,6
3,1,1,4,0,5
3,1,1,4,0
3,1,1,4
1,2,3
9,5,1,7"""


st.set_page_config(
    page_title="Comparador Seq2Seq vs Transformer",
    page_icon="🧠",
    layout="wide",
)


st.title("Comparador: Seq2Seq com GRU vs Transformer encoder-decoder")
st.caption(
    "Interface para manipular hiperparametros, treinar os dois modelos no mesmo desafio "
    "e comparar resultado preditivo, custo de processamento e estabilidade da geracao."
)


with st.sidebar:
    st.header("Configuracao geral")

    device_option = st.selectbox(
        "Device",
        options=["Auto", "CPU", "CUDA"],
        index=0,
        help="Use Auto para CUDA quando disponivel; caso contrario, CPU.",
    )

    if device_option == "CUDA" and not torch.cuda.is_available():
        st.warning("CUDA nao esta disponivel. O app usara CPU.")
        device = torch.device("cpu")
    elif device_option == "CPU":
        device = torch.device("cpu")
    else:
        device = get_device(prefer_cuda=True)

    st.write(f"Device atual: `{device}`")

    seed = st.number_input("Seed", min_value=0, max_value=999_999, value=42, step=1)
    min_len = st.number_input("MIN_LEN", min_value=1, max_value=100, value=3, step=1)
    max_len = st.number_input("MAX_LEN", min_value=1, max_value=100, value=7, step=1)
    batches_per_epoch = st.number_input("BATCHES_PER_EPOCH", min_value=1, max_value=10_000, value=100, step=10)
    learning_rate = st.number_input("Learning rate", min_value=0.00001, max_value=1.0, value=0.001, step=0.0001, format="%.5f")
    eos_weight = st.slider("Peso do token EOS na loss", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

    st.divider()
    st.header("Sequencias de teste")
    tests_text = st.text_area(
        "Uma sequencia por linha",
        value=DEFAULT_TESTS,
        height=180,
        help="Use digitos separados por virgula. Exemplo: 3,1,1,4,0,5",
    )


tab_compare, tab_explain, tab_params, tab_structure = st.tabs(
    ["Comparacao", "Como os modelos funcionam", "Guia de parametros", "Estrutura do projeto"]
)


with tab_compare:
    st.subheader("1. Hiperparametros dos modelos")

    col_gru, col_transformer = st.columns(2)

    with col_gru:
        st.markdown("### Seq2Seq com GRU")
        gru_embed_size = st.number_input("GRU - EMBED_SIZE", min_value=2, max_value=1024, value=32, step=2)
        gru_hidden_size = st.number_input("GRU - HIDDEN_SIZE", min_value=4, max_value=2048, value=64, step=4)
        gru_batch_size = st.number_input("GRU - BATCH_SIZE", min_value=1, max_value=2048, value=128, step=16)
        gru_epochs = st.number_input("GRU - EPOCHS", min_value=1, max_value=1000, value=17, step=1)
        teacher_forcing_ratio = st.slider(
            "GRU - TEACHER_FORCING_RATIO",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
        )

    with col_transformer:
        st.markdown("### Transformer encoder-decoder")
        d_model = st.number_input("Transformer - D_MODEL", min_value=2, max_value=1024, value=16, step=2)
        nhead = st.number_input("Transformer - NHEAD", min_value=1, max_value=64, value=2, step=1)
        dim_feedforward = st.number_input("Transformer - DIM_FEEDFORWARD", min_value=4, max_value=4096, value=64, step=4)
        num_encoder_layers = st.number_input("Transformer - NUM_ENCODER_LAYERS", min_value=1, max_value=24, value=2, step=1)
        num_decoder_layers = st.number_input("Transformer - NUM_DECODER_LAYERS", min_value=1, max_value=24, value=2, step=1)
        transformer_batch_size = st.number_input("Transformer - BATCH_SIZE", min_value=1, max_value=2048, value=128, step=16)
        transformer_epochs = st.number_input("Transformer - EPOCHS", min_value=1, max_value=1000, value=10, step=1)
        dropout = st.slider("Transformer - DROPOUT", min_value=0.0, max_value=0.9, value=0.1, step=0.05)
        max_seq_len = st.number_input("Transformer - MAX_SEQ_LEN", min_value=4, max_value=4096, value=64, step=4)

    st.divider()

    run_button = st.button("Treinar e comparar", type="primary", use_container_width=True)

    if max_len < min_len:
        st.error("MAX_LEN precisa ser maior ou igual a MIN_LEN.")
        st.stop()

    if int(d_model) % int(nhead) != 0:
        st.error("No Transformer, D_MODEL precisa ser divisivel por NHEAD.")
        st.stop()

    try:
        tests = parse_test_sequences(tests_text)
    except Exception as exc:
        st.error(f"Erro ao ler sequencias de teste: {exc}")
        st.stop()

    if run_button:
        st.subheader("2. Treinamento")

        gru_config = GRUTrainConfig(
            min_len=int(min_len),
            max_len=int(max_len),
            batch_size=int(gru_batch_size),
            epochs=int(gru_epochs),
            batches_per_epoch=int(batches_per_epoch),
            learning_rate=float(learning_rate),
            eos_weight=float(eos_weight),
            seed=int(seed),
            embed_size=int(gru_embed_size),
            hidden_size=int(gru_hidden_size),
            teacher_forcing_ratio=float(teacher_forcing_ratio),
        )

        transformer_config = TransformerTrainConfig(
            min_len=int(min_len),
            max_len=int(max_len),
            batch_size=int(transformer_batch_size),
            epochs=int(transformer_epochs),
            batches_per_epoch=int(batches_per_epoch),
            learning_rate=float(learning_rate),
            eos_weight=float(eos_weight),
            seed=int(seed),
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            num_encoder_layers=int(num_encoder_layers),
            num_decoder_layers=int(num_decoder_layers),
            dropout=float(dropout),
            max_seq_len=int(max_seq_len),
        )

        left, right = st.columns(2)

        with left:
            st.markdown("#### Treinando Seq2Seq + GRU")
            gru_progress = st.progress(0)
            gru_status = st.empty()

            def update_gru(epoch: int, epochs: int, loss: float) -> None:
                gru_progress.progress(epoch / epochs)
                gru_status.write(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

            with st.spinner("Treinando GRU..."):
                gru_result = train_gru(gru_config, device, progress_callback=update_gru)

        with right:
            st.markdown("#### Treinando Transformer")
            transformer_progress = st.progress(0)
            transformer_status = st.empty()

            def update_transformer(epoch: int, epochs: int, loss: float) -> None:
                transformer_progress.progress(epoch / epochs)
                transformer_status.write(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

            with st.spinner("Treinando Transformer..."):
                transformer_result = train_transformer(transformer_config, device, progress_callback=update_transformer)

        st.subheader("3. Avaliacao nos testes")

        gru_eval = evaluate_gru_model(gru_result["model"], tests, device)
        transformer_eval = evaluate_transformer_model(transformer_result["model"], tests, device)

        summary_rows = []
        for result, evaluation in [(gru_result, gru_eval), (transformer_result, transformer_eval)]:
            summary = evaluation["summary"]
            summary_rows.append(
                {
                    "Modelo": result["name"],
                    "Acuracia media": f"{summary['avg_token_accuracy'] * 100:.2f}%",
                    "Sequencia exata": f"{summary['exact_match_rate'] * 100:.2f}%",
                    "Loss final": f"{result['final_loss']:.4f}",
                    "Tempo treino (s)": f"{result['train_seconds']:.2f}",
                    "Inferencia media (ms)": f"{summary['avg_inference_ms']:.3f}",
                    "Parametros": f"{result['parameters']:,}".replace(",", "."),
                    "Memoria pico CUDA (MB)": "-" if result["peak_memory_mb"] is None else f"{result['peak_memory_mb']:.2f}",
                    "Tokens extras medio": f"{summary['avg_extra_tokens']:.2f}",
                    "Tokens faltantes medio": f"{summary['avg_missing_tokens']:.2f}",
                }
            )

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.subheader("4. Resultados por sequencia")
        detail_rows = []

        for model_name, evaluation in [
            ("Seq2Seq + GRU", gru_eval),
            ("Transformer encoder-decoder", transformer_eval),
        ]:
            for test, metrics in zip(tests, evaluation["sequence_metrics"]):
                detail_rows.append(
                    {
                        "Modelo": model_name,
                        "Entrada": str(list(test)),
                        "Esperado": str(metrics.expected),
                        "Predito": str(metrics.predicted),
                        "Acuracia": f"{metrics.token_accuracy * 100:.2f}%",
                        "Exata": metrics.exact_match,
                        "Acertos": f"{metrics.correct_tokens}/{metrics.total_expected_tokens}",
                        "Extras": metrics.extra_tokens,
                        "Faltantes": metrics.missing_tokens,
                    }
                )

        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

        st.subheader("5. Curva de loss")
        max_epochs_chart = max(len(gru_result["loss_history"]), len(transformer_result["loss_history"]))
        loss_df = pd.DataFrame({"epoch": list(range(1, max_epochs_chart + 1))})
        loss_df["Seq2Seq + GRU"] = pd.Series(gru_result["loss_history"])
        loss_df["Transformer encoder-decoder"] = pd.Series(transformer_result["loss_history"])
        st.line_chart(loss_df.set_index("epoch"))


with tab_explain:
    st.subheader("Como os modelos funcionam")

    st.markdown(
        """
        ### Objetivo do experimento

        O desafio e inverter sequencias numericas. Por exemplo:

        `entrada = [3, 3, 1, 9, 0, 5, 6]`  
        `saida esperada = [6, 5, 0, 9, 1, 3, 3]`

        O problema e simples, mas permite observar pontos importantes de geracao sequencial:
        representacao dos tokens, dependencia de contexto, previsao autoregressiva e controle de parada.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Seq2Seq com GRU

            A arquitetura recorrente usa um **encoder GRU** para ler a entrada token por token.
            Ao final, o encoder produz um **hidden state**, que funciona como uma representacao compactada
            da sequencia de entrada.

            O **decoder GRU** recebe esse estado e gera a resposta um token por vez. Ele comeca com `<SOS>`
            e deve aprender a gerar os digitos invertidos ate prever `<EOS>`.

            Principal caracteristica:
            - processa a sequencia de forma recorrente;
            - depende fortemente do estado final do encoder;
            - pode aprender bem a tarefa central, mas ter dificuldade para controlar exatamente o fim da geracao.
            """
        )

    with col2:
        st.markdown(
            """
            ### Transformer encoder-decoder

            O Transformer tambem e uma arquitetura Sequence-to-Sequence, mas substitui a recorrencia por mecanismos
            de atencao.

            O encoder usa **self-attention** para representar a entrada considerando relacoes entre tokens.
            O decoder usa **self-attention causal** para olhar apenas para os tokens ja gerados e usa
            **cross-attention** para consultar a representacao produzida pelo encoder.

            Principal caracteristica:
            - relaciona tokens de forma mais direta;
            - paraleliza melhor o processamento durante o treino;
            - tende a controlar melhor a relacao entre entrada, saida e token de parada.
            """
        )

    st.markdown(
        """
        ### Tokens especiais

        - `<PAD>`: usado apenas para completar sequencias menores dentro do batch.
        - `<SOS>`: marca o inicio da geracao no decoder.
        - `<EOS>`: marca o fim da sequencia e deve ser aprendido pelo modelo.

        No experimento, uma diferenca importante a observar e se o modelo consegue prever `<EOS>` no momento correto.
        Quando isso falha, a saida pode continuar gerando tokens extras mesmo depois de ja ter produzido os valores esperados.
        """
    )


with tab_params:
    st.subheader("O que cada parametro representa")

    common_params = pd.DataFrame(
        [
            ["MIN_LEN", "Tamanho minimo das sequencias geradas no treino.", "Baixo a medio"],
            ["MAX_LEN", "Tamanho maximo das sequencias geradas no treino.", "Alto, principalmente no Transformer"],
            ["BATCH_SIZE", "Quantidade de exemplos processados por batch.", "Alto"],
            ["EPOCHS", "Quantidade de rodadas de treino.", "Alto no custo total"],
            ["BATCHES_PER_EPOCH", "Quantidade de batches em cada epoca.", "Alto no custo total"],
            ["Learning rate", "Tamanho do passo do otimizador ao atualizar os pesos.", "Baixo no custo, alto na estabilidade"],
            ["EOS weight", "Peso maior para erros no token de parada EOS.", "Baixo no custo, relevante para qualidade"],
            ["Seed", "Controla a aleatoriedade para tornar comparacoes mais reproduziveis.", "Sem impacto relevante"],
        ],
        columns=["Parametro", "Significado", "Impacto de custo"],
    )

    gru_params = pd.DataFrame(
        [
            ["EMBED_SIZE", "Tamanho do vetor de representacao de cada token.", "Medio"],
            ["HIDDEN_SIZE", "Tamanho da memoria interna da GRU.", "Alto"],
            ["TEACHER_FORCING_RATIO", "Probabilidade de o decoder receber o token correto anterior durante o treino.", "Muito baixo"],
        ],
        columns=["Parametro", "Significado", "Impacto de custo"],
    )

    transformer_params = pd.DataFrame(
        [
            ["D_MODEL", "Tamanho dos vetores internos do Transformer.", "Alto"],
            ["NHEAD", "Quantidade de cabecas de atencao.", "Medio/baixo se D_MODEL for fixo"],
            ["DIM_FEEDFORWARD", "Tamanho da MLP interna dos blocos Transformer.", "Medio a alto"],
            ["NUM_ENCODER_LAYERS", "Numero de camadas no encoder.", "Alto"],
            ["NUM_DECODER_LAYERS", "Numero de camadas no decoder.", "Alto"],
            ["DROPOUT", "Regularizacao que desliga parte das ativacoes durante o treino.", "Baixo"],
            ["MAX_SEQ_LEN", "Limite de posicoes suportadas pelo embedding posicional.", "Baixo se as sequencias reais forem curtas"],
        ],
        columns=["Parametro", "Significado", "Impacto de custo"],
    )

    st.markdown("### Parametros gerais")
    st.dataframe(common_params, use_container_width=True)

    st.markdown("### Parametros do Seq2Seq com GRU")
    st.dataframe(gru_params, use_container_width=True)

    st.markdown("### Parametros do Transformer")
    st.dataframe(transformer_params, use_container_width=True)

    st.info(
        "Regra pratica: no GRU, HIDDEN_SIZE costuma ser o parametro estrutural mais caro. "
        "No Transformer, MAX_LEN, numero de camadas, D_MODEL e DIM_FEEDFORWARD tendem a pesar mais."
    )


with tab_structure:
    st.subheader("Estrutura profissional do projeto")

    st.code(
        """
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
        """.strip(),
        language="text",
    )

    st.markdown(
        """
        - `app.py`: interface Streamlit.
        - `src/data.py`: geracao de dados sinteticos, tokenizacao, padding e decodificacao.
        - `src/models/gru_seq2seq.py`: modelo Seq2Seq com GRU.
        - `src/models/transformer_seq2seq.py`: modelo Transformer encoder-decoder.
        - `src/training.py`: treino, avaliacao e medicao de performance.
        - `src/metrics.py`: metricas de acuracia token a token e sequencia exata.
        - `src/utils.py`: seed, device, contagem de parametros e memoria.
        """
    )
