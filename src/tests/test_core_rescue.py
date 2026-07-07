import numpy as np
from dotmap import DotMap

from src.core import ImageInstanceOps
from src.template import Bubble


def make_ops():
    tuning_config = DotMap({"outputs": {"save_image_level": 0}}, _dynamic=False)
    return ImageInstanceOps(tuning_config)


def make_row(values):
    """4 bubbles A..D com as intensidades dadas, imitando field_block_bubbles."""
    return [
        Bubble((0, 0), "q1", "QTYPE_MCQ4", letter)
        for letter in ["A", "B", "C", "D"]
    ], values


class FakeFieldBlock:
    def __init__(self, n_rows, name="MCQ_Block_Q06_10", shift=0, empty_val=""):
        self.name = name
        self.shift = shift
        self.empty_val = empty_val
        self.traverse_bubbles = [None] * n_rows  # só o tamanho importa aqui


def test_rescue_disabled_by_default_returns_none():
    ops = make_ops()
    bubbles, values = make_row([228.9, 230.5, 230.1, 189.2])  # caso real observado em produção
    threshold_params = {"RESCUE_ABS_DELTA": 0, "RESCUE_MIN_ROW_GAP": 6}
    assert ops.rescue_diluted_mark(bubbles, values, 230.0, threshold_params) is None


def test_rescue_recovers_clearly_diluted_mark():
    ops = make_ops()
    # Valores reais observados em produção: D visivelmente mais escura,
    # A/B/C no nível de branco típico da página.
    bubbles, values = make_row([228.9, 230.5, 230.1, 189.2])
    threshold_params = {"RESCUE_ABS_DELTA": 10, "RESCUE_MIN_ROW_GAP": 6}
    rescued = ops.rescue_diluted_mark(bubbles, values, 230.0, threshold_params)
    assert rescued is not None
    assert rescued.field_value == "D"


def test_rescue_does_not_fire_on_genuinely_blank_row():
    ops = make_ops()
    # Valores reais observados em produção: nenhuma bolha marcada, as 4
    # intensidades ficam dentro do ruído normal de "branco".
    bubbles, values = make_row([233.9, 233.0, 232.8, 231.5])
    threshold_params = {"RESCUE_ABS_DELTA": 10, "RESCUE_MIN_ROW_GAP": 6}
    assert ops.rescue_diluted_mark(bubbles, values, 233.0, threshold_params) is None


def test_rescue_requires_both_conditions_not_just_darkest():
    ops = make_ops()
    # Escura em termos absolutos, mas SEM discriminação interna (as outras 3
    # também estão quase tão escuras - sombra/vinco uniforme, não uma marca).
    bubbles, values = make_row([214.0, 213.0, 212.0, 208.0])
    threshold_params = {"RESCUE_ABS_DELTA": 10, "RESCUE_MIN_ROW_GAP": 6}
    assert ops.rescue_diluted_mark(bubbles, values, 230.0, threshold_params) is None


def test_commit_discards_systematic_column_bias():
    ops = make_ops()
    field_block = FakeFieldBlock(n_rows=5)
    omr_response = {}
    final_marked = None  # não desenhamos nada nos casos de teste abaixo

    bubbles_q16, _ = make_row([225.9, 202.8, 219.0, 211.2])
    bubbles_q17, _ = make_row([226.7, 201.1, 220.3, 210.7])
    bubbles_q18, _ = make_row([229.2, 207.8, 222.5, 212.9])
    # coluna 1 (B) é a "mais escura" em 3 das 5 linhas do bloco -> viés
    candidates = [
        ("q16", bubbles_q16[1], 1, [225.9, 202.8, 219.0, 211.2]),
        ("q17", bubbles_q17[1], 1, [226.7, 201.1, 220.3, 210.7]),
        ("q18", bubbles_q18[1], 1, [229.2, 207.8, 222.5, 212.9]),
    ]

    ops.commit_rescue_candidates(
        candidates, field_block, 28, 28, 230.0, omr_response, final_marked
    )

    assert omr_response == {"q16": "", "q17": "", "q18": ""}


def test_commit_keeps_isolated_candidate_in_same_block():
    ops = make_ops()
    field_block = FakeFieldBlock(n_rows=5)
    omr_response = {}

    bubbles_q7, _ = make_row([228.9, 230.5, 230.1, 189.2])
    bubbles_q8, _ = make_row([233.6, 232.6, 233.6, 191.1])
    # só 2 das 5 linhas do bloco - dentro do esperado para respostas reais
    # repetidas (ex: aluno marca D em duas questões consecutivas).
    candidates = [
        ("q7", bubbles_q7[3], 3, [228.9, 230.5, 230.1, 189.2]),
        ("q8", bubbles_q8[3], 3, [233.6, 232.6, 233.6, 191.1]),
    ]

    final_marked = np.zeros((50, 50, 3), dtype=np.uint8)
    ops.commit_rescue_candidates(
        candidates, field_block, 28, 28, 230.0, omr_response, final_marked
    )

    assert omr_response == {"q7": "D", "q8": "D"}
