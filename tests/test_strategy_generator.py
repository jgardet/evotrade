from src.strategies.generator import StrategyGenerator


def test_bootstrap_explore_strategy_is_valid_python_and_named():
    generator = StrategyGenerator()

    code = generator._build_bootstrap_explore_strategy(
        "Agent_EXPLORE_Test",
        {"pair": "BTC/USDT", "timeframe": "1h"},
    )

    assert "class Agent_EXPLORE_Test(IStrategy):" in code
    assert 'timeframe = "1h"' in code
    assert "populate_indicators" in code
    assert "populate_entry_trend" in code
    assert "populate_exit_trend" in code


def test_finalize_strategy_code_renames_class_to_strategy_name():
    generator = StrategyGenerator()
    raw = """
from freqtrade.strategy.interface import IStrategy

class WrongName(IStrategy):
    timeframe = "4h"
    minimal_roi = {"0": 0.1}
    stoploss = -0.05

    def populate_indicators(self, dataframe, metadata):
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        return dataframe
"""

    finalized = generator._finalize_strategy_code(raw, "CorrectName")

    assert "class CorrectName(IStrategy):" in finalized
    assert "class WrongName(IStrategy):" not in finalized
