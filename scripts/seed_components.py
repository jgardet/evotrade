"""
Bootstrap script — seeds the component library with a curated set of
human-authored components to give the agent a solid starting point.

Run once:
  python -m scripts.seed_components
"""
import asyncio
import uuid
from src.db.database import db
from src.models import Component, ComponentCategory, ComponentOrigin


SEED_COMPONENTS = [
    Component(
        id=uuid.uuid4(),
        category=ComponentCategory.INDICATOR,
        name="rsi_14",
        code_snippet="""
def populate_rsi_14(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
    return dataframe
""",
        parameters={"rsi_period": 14},
        parameter_space={"rsi_period": [5, 50]},
        dependencies=["talib"],
        description="RSI with configurable period (default 14)",
        origin=ComponentOrigin.HUMAN_AUTHORED,
        tags=["momentum", "oscillator"],
    ),
    Component(
        id=uuid.uuid4(),
        category=ComponentCategory.INDICATOR,
        name="bollinger_bands_20",
        code_snippet="""
def populate_bb(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
    dataframe['bb_upper'] = bollinger['upper']
    dataframe['bb_lower'] = bollinger['lower']
    dataframe['bb_mid'] = bollinger['mid']
    dataframe['bb_width'] = (bollinger['upper'] - bollinger['lower']) / bollinger['mid']
    return dataframe
""",
        parameters={"bb_period": 20, "bb_std": 2},
        parameter_space={"bb_period": [10, 50], "bb_std": [1.5, 3.0]},
        dependencies=["qtpylib"],
        description="Bollinger Bands with width indicator",
        origin=ComponentOrigin.HUMAN_AUTHORED,
        tags=["volatility", "bands"],
    ),
    Component(
        id=uuid.uuid4(),
        category=ComponentCategory.INDICATOR,
        name="macd_12_26_9",
        code_snippet="""
def populate_macd(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    macd = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']
    return dataframe
""",
        parameters={"macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
        parameter_space={"macd_fast": [5, 20], "macd_slow": [20, 50], "macd_signal": [5, 15]},
        dependencies=["talib"],
        description="MACD with histogram",
        origin=ComponentOrigin.HUMAN_AUTHORED,
        tags=["trend", "momentum"],
    ),
    Component(
        id=uuid.uuid4(),
        category=ComponentCategory.INDICATOR,
        name="atr_14",
        code_snippet="""
def populate_atr(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
    dataframe['atr_pct'] = dataframe['atr'] / dataframe['close']  # normalized
    return dataframe
""",
        parameters={"atr_period": 14},
        parameter_space={"atr_period": [7, 28]},
        dependencies=["talib"],
        description="Average True Range (absolute and % normalized)",
        origin=ComponentOrigin.HUMAN_AUTHORED,
        tags=["volatility", "risk"],
    ),
    Component(
        id=uuid.uuid4(),
        category=ComponentCategory.INDICATOR,
        name="adx_14",
        code_snippet="""
def populate_adx(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe['adx'] = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
    dataframe['plus_di'] = ta.PLUS_DI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
    dataframe['minus_di'] = ta.MINUS_DI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
    return dataframe
""",
        parameters={"adx_period": 14},
        parameter_space={"adx_period": [7, 28]},
        dependencies=["talib"],
        description="ADX with +DI and -DI for trend strength and direction",
        origin=ComponentOrigin.HUMAN_AUTHORED,
        tags=["trend", "strength"],
    ),
    Component(
        id=uuid.uuid4(),
        category=ComponentCategory.ENTRY_SIGNAL,
        name="rsi_oversold_entry",
        code_snippet="""
def entry_rsi_oversold(self, dataframe: DataFrame) -> pd.Series:
    return (
        (dataframe['rsi'] < self.rsi_entry_threshold.value) &
        (dataframe['rsi'].shift(1) < self.rsi_entry_threshold.value) &
        (dataframe['volume'] > 0)
    )
""",
        parameters={"rsi_entry_threshold": 30},
        parameter_space={"rsi_entry_threshold": [20, 40]},
        dependencies=[],
        description="Enter when RSI crosses below oversold threshold",
        origin=ComponentOrigin.HUMAN_AUTHORED,
        tags=["entry", "rsi", "mean_reversion"],
    ),
    Component(
        id=uuid.uuid4(),
        category=ComponentCategory.ENTRY_SIGNAL,
        name="bb_lower_bounce_entry",
        code_snippet="""
def entry_bb_lower_bounce(self, dataframe: DataFrame) -> pd.Series:
    return (
        (dataframe['close'] < dataframe['bb_lower']) &
        (dataframe['close'].shift(1) < dataframe['bb_lower'].shift(1)) &
        (dataframe['close'] > dataframe['close'].shift(1)) &  # price turning up
        (dataframe['volume'] > dataframe['volume'].rolling(20).mean())
    )
""",
        parameters={},
        parameter_space={},
        dependencies=["bollinger_bands_20"],
        description="Enter when price bounces up from below Bollinger lower band with volume",
        origin=ComponentOrigin.HUMAN_AUTHORED,
        tags=["entry", "bollinger", "mean_reversion"],
    ),
    Component(
        id=uuid.uuid4(),
        category=ComponentCategory.EXIT_SIGNAL,
        name="rsi_overbought_exit",
        code_snippet="""
def exit_rsi_overbought(self, dataframe: DataFrame) -> pd.Series:
    return (
        (dataframe['rsi'] > self.rsi_exit_threshold.value) &
        (dataframe['volume'] > 0)
    )
""",
        parameters={"rsi_exit_threshold": 70},
        parameter_space={"rsi_exit_threshold": [60, 80]},
        dependencies=[],
        description="Exit when RSI enters overbought territory",
        origin=ComponentOrigin.HUMAN_AUTHORED,
        tags=["exit", "rsi"],
    ),
    Component(
        id=uuid.uuid4(),
        category=ComponentCategory.FILTER,
        name="adx_trend_filter",
        code_snippet="""
def filter_adx_trend(self, dataframe: DataFrame) -> pd.Series:
    \"\"\"Only trade when ADX indicates sufficient trend strength.\"\"\"
    return dataframe['adx'] > self.adx_threshold.value
""",
        parameters={"adx_threshold": 25},
        parameter_space={"adx_threshold": [15, 40]},
        dependencies=["adx_14"],
        description="Filter: only allow entries when ADX > threshold (trend filter)",
        origin=ComponentOrigin.HUMAN_AUTHORED,
        tags=["filter", "trend", "adx"],
    ),
    Component(
        id=uuid.uuid4(),
        category=ComponentCategory.RISK_RULE,
        name="atr_stoploss",
        code_snippet="""
# ATR-based dynamic stoploss
# Set stoploss = -2 * ATR% at entry time
# Implemented via custom_stoploss callback
def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    last_candle = dataframe.iloc[-1]
    atr_pct = last_candle.get('atr_pct', 0.02)
    return -2 * atr_pct  # dynamic stoploss = 2x ATR
""",
        parameters={"atr_multiplier": 2.0},
        parameter_space={"atr_multiplier": [1.0, 4.0]},
        dependencies=["atr_14"],
        description="Dynamic stoploss based on ATR percentage",
        origin=ComponentOrigin.HUMAN_AUTHORED,
        tags=["risk", "stoploss", "atr"],
    ),
    Component(
        id=uuid.uuid4(),
        category=ComponentCategory.REGIME_FILTER,
        name="high_volatility_regime_filter",
        code_snippet="""
def filter_high_volatility(self, dataframe: DataFrame) -> pd.Series:
    \"\"\"Only trade in high-volatility regimes (BB width above threshold).\"\"\"
    bb_width_ma = dataframe['bb_width'].rolling(20).mean()
    return dataframe['bb_width'] > bb_width_ma * self.vol_regime_multiplier.value
""",
        parameters={"vol_regime_multiplier": 1.2},
        parameter_space={"vol_regime_multiplier": [1.0, 2.0]},
        dependencies=["bollinger_bands_20"],
        description="Regime filter: only active in high-volatility conditions",
        origin=ComponentOrigin.HUMAN_AUTHORED,
        tags=["regime", "volatility", "filter"],
    ),
]


async def main():
    await db.connect()
    print(f"Seeding {len(SEED_COMPONENTS)} components...")
    for comp in SEED_COMPONENTS:
        await db.save_component(comp)
        print(f"  ✓ {comp.category.value}: {comp.name}")
    await db.disconnect()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
