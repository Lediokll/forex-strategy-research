import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import MetaTrader5  # noqa: F401
except ImportError:
    # MetaTrader5 only installs/works on Windows. Provide a minimal stand-in
    # so the rest of the codebase (and its tests) can still be imported and
    # exercised for pure-Python logic elsewhere (e.g. CI on Linux/macOS).
    fake_mt5 = types.ModuleType("MetaTrader5")

    fake_mt5.TIMEFRAME_H1 = 16385

    fake_mt5.ORDER_FILLING_FOK = 0
    fake_mt5.ORDER_FILLING_IOC = 1
    fake_mt5.ORDER_FILLING_RETURN = 2

    fake_mt5.ORDER_TYPE_BUY = 0
    fake_mt5.ORDER_TYPE_SELL = 1

    fake_mt5.POSITION_TYPE_BUY = 0
    fake_mt5.POSITION_TYPE_SELL = 1

    fake_mt5.TRADE_ACTION_DEAL = 1
    fake_mt5.TRADE_ACTION_SLTP = 6
    fake_mt5.TRADE_RETCODE_DONE = 10009
    fake_mt5.ORDER_TIME_GTC = 0

    fake_mt5.ACCOUNT_TRADE_MODE_DEMO = 0
    fake_mt5.ACCOUNT_TRADE_MODE_CONTEST = 1
    fake_mt5.ACCOUNT_TRADE_MODE_REAL = 2

    fake_mt5.initialize = lambda *a, **k: False
    fake_mt5.last_error = lambda: (0, "mocked MetaTrader5: no terminal available")
    fake_mt5.terminal_info = lambda: None
    fake_mt5.account_info = lambda: None
    fake_mt5.symbol_info = lambda symbol: None
    fake_mt5.symbol_info_tick = lambda symbol: None
    fake_mt5.symbol_select = lambda *a, **k: True
    fake_mt5.positions_get = lambda **k: ()
    fake_mt5.order_send = lambda request: None
    fake_mt5.order_calc_margin = lambda *a, **k: 0.0
    fake_mt5.copy_rates_from_pos = lambda *a, **k: None
    fake_mt5.shutdown = lambda: None

    sys.modules["MetaTrader5"] = fake_mt5
