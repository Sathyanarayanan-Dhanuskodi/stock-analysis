"""AngelOne brokerage and charges calculation."""


def calc_angel_one_charges(
    buy_price: float, sell_price: float, qty: int = 1, trade_type: str = "intraday"
) -> dict:
    """Calculate AngelOne charges and net profit for a trade.

    trade_type: "delivery" (swing) or "intraday" (scalp)

    Delivery charges:
      - Brokerage: ₹0 (free on AngelOne)
      - STT: 0.1% on both buy and sell
      - Stamp duty: 0.015% on buy side

    Intraday charges:
      - Brokerage: min(₹20, 0.03%) per side, min ₹5
      - STT: 0.025% on sell side only
      - Stamp duty: 0.003% on buy side

    Common charges:
      - Exchange transaction (NSE): 0.00325% of turnover
      - SEBI: ₹10 per crore (0.000001)
      - GST: 18% on (brokerage + exchange + SEBI)
    """
    buy_value = buy_price * qty
    sell_value = sell_price * qty
    turnover = buy_value + sell_value

    if trade_type == "delivery":
        buy_brokerage = 0.0
        sell_brokerage = 0.0
        stt = turnover * 0.001          # 0.1% on both sides
        stamp_duty = buy_value * 0.00015  # 0.015% on buy
    else:  # intraday
        buy_brokerage = max(5.0, min(20.0, buy_value * 0.0003))   # 0.03%
        sell_brokerage = max(5.0, min(20.0, sell_value * 0.0003))
        stt = sell_value * 0.00025      # 0.025% on sell only
        stamp_duty = buy_value * 0.00003  # 0.003% on buy

    total_brokerage = buy_brokerage + sell_brokerage

    # Common charges
    exchange_txn = turnover * 0.0000325   # NSE: 0.00325%
    sebi = turnover * 0.000001            # ₹10 per crore
    gst = (total_brokerage + exchange_txn + sebi) * 0.18

    total_charges = total_brokerage + stt + exchange_txn + sebi + stamp_duty + gst
    gross_profit = (sell_price - buy_price) * qty
    net_profit = gross_profit - total_charges

    return {
        "gross_profit": round(gross_profit, 2),
        "total_charges": round(total_charges, 2),
        "net_profit": round(net_profit, 2),
        "brokerage": round(total_brokerage, 2),
        "stt": round(stt, 2),
        "gst": round(gst, 2),
        "other": round(exchange_txn + sebi + stamp_duty, 2),
    }
