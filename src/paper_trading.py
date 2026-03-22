"""Paper trading tracker for swing and scalping signals.

Trade lifecycle:
  PENDING  → entry price not yet hit (order placed, waiting for fill)
  ACTIVE   → entry price was hit, trade is live, watching targets & stop loss
  HIT_T1   → target 1 hit (trade closed with profit)
  HIT_T2   → target 2 hit
  HIT_T3   → target 3 hit
  STOPPED_OUT → stop loss hit after entry was filled
  EXPIRED  → entry never filled or trade expired while active
  CLOSED   → manually closed
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.charges import calc_angel_one_charges

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "paper_trades.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_paper_db():
    """Create paper_trades table if it doesn't exist."""
    conn = _get_conn()
    # Drop old table if schema changed (no user data we need to keep)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            trade_type      TEXT NOT NULL,
            direction       TEXT NOT NULL,
            entry_price     REAL NOT NULL,
            stop_loss       REAL NOT NULL,
            target_1        REAL NOT NULL,
            target_2        REAL,
            target_3        REAL,
            quantity         INTEGER NOT NULL DEFAULT 1,
            capital_used    REAL,
            status          TEXT NOT NULL DEFAULT 'PENDING',
            entry_hit       INTEGER NOT NULL DEFAULT 0,
            entry_hit_at    TEXT,
            t1_hit          INTEGER NOT NULL DEFAULT 0,
            t1_hit_at       TEXT,
            t2_hit          INTEGER NOT NULL DEFAULT 0,
            t2_hit_at       TEXT,
            t3_hit          INTEGER NOT NULL DEFAULT 0,
            t3_hit_at       TEXT,
            sl_hit          INTEGER NOT NULL DEFAULT 0,
            sl_hit_at       TEXT,
            closed_price    REAL,
            pnl             REAL,
            pnl_pct         REAL,
            highest_price   REAL,
            lowest_price    REAL,
            opened_at       TEXT NOT NULL,
            closed_at       TEXT,
            expiry_at       TEXT,
            signal_strength TEXT,
            confidence      REAL,
            reasons         TEXT,
            notes           TEXT
        )
    """)
    # Migrate old tables: add new columns if missing
    try:
        conn.execute("SELECT entry_hit FROM paper_trades LIMIT 1")
    except sqlite3.OperationalError:
        for col, default in [
            ("entry_hit", "0"), ("entry_hit_at", "NULL"),
            ("t1_hit", "0"), ("t1_hit_at", "NULL"),
            ("t2_hit", "0"), ("t2_hit_at", "NULL"),
            ("t3_hit", "0"), ("t3_hit_at", "NULL"),
            ("sl_hit", "0"), ("sl_hit_at", "NULL"),
        ]:
            try:
                col_type = "INTEGER NOT NULL DEFAULT" if "hit_at" not in col else "TEXT DEFAULT"
                conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {col_type} {default}")
            except sqlite3.OperationalError:
                pass
        # Update old OPEN trades to PENDING
        conn.execute("UPDATE paper_trades SET status='PENDING' WHERE status='OPEN'")

    # Migrate: add exit_target column (T1/T2/T3 — which target to close at)
    try:
        conn.execute("SELECT exit_target FROM paper_trades LIMIT 1")
    except sqlite3.OperationalError:
        try:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN exit_target TEXT DEFAULT 'T1'")
        except sqlite3.OperationalError:
            pass

    # Migrate: add charges column to store actual brokerage+taxes per trade
    try:
        conn.execute("SELECT charges FROM paper_trades LIMIT 1")
    except sqlite3.OperationalError:
        try:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN charges REAL")
        except sqlite3.OperationalError:
            pass

    # Migrate: add setup_type and tags columns for trade journaling
    try:
        conn.execute("SELECT setup_type FROM paper_trades LIMIT 1")
    except sqlite3.OperationalError:
        try:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN setup_type TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass

    try:
        conn.execute("SELECT tags FROM paper_trades LIMIT 1")
    except sqlite3.OperationalError:
        try:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN tags TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_pt_status ON paper_trades(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pt_ticker ON paper_trades(ticker, trade_type)")
    conn.commit()
    conn.close()

    # Backfill charges for existing closed trades that have no charges recorded
    _migrate_charges()


def _migrate_charges():
    """Backfill the charges column for existing closed trades."""
    conn = _get_conn()
    trades = conn.execute(
        "SELECT id, trade_type, direction, entry_price, closed_price, quantity "
        "FROM paper_trades WHERE status NOT IN ('PENDING', 'ACTIVE') "
        "AND closed_price IS NOT NULL AND charges IS NULL"
    ).fetchall()
    for tid, tt, direction, entry, closed, qty in trades:
        is_long = direction in ("BUY", "LONG")
        charge_type = "delivery" if tt == "swing" else "intraday"
        buy_p = entry if is_long else closed
        sell_p = closed if is_long else entry
        ch = calc_angel_one_charges(buy_p, sell_p, qty, charge_type)
        conn.execute("UPDATE paper_trades SET charges=? WHERE id=?", (ch["total_charges"], tid))
    conn.commit()
    conn.close()


def place_trade(
    ticker: str,
    trade_type: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    target_1: float,
    target_2: float | None = None,
    target_3: float | None = None,
    quantity: int = 1,
    signal_strength: str = "",
    confidence: float = 0.0,
    reasons: str = "",
    notes: str = "",
    exit_target: str = "T1",
    setup_type: str = "",
    tags: str = "",
) -> int:
    """Place a new paper trade in PENDING state. Returns the trade ID.

    exit_target: which target to auto-close at ('T1', 'T2', or 'T3').
    Higher targets are still tracked as hit/not-hit for history.
    """
    now = datetime.now()
    if trade_type == "scalp":
        today_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        expiry = today_close if now <= today_close else (now + timedelta(days=1)).replace(hour=15, minute=30, second=0, microsecond=0)
    else:
        expiry = now + timedelta(days=7)

    position_value = entry_price * quantity
    # Scalp trades get 5x leverage (intraday margin) — only 20% margin needed
    if trade_type == "scalp":
        capital_used = position_value / 5
    else:
        capital_used = position_value

    # Check if sufficient funds available
    fund_bal = get_fund_balance(trade_type)
    if fund_bal["initial_capital"] > 0 and fund_bal["available"] < capital_used:
        raise ValueError(
            f"Insufficient {trade_type} funds. Need ₹{capital_used:,.2f} "
            f"{'(5× margin on ₹' + f'{position_value:,.0f}' + ')' if trade_type == 'scalp' else ''}"
            f" but only ₹{fund_bal['available']:,.2f} available."
        )

    # Deduct funds if fund tracking is active (initial_capital > 0)
    if fund_bal["initial_capital"] > 0:
        deduct_funds(trade_type, capital_used)

    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO paper_trades
           (ticker, trade_type, direction, entry_price, stop_loss,
            target_1, target_2, target_3, quantity, capital_used,
            status, entry_hit, t1_hit, t2_hit, t3_hit, sl_hit,
            highest_price, lowest_price, opened_at, expiry_at,
            signal_strength, confidence, reasons, notes, exit_target,
            setup_type, tags)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', 0, 0, 0, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticker, trade_type, direction, entry_price, stop_loss,
            target_1, target_2, target_3, quantity, capital_used,
            entry_price, entry_price, now.isoformat(), expiry.isoformat(),
            signal_strength, confidence, reasons, notes, exit_target,
            setup_type, tags,
        ),
    )
    conn.commit()
    trade_id = cur.lastrowid
    conn.close()
    return trade_id


def check_open_trades() -> dict:
    """Validate all PENDING and ACTIVE trades against price history.

    Lifecycle:
      PENDING: check if entry price was hit → becomes ACTIVE
      ACTIVE:  check SL and targets in order (only T1 after entry, T2 after T1, etc.)
    """
    import yfinance as yf
    from src.data_fetcher import _yf_retry

    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, ticker, trade_type, direction, entry_price, stop_loss, "
        "target_1, target_2, target_3, quantity, highest_price, lowest_price, "
        "expiry_at, opened_at, status, entry_hit, t1_hit, t2_hit, t3_hit, sl_hit, "
        "capital_used, exit_target "
        "FROM paper_trades WHERE status IN ('PENDING', 'ACTIVE')"
    ).fetchall()

    if not rows:
        conn.close()
        return {"checked": 0, "updated": 0, "details": []}

    now = datetime.now()
    now_iso = now.isoformat()
    today_str = now.strftime("%Y-%m-%d")
    updates = []
    fund_releases = []  # (fund_type, capital_used, pnl) tuples to process after commit

    # Group by ticker
    ticker_trades: dict[str, list] = {}
    for row in rows:
        ticker_trades.setdefault(row[1], []).append(row)

    for ticker, trades in ticker_trades.items():
        try:
            stock = yf.Ticker(ticker)
            hist = _yf_retry(lambda: stock.history(period="5d"))
            if hist.empty:
                continue
            current_price = hist["Close"].iloc[-1]
            day_high = hist["High"].iloc[-1]
            day_low = hist["Low"].iloc[-1]

            # Prepare date-indexed history for multi-day checks
            hist_clean = hist.copy()
            hist_clean.index = pd.to_datetime(hist_clean.index)
            if hist_clean.index.tz is not None:
                hist_clean.index = hist_clean.index.tz_localize(None)
        except Exception:
            continue

        for row in trades:
            (tid, _, trade_type, direction, entry, sl,
             t1, t2, t3, qty, peak, trough,
             expiry_str, opened_at_str, status,
             entry_hit, t1_hit, t2_hit, t3_hit, sl_hit,
             cap_used, exit_tgt) = row
            exit_tgt = exit_tgt or "T1"

            is_long = direction in ("BUY", "LONG")
            opened_today = opened_at_str and opened_at_str[:10] == today_str

            # Determine price range to check
            if opened_today:
                check_high = current_price
                check_low = current_price
            else:
                check_high = day_high
                check_low = day_low
                if opened_at_str:
                    opened_date = opened_at_str[:10]
                    after_open = hist_clean[hist_clean.index >= opened_date]
                    if len(after_open) > 1:
                        post_bars = after_open.iloc[1:]
                        if not post_bars.empty:
                            check_high = max(check_high, post_bars["High"].max())
                            check_low = min(check_low, post_bars["Low"].min())

            new_peak = max(peak or entry, check_high)
            new_trough = min(trough or entry, check_low)

            # Track what changed this check
            changed = False
            new_status = status
            final_price = None

            # --- STEP 1: Check entry fill ---
            if not entry_hit:
                entry_filled = False
                if is_long:
                    # For a buy: price must drop to or below entry (limit buy)
                    entry_filled = check_low <= entry
                else:
                    # For a sell/short: price must rise to or above entry
                    entry_filled = check_high >= entry

                if entry_filled:
                    entry_hit = 1
                    new_status = "ACTIVE"
                    changed = True
                    conn.execute(
                        "UPDATE paper_trades SET entry_hit=1, entry_hit_at=?, status='ACTIVE' WHERE id=?",
                        (now_iso, tid),
                    )
                else:
                    # Check expiry for unfilled orders
                    if expiry_str:
                        try:
                            if now > datetime.fromisoformat(expiry_str):
                                conn.execute(
                                    "UPDATE paper_trades SET status='EXPIRED', closed_at=?, "
                                    "pnl=0, pnl_pct=0, highest_price=?, lowest_price=? WHERE id=?",
                                    (now_iso, new_peak, new_trough, tid),
                                )
                                updates.append({"id": tid, "ticker": ticker, "status": "EXPIRED", "pnl": 0})
                                if cap_used:
                                    fund_releases.append((trade_type, cap_used, 0))
                                changed = True
                                continue
                        except ValueError:
                            pass
                    # Just update tracking
                    conn.execute(
                        "UPDATE paper_trades SET highest_price=?, lowest_price=? WHERE id=?",
                        (new_peak, new_trough, tid),
                    )
                    continue

            # --- STEP 2: Trade is ACTIVE — check SL and targets ---
            # Check stop loss first (SL can trigger anytime after entry)
            if not sl_hit:
                sl_triggered = False
                if is_long and check_low <= sl:
                    sl_triggered = True
                elif not is_long and check_high >= sl:
                    sl_triggered = True

                if sl_triggered:
                    sl_hit = 1
                    final_price = sl
                    new_status = "STOPPED_OUT"
                    changed = True

            # Check targets in order (T1 → T2 → T3), only if not stopped out
            # All targets are tracked for history, but trade closes at exit_target
            if new_status != "STOPPED_OUT":
                if not t1_hit:
                    t1_triggered = (check_high >= t1) if is_long else (check_low <= t1)
                    if t1_triggered:
                        t1_hit = 1
                        changed = True
                        conn.execute(
                            "UPDATE paper_trades SET t1_hit=1, t1_hit_at=? WHERE id=?",
                            (now_iso, tid),
                        )

                if t1_hit and t2 and not t2_hit:
                    t2_triggered = (check_high >= t2) if is_long else (check_low <= t2)
                    if t2_triggered:
                        t2_hit = 1
                        changed = True
                        conn.execute(
                            "UPDATE paper_trades SET t2_hit=1, t2_hit_at=? WHERE id=?",
                            (now_iso, tid),
                        )

                if t2_hit and t3 and not t3_hit:
                    t3_triggered = (check_high >= t3) if is_long else (check_low <= t3)
                    if t3_triggered:
                        t3_hit = 1
                        changed = True
                        conn.execute(
                            "UPDATE paper_trades SET t3_hit=1, t3_hit_at=? WHERE id=?",
                            (now_iso, tid),
                        )

            # Determine final status — close at the user's chosen exit_target
            if new_status == "STOPPED_OUT":
                final_price = sl
            elif exit_tgt == "T1" and t1_hit:
                new_status = "HIT_T1"
                final_price = t1
            elif exit_tgt == "T2" and t2_hit:
                new_status = "HIT_T2"
                final_price = t2
            elif exit_tgt == "T3" and t3_hit:
                new_status = "HIT_T3"
                final_price = t3

            # Check expiry for active trades
            if final_price is None and expiry_str:
                try:
                    if now > datetime.fromisoformat(expiry_str):
                        new_status = "EXPIRED"
                        final_price = current_price
                        changed = True
                except ValueError:
                    pass

            # If trade is resolved (has a final price), close it
            if final_price is not None:
                pnl = ((final_price - entry) if is_long else (entry - final_price)) * qty
                pnl_pct = (pnl / (entry * qty)) * 100
                charge_type = "delivery" if trade_type == "swing" else "intraday"
                buy_p = entry if is_long else final_price
                sell_p = final_price if is_long else entry
                ch = calc_angel_one_charges(buy_p, sell_p, qty, charge_type)
                charges = ch["total_charges"]
                net_pnl = round(pnl - charges, 2)
                conn.execute(
                    """UPDATE paper_trades
                       SET status=?, closed_price=?, pnl=?, pnl_pct=?, charges=?,
                           t1_hit=?, t2_hit=?, t3_hit=?, sl_hit=?,
                           highest_price=?, lowest_price=?, closed_at=?
                       WHERE id=?""",
                    (new_status, final_price, round(pnl, 2), round(pnl_pct, 2), charges,
                     t1_hit, t2_hit, t3_hit, sl_hit,
                     new_peak, new_trough, now_iso, tid),
                )
                updates.append({
                    "id": tid, "ticker": ticker, "status": new_status,
                    "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
                    "charges": charges, "net_pnl": net_pnl,
                })
                if cap_used:
                    fund_releases.append((trade_type, cap_used, net_pnl))
            elif changed:
                conn.execute(
                    "UPDATE paper_trades SET status=?, highest_price=?, lowest_price=? WHERE id=?",
                    (new_status, new_peak, new_trough, tid),
                )
                updates.append({"id": tid, "ticker": ticker, "status": new_status, "pnl": 0})
            else:
                conn.execute(
                    "UPDATE paper_trades SET highest_price=?, lowest_price=? WHERE id=?",
                    (new_peak, new_trough, tid),
                )

    conn.commit()
    conn.close()

    # Release funds for closed trades
    for ft, cu, pnl_amt in fund_releases:
        bal = get_fund_balance(ft)
        if bal["initial_capital"] > 0:
            release_funds(ft, cu, pnl_amt)

    return {"checked": len(rows), "updated": len(updates), "details": updates}


def close_trade(trade_id: int, close_price: float) -> dict:
    """Manually close a trade at given price."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT direction, entry_price, quantity, entry_hit, trade_type, capital_used FROM paper_trades "
        "WHERE id=? AND status IN ('PENDING', 'ACTIVE')",
        (trade_id,),
    ).fetchone()
    if not row:
        conn.close()
        return {"error": "Trade not found or already closed"}

    direction, entry, qty, entry_hit, trade_type, capital_used = row
    is_long = direction in ("BUY", "LONG")
    if entry_hit:
        pnl = ((close_price - entry) if is_long else (entry - close_price)) * qty
        pnl_pct = (pnl / (entry * qty)) * 100
        charge_type = "delivery" if trade_type == "swing" else "intraday"
        buy_p = entry if is_long else close_price
        sell_p = close_price if is_long else entry
        ch = calc_angel_one_charges(buy_p, sell_p, qty, charge_type)
        charges = ch["total_charges"]
        net_pnl = round(pnl - charges, 2)
    else:
        pnl = 0
        pnl_pct = 0
        charges = 0.0
        net_pnl = 0.0

    conn.execute(
        """UPDATE paper_trades
           SET status='CLOSED', closed_price=?, pnl=?, pnl_pct=?, charges=?, closed_at=?
           WHERE id=?""",
        (close_price, round(pnl, 2), round(pnl_pct, 2), charges,
         datetime.now().isoformat(), trade_id),
    )
    conn.commit()
    conn.close()

    # Release funds back using net P&L (after charges)
    fund_bal = get_fund_balance(trade_type)
    if fund_bal["initial_capital"] > 0 and capital_used:
        release_funds(trade_type, capital_used, net_pnl)

    return {"id": trade_id, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "charges": charges, "net_pnl": net_pnl}


def delete_trade(trade_id: int) -> dict:
    """Delete a PENDING trade (entry not yet hit). Refunds the reserved capital."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT trade_type, capital_used, status, entry_hit FROM paper_trades WHERE id=?",
        (trade_id,),
    ).fetchone()
    if not row:
        conn.close()
        return {"error": "Trade not found"}
    trade_type, capital_used, status, entry_hit = row
    if status not in ("PENDING",) or entry_hit:
        conn.close()
        return {"error": "Only PENDING trades (entry not hit) can be deleted"}
    conn.execute("DELETE FROM paper_trades WHERE id=?", (trade_id,))
    conn.commit()
    conn.close()
    # Refund capital (release with 0 P&L)
    if capital_used:
        fund_bal = get_fund_balance(trade_type)
        if fund_bal["initial_capital"] > 0:
            release_funds(trade_type, capital_used, 0)
    return {"id": trade_id, "deleted": True}


def get_open_trades(trade_type: str | None = None) -> pd.DataFrame:
    """Get all PENDING and ACTIVE paper trades."""
    conn = _get_conn()
    query = "SELECT * FROM paper_trades WHERE status IN ('PENDING', 'ACTIVE')"
    params = []
    if trade_type:
        query += " AND trade_type = ?"
        params.append(trade_type)
    query += " ORDER BY opened_at DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_trade_history(
    trade_type: str | None = None,
    ticker: str | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Get closed/resolved paper trades."""
    conn = _get_conn()
    query = "SELECT * FROM paper_trades WHERE status NOT IN ('PENDING', 'ACTIVE')"
    params: list = []
    if trade_type:
        query += " AND trade_type = ?"
        params.append(trade_type)
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    query += " ORDER BY closed_at DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_paper_stats(trade_type: str | None = None) -> dict:
    """Get paper trading performance statistics."""
    conn = _get_conn()
    where = "WHERE status NOT IN ('PENDING', 'ACTIVE')"
    params: list = []
    if trade_type:
        where += " AND trade_type = ?"
        params.append(trade_type)

    # Use net P&L (pnl - charges) for all statistics; fall back to gross if charges is NULL
    row = conn.execute(
        f"""SELECT
                COUNT(*) as total,
                SUM(CASE WHEN (pnl - COALESCE(charges,0)) > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN (pnl - COALESCE(charges,0)) <= 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl - COALESCE(charges,0)) as total_pnl,
                AVG(pnl_pct) as avg_pnl_pct,
                MAX(pnl - COALESCE(charges,0)) as best_trade,
                MIN(pnl - COALESCE(charges,0)) as worst_trade,
                SUM(CASE WHEN status LIKE 'HIT_T%%' THEN 1 ELSE 0 END) as targets_hit,
                SUM(CASE WHEN status = 'STOPPED_OUT' THEN 1 ELSE 0 END) as stopped_out,
                SUM(CASE WHEN status = 'EXPIRED' THEN 1 ELSE 0 END) as expired,
                AVG(CASE WHEN (pnl - COALESCE(charges,0)) > 0 THEN (pnl - COALESCE(charges,0)) END) as avg_win,
                AVG(CASE WHEN (pnl - COALESCE(charges,0)) <= 0 THEN (pnl - COALESCE(charges,0)) END) as avg_loss,
                SUM(CASE WHEN (pnl - COALESCE(charges,0)) > 0 THEN (pnl - COALESCE(charges,0)) ELSE 0 END) as total_profit,
                SUM(CASE WHEN (pnl - COALESCE(charges,0)) < 0 THEN (pnl - COALESCE(charges,0)) ELSE 0 END) as total_loss,
                SUM(COALESCE(charges,0)) as total_charges
           FROM paper_trades {where}""",
        params,
    ).fetchone()

    total = row[0] or 0
    wins = row[1] or 0
    losses = row[2] or 0

    # Open trades count (PENDING + ACTIVE)
    open_q = "SELECT COUNT(*) FROM paper_trades WHERE status IN ('PENDING', 'ACTIVE')"
    open_params: list = []
    if trade_type:
        open_q += " AND trade_type = ?"
        open_params.append(trade_type)
    open_count = conn.execute(open_q, open_params).fetchone()[0]

    conn.close()

    win_rate = (wins / total * 100) if total > 0 else 0
    avg_win = row[10] or 0
    avg_loss = abs(row[11]) if row[11] else 0
    profit_factor = (avg_win * wins) / (avg_loss * losses) if (avg_loss and losses) else 0

    return {
        "total_trades": total,
        "open_trades": open_count,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(row[3] or 0, 2),       # net P&L after charges
        "total_profit": round(row[12] or 0, 2),
        "total_loss": round(row[13] or 0, 2),
        "total_charges": round(row[14] or 0, 2),  # total charges paid
        "avg_pnl_pct": round(row[4] or 0, 2),
        "best_trade": round(row[5] or 0, 2),
        "worst_trade": round(row[6] or 0, 2),
        "targets_hit": row[7] or 0,
        "stopped_out": row[8] or 0,
        "expired": row[9] or 0,
        "profit_factor": round(profit_factor, 2),
    }


def clear_all_trades():
    """Delete all paper trades and reset fund balances. Returns count deleted."""
    conn = _get_conn()
    count = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    conn.execute("DELETE FROM paper_trades")
    # Reset fund balances to their initial capital (deployed → 0, available → initial)
    rows = conn.execute("SELECT fund_type, initial_capital FROM paper_funds").fetchall()
    for fund_type, initial in rows:
        conn.execute(
            "UPDATE paper_funds SET available=?, deployed=0, realized_pnl=0 WHERE fund_type=?",
            (initial, fund_type),
        )
    conn.commit()
    conn.close()
    return count


# ============================================================
# FUND MANAGEMENT
# ============================================================

def init_paper_funds():
    """Create paper_funds table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_funds (
            fund_type       TEXT PRIMARY KEY,
            initial_capital REAL NOT NULL DEFAULT 0,
            available       REAL NOT NULL DEFAULT 0,
            deployed        REAL NOT NULL DEFAULT 0,
            realized_pnl    REAL NOT NULL DEFAULT 0
        )
    """)
    # Seed default rows if missing
    for ft in ("swing", "scalp"):
        existing = conn.execute("SELECT 1 FROM paper_funds WHERE fund_type=?", (ft,)).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO paper_funds (fund_type, initial_capital, available, deployed, realized_pnl) "
                "VALUES (?, 0, 0, 0, 0)", (ft,),
            )
    conn.commit()
    conn.close()


def get_fund_balance(fund_type: str | None = None) -> dict | list[dict]:
    """Get fund balance for a trade type, or all if None."""
    conn = _get_conn()
    if fund_type:
        row = conn.execute(
            "SELECT fund_type, initial_capital, available, deployed, realized_pnl "
            "FROM paper_funds WHERE fund_type=?", (fund_type,),
        ).fetchone()
        conn.close()
        if not row:
            return {"fund_type": fund_type, "initial_capital": 0, "available": 0, "deployed": 0, "realized_pnl": 0}
        return {
            "fund_type": row[0], "initial_capital": row[1],
            "available": row[2], "deployed": row[3], "realized_pnl": row[4],
        }
    else:
        rows = conn.execute(
            "SELECT fund_type, initial_capital, available, deployed, realized_pnl FROM paper_funds"
        ).fetchall()
        conn.close()
        return [
            {"fund_type": r[0], "initial_capital": r[1], "available": r[2], "deployed": r[3], "realized_pnl": r[4]}
            for r in rows
        ]


def set_fund_capital(fund_type: str, amount: float):
    """Set the initial capital for a fund type. Adjusts available balance accordingly."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT initial_capital, available, deployed FROM paper_funds WHERE fund_type=?",
        (fund_type,),
    ).fetchone()
    if row:
        old_initial, old_available, deployed = row
        # Adjust available by the difference in initial capital
        diff = amount - old_initial
        new_available = max(0, old_available + diff)
        conn.execute(
            "UPDATE paper_funds SET initial_capital=?, available=? WHERE fund_type=?",
            (amount, new_available, fund_type),
        )
    else:
        conn.execute(
            "INSERT INTO paper_funds (fund_type, initial_capital, available, deployed, realized_pnl) "
            "VALUES (?, ?, ?, 0, 0)", (fund_type, amount, amount),
        )
    conn.commit()
    conn.close()


def deduct_funds(fund_type: str, amount: float) -> bool:
    """Deduct capital from available funds when placing a trade. Returns False if insufficient."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT available FROM paper_funds WHERE fund_type=?", (fund_type,),
    ).fetchone()
    if not row or row[0] < amount:
        conn.close()
        return False
    conn.execute(
        "UPDATE paper_funds SET available=available-?, deployed=deployed+? WHERE fund_type=?",
        (amount, amount, fund_type),
    )
    conn.commit()
    conn.close()
    return True


def release_funds(fund_type: str, capital_used: float, pnl: float):
    """Release deployed capital back to available + add P&L when a trade closes."""
    conn = _get_conn()
    conn.execute(
        "UPDATE paper_funds SET available=available+?+?, deployed=deployed-?, realized_pnl=realized_pnl+? "
        "WHERE fund_type=?",
        (capital_used, pnl, capital_used, pnl, fund_type),
    )
    conn.commit()
    conn.close()
