from typing import Dict, Any
import MetaTrader5 as mt5


class ExecutionEngine:
    """
    โมดูลส่งคำสั่งซื้อขายแบบจำกัดความเสี่ยงคงที่ (Fixed Risk Management)
    ควบคุมขนาด Lot คงที่, ระยะ Stop Loss / Take Profit คงที่ และจำกัด Slippage อย่างเข้มงวด
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        fixed_lot: float = 0.01,
        fixed_sl_points: int = 300,      # 300 points (หรือ 30 pips มาตรฐาน = $3.00 บนราคาทองคำ)
        reward_to_risk: float = 1.5,     # อัตราส่วน TP ต่อ SL (เช่น SL 300 -> TP 450 points)
        max_deviation_points: int = 15,  # คุม Slippage สูงสุดไม่เกิน 15 points
        magic_number: int = 506012,
    ) -> None:
        self.symbol: str = symbol
        self.fixed_lot: float = fixed_lot
        self.fixed_sl_points: int = fixed_sl_points
        self.reward_to_risk: float = reward_to_risk
        self.max_deviation_points: int = max_deviation_points
        self.magic_number: int = magic_number

    def has_open_position(self) -> bool:
        """ตรวจสอบว่ามีออเดอร์ของระบบนี้ถือครองอยู่หรือไม่ (จำกัด 1 นัดต่อรอบเพื่อคุมความเสี่ยง)"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return False
        return any(pos.magic == self.magic_number for pos in positions)

    def execute_order(self, signal: str, p_win: float) -> Dict[str, Any]:
        """
        ส่งคำสั่ง BUY หรือ SELL พร้อมแนบราคา Stop Loss, Take Profit และค่า Max Deviation
        """
        if signal not in ("BUY", "SELL"):
            return {"status": "IGNORED", "detail": "ไม่มีสัญญาณเข้าเทรด"}

        if self.has_open_position():
            return {"status": "BLOCKED", "detail": "มีสถานะที่เปิดค้างอยู่แล้ว (Max 1 Position Rule)"}

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return {"status": "ERROR", "detail": f"ไม่พบข้อมูลของสัญลักษณ์ {self.symbol}"}

        if not symbol_info.visible:
            mt5.symbol_select(self.symbol, True)

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return {"status": "ERROR", "detail": "ไม่สามารถดึงราคา Tick ปัจจุบันได้"}

        point = symbol_info.point
        digits = symbol_info.digits
        sl_distance = self.fixed_sl_points * point
        tp_distance = int(self.fixed_sl_points * self.reward_to_risk) * point

        if signal == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            entry_price = tick.ask
            sl_price = round(entry_price - sl_distance, digits)
            tp_price = round(entry_price + tp_distance, digits)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            entry_price = tick.bid
            sl_price = round(entry_price + sl_distance, digits)
            tp_price = round(entry_price - tp_distance, digits)

        # สร้างโครงสร้างคำสั่งเทรดที่มีการล็อกขนาด Lot, SL, TP และคุม Slippage (deviation)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.fixed_lot,
            "type": order_type,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": self.max_deviation_points,
            "magic": self.magic_number,
            "comment": f"DSP_MR_P{int(p_win * 100)}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or Cancel ป้องกันการได้ราคาหลุดกรอบ
        }

        result = mt5.order_send(request)
        if result is None:
            return {"status": "FAILED", "detail": f"order_send returned None: {mt5.last_error()}"}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "status": "REJECTED",
                "retcode": result.retcode,
                "detail": result.comment,
            }

        return {
            "status": "FILLED",
            "ticket": result.order,
            "signal": signal,
            "price": result.price,
            "sl": sl_price,
            "tp": tp_price,
            "lot": self.fixed_lot,
        }
