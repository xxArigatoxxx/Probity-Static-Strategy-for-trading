import time
from datetime import datetime
import MetaTrader5 as mt5

from data_engine import DataEngine
from model_engine import ModelEngine
from inference_engine import InferenceEngine
from execution_engine import ExecutionEngine


def initialize_and_train_pipeline(symbol: str = "XAUUSD") -> None:
    """
    ฟังก์ชันเริ่มต้นสำหรับดึงข้อมูล M5 ลง SSD (.parquet), คำนวณฟีเจอร์ DSP บน RAM 32GB
    และสั่งเทรนโมเดลด้วย RTX 5060 หากยังไม่มีไฟล์โมเดล .onnx
    """
    data_eng = DataEngine(symbol=symbol, ssd_data_dir="./data")
    model_eng = ModelEngine(model_dir="./models")

    print("[STEP 1] ดึงข้อมูลประวัติศาสตร์ XAUUSD M5 และบันทึกลง SSD (.parquet)...")
    raw_df = data_eng.fetch_m5_data(num_bars=50000)
    data_eng.save_to_parquet(raw_df)

    print("[STEP 2] โหลดไฟล์ Parquet ขึ้น RAM 32GB และคำนวณ DSP Features ด้วย Polars (12 Threads)...")
    in_memory_df = data_eng.load_from_parquet()
    feat_df = data_eng.build_features(in_memory_df)

    print("[STEP 3] เทรนโมเดล XGBoost บน RTX 5060 พร้อม Purged CV และแปลงเป็น ONNX...")
    model_eng.train_and_export(feat_df)


def run_realtime_loop(symbol: str = "XAUUSD") -> None:
    """
    ลูปทำงานแบบ Real-time บน Timeframe M5:
    ประมวลผลเมื่อปิดแท่งเทียน 5 นาทีใหม่ทุกครั้ง
    """
    data_eng = DataEngine(symbol=symbol, ssd_data_dir="./data")
    model_eng = ModelEngine(model_dir="./models")

    # หากยังไม่มีไฟล์ ONNX ให้รันการเทรนก่อน
    if not model_eng.onnx_path.exists():
        initialize_and_train_pipeline(symbol=symbol)

    inf_eng = InferenceEngine(
        onnx_path=str(model_eng.onnx_path),
        meta_path=str(model_eng.meta_path),
        max_entropy_threshold=0.92,
        min_win_prob=0.62,
    )
    exec_eng = ExecutionEngine(
        symbol=symbol,
        fixed_lot=0.01,
        fixed_sl_points=300,
        reward_to_risk=1.5,
        max_deviation_points=15,
    )

    last_bar_time = None
    print(f"\n[SYSTEM ONLINE] เริ่มเดินเครื่องระบบเทรด {symbol} (M5) DSP Mean-Reversion...")

    try:
        while True:
            if not data_eng.connect_mt5():
                time.sleep(5)
                continue

            # เช็คเวลาเปิดของแท่งเทียนล่าสุดเพื่อประมวลผลเฉพาะเมื่อขึ้นแท่ง M5 ใหม่
            latest_bar = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1)
            if latest_bar is None or len(latest_bar) == 0:
                time.sleep(1)
                continue

            current_bar_time = int(latest_bar[0]["time"])
            if current_bar_time != last_bar_time:
                last_bar_time = current_bar_time

                # ดึงข้อมูล 1,000 แท่งล่าสุดเข้า RAM เพื่อคำนวณ Rolling DSP อย่างรวดเร็ว
                df_recent = data_eng.fetch_m5_data(num_bars=1000)
                df_features = data_eng.build_features(df_recent)

                # ใช้แท่งเทียนที่เพิ่งปิดสมบูรณ์ (index -2 เพราะ -1 คือแท่งที่เพิ่งเปิดใหม่)
                closed_bar_row = df_features.row(-2, named=True)

                # ดึงค่า Live Spread ณ วินาทีปัจจุบันจาก MT5
                sym_info = mt5.symbol_info(symbol)
                live_spread = float(sym_info.spread) if sym_info else 999.0

                # ประเมิน Guardrails และทำนายผลด้วย ONNX บน RTX 5060
                decision = inf_eng.evaluate_and_predict(closed_bar_row, live_spread)
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{now_str}] Action: {decision['action']} | "
                    f"P(Win): {decision['p_win']:.2%} | Reason: {decision['reason']}"
                )

                # หากสัญญาณผ่านทุกด่าน ให้ส่งคำสั่งเทรดทันที
                if decision["action"] in ("BUY", "SELL"):
                    order_res = exec_eng.execute_order(
                        signal=decision["action"],
                        p_win=decision["p_win"],
                    )
                    print(f"  >>> [EXECUTION RESULT]: {order_res}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] หยุดการทำงานของระบบโดยผู้ใช้งาน ปิดการเชื่อมต่อ MT5...")
        mt5.shutdown()


if __name__ == "__main__":
    run_realtime_loop(symbol="XAUUSD")
