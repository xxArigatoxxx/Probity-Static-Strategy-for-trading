import os

# [HARDWARE OPTIMIZATION] บังคับให้ Polars ใช้ 12 Threads ของ Ryzen 5 5600X เต็มประสิทธิภาพ
# ต้องตั้งค่า Environment Variable ก่อนการ import polars
os.environ["POLARS_MAX_THREADS"] = "12"

from pathlib import Path
from typing import Optional
import MetaTrader5 as mt5
import numpy as np
import polars as pl
from scipy.signal import hilbert
from datetime import datetime, timezone


class DataEngine:
    """
    โมดูลจัดการข้อมูลความเร็วสูงแบบ In-Memory (RAM 32GB) และคำนวณฟีเจอร์จากทฤษฎี DSP
    ใช้ Polars (12 Threads) ร่วมกับ Scipy และจัดเก็บข้อมูลลง SSD ในรูปแบบ Parquet
    """

    def __init__(self, symbol: str = "XAUUSD", ssd_data_dir: str = "./data") -> None:
        self.symbol: str = symbol
        self.data_dir: Path = Path(ssd_data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path: Path = self.data_dir / f"{self.symbol.lower()}_m5.parquet"

    def connect_mt5(self) -> bool:
        """เริ่มต้นการเชื่อมต่อกับเทอร์มินัล MetaTrader 5"""
        if not mt5.initialize():
            print(f"[ERROR] MT5 Initialize Failed: {mt5.last_error()}")
            return False
        return True

    def fetch_m5_data(self, num_bars: int = 50000) -> pl.DataFrame:
        """
        ดึงข้อมูลแท่งเทียน 5 นาที (M5) จาก MT5 เข้าสู่หน่วยความจำหลัก (RAM) โดยตรง
        """
        if not self.connect_mt5():
            raise ConnectionError("ไม่สามารถเชื่อมต่อกับ MetaTrader 5 ได้")

        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, num_bars)
        if rates is None or len(rates) == 0:
            raise ValueError(f"ไม่พบข้อมูลสำหรับสัญลักษณ์ {self.symbol}")

        # แปลง NumPy Structured Array จาก MT5 เข้าสู่ Polars DataFrame บน RAM ทันที
        df = pl.from_numpy(rates)
        df = df.with_columns(
            pl.from_epoch("time", time_unit="s").alias("datetime")
        )
        return df

    def save_to_parquet(self, df: pl.DataFrame) -> None:
        """
        บันทึกข้อมูลลงบน SSD SATA ในรูปแบบ .parquet โดยใช้การบีบอัด Zstandard (zstd)
        เพื่อความเร็วในการอ่าน/เขียนสูงสุดและลดคอขวด I/O
        """
        df.write_parquet(self.parquet_path, compression="zstd")

    def load_from_parquet(self) -> pl.DataFrame:
        """โหลดข้อมูลจากไฟล์ .parquet บน SSD ขึ้นสู่ RAM 32GB ทั้งก้อน"""
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ Parquet ที่ {self.parquet_path}")
        return pl.read_parquet(self.parquet_path)

    @staticmethod
    def _compute_kalman_filter(
        prices: np.ndarray, q: float = 1e-5, r: float = 1e-3
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        [DSP THEORY] 1D Kalman Filter สำหรับสกัดสถานะที่แท้จริง (True Equilibrium Price)
        ออกจากสัญญาณรบกวน (Market Microstructure Noise)
        - q (Process Variance): ความแปรปรวนของการเปลี่ยนแปลงสถานะจริง
        - r (Measurement Variance): ความแปรปรวนของสัญญาณรบกวนในตลาด
        - Residual = ราคาตลาดจริง - ราคาประเมินจาก Kalman (ใช้บอกความตึงตัวก่อนเด้งกลับ)
        """
        n = len(prices)
        x_hat = np.zeros(n, dtype=np.float64)
        p = np.zeros(n, dtype=np.float64)
        residuals = np.zeros(n, dtype=np.float64)

        x_hat[0] = prices[0]
        p[0] = 1.0

        for k in range(1, n):
            # 1. ขั้นตอนการทำนาย (Time Update / Prediction)
            x_hat_minus = x_hat[k - 1]
            p_minus = p[k - 1] + q

            # 2. ขั้นตอนการปรับปรุงด้วยข้อมูลจริง (Measurement Update / Correction)
            k_gain = p_minus / (p_minus + r)
            innovation = prices[k] - x_hat_minus
            x_hat[k] = x_hat_minus + k_gain * innovation
            p[k] = (1.0 - k_gain) * p_minus
            residuals[k] = innovation

        return x_hat, residuals

    @staticmethod
    def _compute_hilbert_phase(detrended_signal: np.ndarray, window: int = 128) -> np.ndarray:
        """
        [DSP THEORY] Hilbert Transform เพื่อคำนวณหา Analytic Signal:
        z(t) = x(t) + j * H{x(t)}
        และสกัดเอา Instantaneous Phase (มุมเฟสขนาด 0 ถึง 360 องศา) แบบ Zero-Lag
        เพื่อระบุจุดยอดคลื่น (Peak ~ 90, 270 deg) และท้องคลื่นสำหรับกลยุทธ์ Mean Reversion
        ใช้ Sliding Window เพื่อป้องกันการรั่วไหลของข้อมูลอนาคต (Look-ahead Bias) ใน FFT
        """
        n = len(detrended_signal)
        phases_deg = np.zeros(n, dtype=np.float64)

        for i in range(window, n):
            segment = detrended_signal[i - window + 1 : i + 1]
            # ลบค่าเฉลี่ยในหน้าต่างย่อยเพื่อให้สัญญาณแกว่งรอบศูนย์ (Zero-mean oscillation)
            segment_zero_mean = segment - np.mean(segment)
            analytic_signal = hilbert(segment_zero_mean)
            # ดึงมุมเฟสของจุดข้อมูลปัจจุบัน (จุดสุดท้ายในหน้าต่าง) แปลงจากเรเดียนเป็น 0-360 องศา
            phase_rad = np.angle(analytic_signal[-1])
            phases_deg[i] = np.degrees(phase_rad) % 360.0

        return phases_deg

    @staticmethod
    def _compute_permutation_entropy(
        series: np.ndarray, order: int = 3, window: int = 60
    ) -> np.ndarray:
        """
        [INFORMATION THEORY] Permutation Entropy (Normalized Shannon Entropy ของลำดับรูปแบบราคา)
        ใช้วัดความไร้ระเบียบ (Complexity / Chaos) ของอนุกรมเวลา:
        - ค่าเข้าใกล้ 0: ตลาดมีรูปแบบชัดเจนหรือกำลังไหลเป็นเทรนด์ทางเดียวอย่างรุนแรง
        - ค่าเข้าใกล้ 1: ตลาดไร้ระเบียบสูงสุด (Pure Random Walk / Chaos)
        """
        n = len(series)
        pe_out = np.ones(n, dtype=np.float64)
        max_entropy = np.log2(np.math.factorial(order))

        # สร้าง Matrix ของลำดับย่อยด้วย Stride Tricks เพื่อความเร็วสูง
        if n < window + order:
            return pe_out

        for i in range(window, n):
            sub = series[i - window : i]
            m = len(sub) - order + 1
            partitions = np.array([sub[j : j + order] for j in range(m)])
            # หาลำดับการเรียงตัว (Ordinal Patterns)
            ranks = np.argsort(np.argsort(partitions, axis=1), axis=1)
            # นับความถี่ของแต่ละ Pattern
            _, counts = np.unique(ranks, axis=0, return_counts=True)
            probs = counts / m
            shannon_entropy = -np.sum(probs * np.log2(probs + 1e-12))
            pe_out[i] = shannon_entropy / max_entropy

        return pe_out

    def build_features(
        self, df: pl.DataFrame, horizon: int = 3, rev_threshold: float = 0.0004
    ) -> pl.DataFrame:
        """
        คำนวณฟีเจอร์ทั้งหมดด้วย Polars (Multi-threaded 12 Threads) และ Scipy DSP
        โดยไม่ใช้ Technical Indicators แบบดั้งเดิมแม้แต่ตัวเดียว
        """
        # 1. คำนวณ Statistical & Order Flow Features ด้วย Polars Expressions (ขนาน 12 Threads)
        # - Log Returns: ln(P_t / P_{t-1}) เพื่อทำให้ข้อมูลนิ่ง (Stationary)
        # - Realized Volatility: ส่วนเบี่ยงเบนมาตรฐานของ Log Returns ในหน้าต่างสั้น 20 แท่ง
        # - Order Flow Imbalance (OFI): ประยุกต์กฎ Tick Rule / BVC จากตำแหน่งราคาปิดในกรอบ High-Low
        #   คูณกับ Tick Volume เพื่อแยกแรงซื้อรุก (Aggressive Buy) และแรงขายรุก (Aggressive Sell)
        df = df.with_columns([
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return"),
            (
                ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close")))
                / (pl.col("high") - pl.col("low") + 1e-9)
                * pl.col("tick_volume")
            ).alias("raw_ofi"),
        ])

        df = df.with_columns([
            pl.col("log_return").rolling_std(window_size=20).alias("realized_vol"),
            pl.col("raw_ofi").rolling_sum(window_size=12).alias("ofi_1h"),
            (
                (pl.col("raw_ofi") - pl.col("raw_ofi").rolling_mean(window_size=20))
                / (pl.col("raw_ofi").rolling_std(window_size=20) + 1e-9)
            ).alias("ofi_zscore"),
            pl.col("spread").rolling_mean(window_size=20).alias("spread_ma20"),
        ])

        # 2. คำนวณฟีเจอร์กลุ่ม DSP (Kalman Filter, Hilbert Transform, Permutation Entropy, SNR)
        close_np = df["close"].to_numpy()
        kalman_state, kalman_res = self._compute_kalman_filter(close_np, q=1e-5, r=1e-3)

        # คำนวณ Hilbert Phase จากสัญญาณที่ตัดแนวโน้มออกแล้ว (Kalman Residuals)
        hilbert_phase = self._compute_hilbert_phase(kalman_res, window=64)

        # แปลงมุมเฟส 0-360 องศาเป็น Sine/Cosine เพื่อให้โมเดล ML เข้าใจความต่อเนื่องของวงกลม (0 == 360)
        phase_rad = np.radians(hilbert_phase)
        phase_sin = np.sin(phase_rad)
        phase_cos = np.cos(phase_rad)

        # คำนวณ Permutation Entropy บน Log Returns
        log_ret_np = df["log_return"].fill_null(0.0).to_numpy()
        perm_entropy = self._compute_permutation_entropy(log_ret_np, order=3, window=48)

        # นำผลลัพธ์ DSP กลับเข้าสู่ Polars DataFrame
        df = df.with_columns([
            pl.Series("kalman_state", kalman_state),
            pl.Series("kalman_residual", kalman_res),
            pl.Series("hilbert_phase_deg", hilbert_phase),
            pl.Series("hilbert_sin", phase_sin),
            pl.Series("hilbert_cos", phase_cos),
            pl.Series("perm_entropy", perm_entropy),
        ])

        # 3. คำนวณ Normalized Kalman Z-Score และ Signal-to-Noise Ratio (SNR) ในหน่วยเดซิเบล (dB)
        # SNR (dB) = 10 * log10(Var(Signal) / Var(Noise))
        df = df.with_columns([
            (
                pl.col("kalman_residual")
                / (pl.col("kalman_residual").rolling_std(window_size=20) + 1e-9)
            ).alias("kalman_zscore"),
            (
                10.0
                * (
                    (pl.col("kalman_state").diff().rolling_var(window_size=20) + 1e-12)
                    / (pl.col("kalman_residual").rolling_var(window_size=20) + 1e-12)
                ).log10()
            ).alias("snr_db"),
        ])

        # 4. สร้าง Target สำหรับการเทรน Mean Reversion:
        # ถ้าราคาตึงตัวออกจากค่าสมดุล Kalman (|kalman_zscore| > 1.0) แล้วภายใน `horizon` แท่งถัดไป
        # ราคาเคลื่อนที่ย้อนกลับเข้าหาค่าเฉลี่ย Kalman เกินกว่า `rev_threshold` ให้ถือว่า Win (Target = 1)
        future_ret = (pl.col("close").shift(-horizon) - pl.col("close")) / pl.col("close")
        df = df.with_columns([
            pl.when(
                (pl.col("kalman_zscore") > 1.0) & (future_ret < -rev_threshold)
            )
            .then(1)
            .when(
                (pl.col("kalman_zscore") < -1.0) & (future_ret > rev_threshold)
            )
            .then(1)
            .otherwise(0)
            .alias("target_reversion_win")
        ])

        # ลบแถวที่มีค่า Null ช่วงเริ่มต้นของหน้าต่างคำนวณ
        return df.drop_nulls()
