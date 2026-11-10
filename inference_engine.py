import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import onnxruntime as ort
from scipy.special import logit, expit


class InferenceEngine:
    """
    โมดูลพยากรณ์ผลแบบ Real-time บนการ์ดจอ RTX 5060 ผ่าน ONNX Runtime (CUDAExecutionProvider)
    พร้อมระบบตัดวงจรความเสี่ยง (Volatility/Entropy Circuit Breaker และ Spread Guard)
    """

    def __init__(
        self,
        onnx_path: str = "./models/xgb_xauusd_dsp.onnx",
        meta_path: str = "./models/calibration_meta.json",
        max_entropy_threshold: float = 0.92,
        min_win_prob: float = 0.62,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.meta_path = Path(meta_path)
        self.max_entropy_threshold = max_entropy_threshold
        self.min_win_prob = min_win_prob

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.temperature: float = float(meta["temperature"])
        self.feature_columns: list[str] = meta["feature_columns"]
        self.max_vol_threshold: float = float(meta["vol_95_threshold"])

        # [GPU INFERENCE] บังคับใช้ CUDAExecutionProvider บน RTX 5060 เป็นอันดับแรก
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def evaluate_and_predict(
        self, latest_row: Dict[str, Any], current_live_spread: float
    ) -> Dict[str, Any]:
        """
        ตรวจสอบสภาวะตลาดผ่านระบบ Guardrails ทั้งหมดก่อนรันโมเดลบน GPU:
        1. Spread Guard Filter: ป้องกันการเทรดช่วงข่าวแรงหรือสภาพคล่องบาง
        2. Volatility & Permutation Entropy Circuit Breaker: ปิดระบบเมื่อตลาดเป็น Trend หรือ Chaos
        3. ONNX GPU Inference + Temperature Scaling + DSP Phase Confirmation
        """
        spread_ma20 = float(latest_row["spread_ma20"])
        realized_vol = float(latest_row["realized_vol"])
        perm_entropy = float(latest_row["perm_entropy"])
        kalman_zscore = float(latest_row["kalman_zscore"])
        hilbert_phase = float(latest_row["hilbert_phase_deg"])

        # -------------------------------------------------------------------------
        # GUARDRAIL 1: Spread Guard Filter
        # หาก Spread ปัจจุบันสูงกว่าค่าเฉลี่ย 20 แท่งล่าสุดเกิน 2 เท่า ให้สั่ง SKIP ทันที
        # -------------------------------------------------------------------------
        if current_live_spread > (2.0 * spread_ma20) and spread_ma20 > 0:
            return {
                "action": "SKIP",
                "reason": f"Spread Guard Triggered: Live Spread ({current_live_spread}) > 2x MA20 ({spread_ma20:.1f})",
                "p_win": 0.0,
            }

        # -------------------------------------------------------------------------
        # GUARDRAIL 2: Volatility & Entropy Circuit Breaker
        # หากความผันผวนจริง หรือ ค่าความไร้ระเบียบ (Permutation Entropy) พุ่งทะลุเพดาน
        # แปลว่าตลาดกำลังเกิด Breakout รุนแรงหรือไร้ระเบียบ (Chaos) ห้ามสวนเทรนด์เด็ดขาด
        # -------------------------------------------------------------------------
        if realized_vol > self.max_vol_threshold:
            return {
                "action": "SKIP",
                "reason": f"Volatility Circuit Breaker: Realized Vol ({realized_vol:.6f}) > Threshold ({self.max_vol_threshold:.6f})",
                "p_win": 0.0,
            }

        if perm_entropy > self.max_entropy_threshold:
            return {
                "action": "SKIP",
                "reason": f"Entropy Circuit Breaker: Permutation Entropy ({perm_entropy:.4f}) > Threshold ({self.max_entropy_threshold})",
                "p_win": 0.0,
            }

        # -------------------------------------------------------------------------
        # GPU INFERENCE & TEMPERATURE SCALING CALIBRATION
        # -------------------------------------------------------------------------
        feature_vec = np.array(
            [[float(latest_row[col]) for col in self.feature_columns]],
            dtype=np.float32,
        )

        # รันการทำนายผ่าน ONNX Runtime บน RTX 5060
        onnx_outputs = self.session.run(None, {self.input_name: feature_vec})
        # ดึงความน่าจะเป็นของคลาส 1 (Win) จากเอาต์พุตของ ONNX
        prob_output = onnx_outputs[1]
        if isinstance(prob_output, list) and isinstance(prob_output[0], dict):
            raw_prob = float(prob_output[0][1])
        else:
            raw_prob = float(np.asarray(prob_output)[0, 1])

        # แปลงกลับเป็น Logit แล้วหารด้วย Temperature T เพื่อลดความมั่นใจเกินจริง (Overconfidence)
        raw_prob_clipped = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
        calibrated_prob = float(expit(logit(raw_prob_clipped) / self.temperature))

        if calibrated_prob < self.min_win_prob:
            return {
                "action": "SKIP",
                "reason": f"Low Calibrated P(Win): {calibrated_prob:.4f} < {self.min_win_prob}",
                "p_win": calibrated_prob,
            }

        # -------------------------------------------------------------------------
        # DSP SIGNAL CONFIRMATION (Kalman Stretch + Hilbert Wave Reversal Zone)
        # -------------------------------------------------------------------------
        # เงื่อนไข SELL (สวนลงเข้าหาค่าเฉลี่ย):
        # ราคาดีดสูงกว่าค่าสมดุล Kalman มาก (kalman_zscore > 1.2) และเฟสคลื่นอยู่ในโซนยอดคลื่น (45 - 165 องศา)
        if kalman_zscore > 1.2 and (45.0 <= hilbert_phase <= 165.0):
            return {
                "action": "SELL",
                "reason": f"DSP Peak Reversion (Z={kalman_zscore:.2f}, Phase={hilbert_phase:.1f} deg)",
                "p_win": calibrated_prob,
            }

        # เงื่อนไข BUY (สวนขึ้นเข้าหาค่าเฉลี่ย):
        # ราคาทุบต่ำกว่าค่าสมดุล Kalman มาก (kalman_zscore < -1.2) และเฟสคลื่นอยู่ในโซนท้องคลื่น (225 - 345 องศา)
        if kalman_zscore < -1.2 and (225.0 <= hilbert_phase <= 345.0):
            return {
                "action": "BUY",
                "reason": f"DSP Trough Reversion (Z={kalman_zscore:.2f}, Phase={hilbert_phase:.1f} deg)",
                "p_win": calibrated_prob,
            }

        return {
            "action": "SKIP",
            "reason": "DSP Phase / Kalman Stretch not in extreme zone",
            "p_win": calibrated_prob,
        }
