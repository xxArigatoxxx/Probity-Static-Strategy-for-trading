import json
from pathlib import Path
from typing import Iterator, List, Tuple
import numpy as np
import polars as pl
import xgboost as xgb
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import log_loss
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType


FEATURE_COLUMNS: List[str] = [
    "log_return",
    "realized_vol",
    "ofi_1h",
    "ofi_zscore",
    "kalman_residual",
    "kalman_zscore",
    "hilbert_phase_deg",
    "hilbert_sin",
    "hilbert_cos",
    "perm_entropy",
    "snr_db",
]


class ModelEngine:
    """
    โมดูลฝึกสอนโมเดล XGBoost ด้วยการ์ดจอ RTX 5060 (CUDA)
    พร้อมระบบ Purged Walk-Forward Cross-Validation, Temperature Scaling Calibration
    และแปลงโมเดลเป็นฟอร์แมต .onnx สำหรับทำ Real-time Inference
    """

    def __init__(self, model_dir: str = "./models") -> None:
        self.model_dir: Path = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_path: Path = self.model_dir / "xgb_xauusd_dsp.onnx"
        self.meta_path: Path = self.model_dir / "calibration_meta.json"
        self.temperature: float = 1.0

    @staticmethod
    def purged_walk_forward_split(
        n_samples: int,
        n_splits: int = 5,
        purge_bars: int = 12,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        [QUANT VALIDATION] Purged Walk-Forward Cross-Validation
        แบ่งข้อมูลตามลำดับเวลาและเว้นระยะห่าง (Purge Window) ระหว่างชุด Train และ Validation
        เพื่อป้องกันปัญหา Data Leakage จาก Rolling Window และ Target Horizon ที่ทับซ้อนกัน
        """
        fold_size = n_samples // (n_splits + 1)
        for i in range(1, n_splits + 1):
            train_end = fold_size * i
            val_start = train_end + purge_bars
            val_end = val_start + fold_size

            if val_end > n_samples:
                break

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx

    @staticmethod
    def fit_temperature_scaling(logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        [PROBABILITY CALIBRATION] Temperature Scaling:
        ปรับสเกล Raw Logits (z) ด้วยตัวแปรอุณหภูมิ T: P_calibrated = Sigmoid(z / T)
        เพื่อแก้ปัญหา Overconfidence ของ Tree-based Models ให้ค่า P(Win) สะท้อนความน่าจะเป็นจริง
        """
        def nll_objective(temp: np.ndarray) -> float:
            t_val = temp[0]
            scaled_probs = expit(logits / t_val)
            return float(log_loss(y_true, scaled_probs, labels=[0, 1]))

        res = minimize(
            nll_objective,
            x0=np.array([1.0]),
            bounds=[(0.05, 10.0)],
            method="L-BFGS-B",
        )
        return float(res.x[0])

    def train_and_export(self, df: pl.DataFrame) -> None:
        """
        รันกระบวนการ Purged Walk-Forward CV, เทรนโมเดลบน RTX 5060 GPU,
        คำนวณ Temperature Scaling และส่งออกเป็นไฟล์ .onnx
        """
        x_all = df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
        y_all = df["target_reversion_win"].to_numpy().astype(np.int32)

        oof_logits: List[np.ndarray] = []
        oof_targets: List[np.ndarray] = []

        # [GPU ACCELERATION] ตั้งค่าพารามิเตอร์ให้รันบน NVIDIA RTX 5060 (device="cuda", tree_method="hist")
        xgb_params = {
            "tree_method": "hist",
            "device": "cuda",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 300,
            "random_state": 42,
        }

        print("[INFO] เริ่มทำ Purged Walk-Forward Validation บน RTX 5060 (CUDA)...")
        for fold, (train_idx, val_idx) in enumerate(
            self.purged_walk_forward_split(len(x_all), n_splits=5, purge_bars=12), 1
        ):
            x_train, y_train = x_all[train_idx], y_all[train_idx]
            x_val, y_val = x_all[val_idx], y_all[val_idx]

            clf = xgb.XGBClassifier(**xgb_params)
            clf.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                verbose=False,
            )

            # ดึงค่า Raw Margin (Logits ก่อนเข้า Sigmoid) เพื่อนำไปคำนวณ Temperature Scaling
            val_logits = clf.predict(x_val, output_margin=True)
            oof_logits.append(val_logits)
            oof_targets.append(y_val)
            print(f"  -> Fold {fold} เสร็จสมบูรณ์ (Train: {len(train_idx)}, Val: {len(val_idx)})")

        # คำนวณค่า Temperature T ที่เหมาะสมที่สุดจาก Out-of-Fold Logits ทั้งหมด
        all_val_logits = np.concatenate(oof_logits)
        all_val_targets = np.concatenate(oof_targets)
        self.temperature = self.fit_temperature_scaling(all_val_logits, all_val_targets)
        print(f"[INFO] ค่า Optimal Temperature (T) สำหรับ Calibration = {self.temperature:.4f}")

        # เทรนโมเดลขั้นสุดท้าย (Final Production Model) บน RTX 5060
        final_clf = xgb.XGBClassifier(**xgb_params)
        final_clf.fit(x_all, y_all, verbose=False)

        # onnxmltools กำหนดให้ชื่อฟีเจอร์ใน Booster ต้องอยู่ในรูปแบบ f0, f1, ..., fn ตอนแปลงเป็น ONNX
        booster = final_clf.get_booster()
        booster.feature_names = [f"f{i}" for i in range(len(FEATURE_COLUMNS))]

        initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLUMNS)]))]
        onnx_model = onnxmltools.convert_xgboost(final_clf, initial_types=initial_type)

        with open(self.onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # บันทึกค่า Temperature และเกณฑ์ Threshold ลงไฟล์ Metadata
        vol_95 = float(np.percentile(df["realized_vol"].to_numpy(), 95))
        meta_data = {
            "temperature": self.temperature,
            "feature_columns": FEATURE_COLUMNS,
            "vol_95_threshold": vol_95,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4)

        print(f"[SUCCESS] บันทึกโมเดล ONNX สำเร็จที่: {self.onnx_path}")
