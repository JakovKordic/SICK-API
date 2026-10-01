import json
import joblib
import numpy as np
import torch
import torch.nn as nn


class GenericStackedTabularNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class FuelTheftPredictor:
    def __init__(self, artifact_dir="artifacts"):
        self.scaler = joblib.load(f"{artifact_dir}/scaler_mobilisis.joblib")
        self.rf_model = joblib.load(f"{artifact_dir}/random_forest_model.joblib")

        with open(f"{artifact_dir}/feature_columns.json", "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

        self.cyclical_feature_columns = [
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "day_of_month_sin",
            "day_of_month_cos",
            "month_sin",
            "month_cos"
        ]

        self.scaled_feature_columns = list(self.scaler.feature_names_in_)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = len(self.scaled_feature_columns) + len(self.cyclical_feature_columns) + 3

        self.model = GenericStackedTabularNet(
            input_dim=input_dim,
            num_classes=3
        )

        self.model.load_state_dict(
            torch.load(
                f"{artifact_dir}/stacked_rf_relu_model.pth",
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.index_to_class = {
            0: -1,
            1: 0,
            2: 1
        }

        self.class_to_label = {
            -1: "krađa",
            0: "normalno",
            1: "točenje"
        }

    def predict(self, feature_row):
        import pandas as pd

        X = pd.DataFrame([feature_row])[self.feature_columns]

        rf_probs = self.rf_model.predict_proba(X)

        X_scaled_part = self.scaler.transform(X[self.scaled_feature_columns])
        X_cyclical_part = X[self.cyclical_feature_columns].to_numpy()

        X_stacked = np.hstack([
            X_scaled_part,
            X_cyclical_part,
            rf_probs
        ])

        x_tensor = torch.tensor(
            X_stacked,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = self.index_to_class[pred_idx]

        probabilities = {
            self.class_to_label[self.index_to_class[i]]: float(probs[i])
            for i in range(len(probs))
        }

        return {
            "prediction_class": pred_class,
            "prediction_label": self.class_to_label[pred_class],
            "confidence": float(probs[pred_idx]),
            "probabilities": probabilities,
            "debug": {
                "input_features": feature_row,
                "scaled_feature_columns": self.scaled_feature_columns,
                "cyclical_feature_columns": self.cyclical_feature_columns,
                "mlp_input_order": (
                    self.scaled_feature_columns
                    + self.cyclical_feature_columns
                    + ["rf_proba_theft", "rf_proba_normal", "rf_proba_fueling"]
                ),
                "X_scaled_part": X_scaled_part[0].tolist(),
                "X_cyclical_part": X_cyclical_part[0].tolist(),
                "rf_probs": rf_probs[0].tolist(),
                "rf_classes": self.rf_model.classes_.tolist(),
                "mlp_probs": probs.tolist(),
                "pred_idx": pred_idx,
                "index_to_class": self.index_to_class
            }
        }