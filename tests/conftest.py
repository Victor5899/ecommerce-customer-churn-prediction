import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from inference import schema_from_model
from train_models import build_preprocessor
from utils import ID_COL, TARGET_COL, separate_features_target

CATEGORIES = {
    "PreferredLoginDevice": ["Mobile Phone", "Computer", "Phone"],
    "PreferredPaymentMode": ["Debit Card", "Credit Card", "E wallet", "UPI", "COD"],
    "Gender": ["Male", "Female"],
    "PreferedOrderCat": ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"],
    "MaritalStatus": ["Married", "Single", "Divorced"],
}
COLUMN_ORDER = [
    ID_COL, TARGET_COL, "Tenure", "PreferredLoginDevice", "CityTier", "WarehouseToHome",
    "PreferredPaymentMode", "Gender", "HourSpendOnApp", "NumberOfDeviceRegistered",
    "PreferedOrderCat", "SatisfactionScore", "MaritalStatus", "NumberOfAddress", "Complain",
    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount",
]


def make_customers(n: int = 300, seed: int = 0, n_duplicates: int = 0, id_offset: int = 1000,
                   missing_rate: float = 0.03) -> pd.DataFrame:
    """Synthetic customers with the real schema; churn depends on tenure, complaints, cashback."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        ID_COL: np.arange(50001, 50001 + n),
        "Tenure": rng.integers(0, 40, n).astype(float),
        "CityTier": rng.integers(1, 4, n),
        "WarehouseToHome": rng.integers(5, 40, n).astype(float),
        "HourSpendOnApp": rng.integers(0, 5, n).astype(float),
        "NumberOfDeviceRegistered": rng.integers(1, 7, n),
        "SatisfactionScore": rng.integers(1, 6, n),
        "NumberOfAddress": rng.integers(1, 10, n),
        "Complain": rng.integers(0, 2, n),
        "OrderAmountHikeFromlastYear": rng.integers(11, 26, n).astype(float),
        "CouponUsed": rng.integers(0, 6, n).astype(float),
        "OrderCount": rng.integers(1, 10, n).astype(float),
        "DaySinceLastOrder": rng.integers(0, 20, n).astype(float),
        "CashbackAmount": np.round(rng.uniform(100, 300, n), 2),
    })
    for col, cats in CATEGORIES.items():
        df[col] = rng.choice(cats, n)
    logit = 1.0 - 0.15 * df["Tenure"] + 1.8 * df["Complain"] - 0.01 * (df["CashbackAmount"] - 180)
    df[TARGET_COL] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)

    for col in ["Tenure", "WarehouseToHome", "DaySinceLastOrder"]:
        df.loc[rng.uniform(size=n) < missing_rate, col] = np.nan

    if n_duplicates:
        copies = df.iloc[:n_duplicates].copy()
        copies[ID_COL] = copies[ID_COL] + id_offset
        df = pd.concat([df, copies], ignore_index=True)
    return df[COLUMN_ORDER]


@pytest.fixture
def customers():
    return make_customers()


@pytest.fixture(scope="session")
def make_df():
    return make_customers


@pytest.fixture(scope="session")
def trained_model():
    X, y = separate_features_target(make_customers(n=400, seed=1))
    pre, _, _ = build_preprocessor(X)
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    return model.fit(X, y)


@pytest.fixture(scope="session")
def schema(trained_model):
    return schema_from_model(trained_model)
