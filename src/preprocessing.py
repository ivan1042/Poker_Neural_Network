from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor():

    categorical_cols = ["position_norm"]
    numeric_cols = [
        "hero_r1",
        "hero_r2",
        "stage_pre",
        "suit_po",
        "stra_po",
        "gut_po",
        "facing_ratio"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("position", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("numerical", StandardScaler(), numeric_cols),
        ],
        remainder="passthrough",

    ).set_output(transform="pandas")

    return preprocessor