from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = BASE_DIR / 'clinical_prediction'
CLUSTER_DIR = BASE_DIR / 'phenotype_clustering'


def load_assets():
    model_path = next(PRED_DIR.glob('best_model_*.joblib'))
    model = joblib.load(model_path)
    with open(CLUSTER_DIR / 'cluster_metadata.json', 'r', encoding='utf-8') as f:
        cluster_meta = json.load(f)
    return model, cluster_meta


MODEL, CLUSTER_META = load_assets()


def build_input_dataframe(values: dict) -> pd.DataFrame:
    row = values.copy()
    row.setdefault('center_creator', 'Web input')
    row.setdefault('receiving_center', 'Web input')
    row.setdefault('sex', 'Male')
    row.setdefault('patient_status', 'On dialysis')
    row.setdefault('primary_disease', 'Unknown')
    row.setdefault('esa_use', 'Use')
    row.setdefault('esa_type', 'Unknown')
    row.setdefault('iron_use', 'Use' if row.get('iron_use_flag', 0) else 'No use')
    row.setdefault('hif_use', 'Use' if row.get('hif_use_flag', 0) else 'No use')
    row.setdefault('eq_esa_dose', row.get('esa_dose', 0.0) if row.get('esa_route') == 'Subcutaneous' else row.get('esa_dose', 0.0) * 2 / 3)
    hb = row.get('hb', np.nan)
    wt = row.get('dry_weight', np.nan)
    eq = row.get('eq_esa_dose', np.nan)
    row.setdefault('eri', eq / wt / (hb / 10.0) if hb and wt and hb > 0 and wt > 0 else np.nan)
    row.setdefault('esa_use_flag', 1)
    row.setdefault('prior_low_response_proxy', 0)
    for col in [
        'delta_hb',
        'delta_esa_dose',
        'delta_eri',
        'delta_crp',
        'delta_albumin',
        'delta_ktv',
        'delta_pth',
        'pre_sbp_q1_std',
        'pre_dbp_q1_std',
        'idh_count_q1',
        'pre_sbp_q2_mean',
        'pre_dbp_q2_mean',
        'idh_any_q2',
        'pre_sbp_q3_mean',
        'pre_dbp_q3_mean',
        'idh_any_q3',
    ]:
        row.setdefault(col, 0)
    return pd.DataFrame([row])


def assign_risk_level(prob: float) -> str:
    if prob <= 0.2:
        return 'Low'
    if prob <= 0.5:
        return 'Intermediate'
    if prob <= 0.8:
        return 'High'
    return 'Very high'


def assign_phenotype(values: dict) -> str:
    centroids = np.array(CLUSTER_META['centroids'])
    feature_order = CLUSTER_META['cluster_features']
    name_map = {int(k): v for k, v in CLUSTER_META['cluster_name_map'].items()}
    arr = np.array([values.get(col, 0) if values.get(col, 0) is not None else 0 for col in feature_order], dtype=float)
    dists = np.linalg.norm(centroids - arr, axis=1)
    return name_map[int(np.argmin(dists))]


def review_suggestions(values: dict) -> list[str]:
    suggestions = []
    if values.get('crp', 0) and values.get('crp', 0) > 5:
        suggestions.append('Review inflammation status first')
    if values.get('albumin', 999) < 35:
        suggestions.append('Review nutritional status first')
    if values.get('ktv', 99) < 1.2 or values.get('urr', 99) < 65:
        suggestions.append('Review dialysis adequacy first')
    if values.get('pth', 0) > 300:
        suggestions.append('Review CKD-MBD / PTH first')
    if values.get('idh_any_q1', 0) == 1:
        suggestions.append('Review hemodynamic instability first')
    return suggestions or ['Integrate with full clinical context']


def predict_case(values: dict) -> dict:
    df = build_input_dataframe(values)
    prob = float(MODEL.predict_proba(df)[0, 1])
    phenotype = assign_phenotype(values)
    return {
        'risk_score': prob,
        'risk_level': assign_risk_level(prob),
        'phenotype': phenotype,
        'suggestions': review_suggestions(values),
    }
