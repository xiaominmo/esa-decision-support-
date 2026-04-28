from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = BASE_DIR / 'clinical_prediction'
CLUSTER_DIR = BASE_DIR / 'phenotype_clustering'
CLUSTER_METADATA = CLUSTER_DIR / 'cluster_metadata.json'
CLUSTER_DATA = CLUSTER_DIR / 'phenotype_analysis_dataset_revised.csv'


def enrich_cluster_meta(cluster_meta: dict) -> dict:
    if {'imputation_values', 'scaler_mean', 'scaler_scale'}.issubset(cluster_meta):
        return cluster_meta
    if not CLUSTER_DATA.exists():
        return cluster_meta
    features = cluster_meta['cluster_features']
    df = pd.read_csv(CLUSTER_DATA, usecols=features)
    cluster_input = df.apply(pd.to_numeric, errors='coerce')
    imputation = cluster_input.median(numeric_only=True).fillna(0)
    filled = cluster_input.fillna(imputation).fillna(0)
    scale = filled.std(axis=0, ddof=0).replace(0, 1)
    cluster_meta = cluster_meta.copy()
    cluster_meta['imputation_values'] = imputation.to_dict()
    cluster_meta['scaler_mean'] = filled.mean(axis=0).to_dict()
    cluster_meta['scaler_scale'] = scale.to_dict()
    return cluster_meta


def load_assets():
    model_path = next(PRED_DIR.glob('best_model_*.joblib'))
    model = joblib.load(model_path)
    with open(CLUSTER_METADATA, 'r', encoding='utf-8') as f:
        cluster_meta = json.load(f)
    cluster_meta = enrich_cluster_meta(cluster_meta)
    return model, cluster_meta


MODEL, CLUSTER_META = load_assets()


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def build_input_dataframe(values: dict) -> pd.DataFrame:
    row = values.copy()
    row.setdefault('center_creator', 'Web input')
    row.setdefault('receiving_center', 'Web input')
    row.setdefault('sex', 'Male')
    row.setdefault('patient_status', 'On dialysis')
    row.setdefault('primary_disease', 'Unknown')
    row.setdefault('esa_use', 'Use')
    row.setdefault('esa_type', 'Unknown')
    row.setdefault('esa_unit', 'IU')
    row.setdefault('iron_use', 'Use' if row.get('iron_use_flag', 0) else 'No use')
    row.setdefault('hif_use', 'Use' if row.get('hif_use_flag', 0) else 'No use')
    row.setdefault('eq_esa_dose', row.get('esa_dose', 0.0) if row.get('esa_route') == 'Subcutaneous' else row.get('esa_dose', 0.0) * 2 / 3)
    hb = row.get('hb', np.nan)
    wt = row.get('dry_weight', np.nan)
    eq = row.get('eq_esa_dose', np.nan)
    row.setdefault('eri', eq / wt / (hb / 10.0) if hb and wt and hb > 0 and wt > 0 else np.nan)
    row.setdefault('esa_use_flag', 1)
    row.setdefault('prior_low_response_proxy', 0)
    row.setdefault('current_pre_sbp_mean', row.get('pre_sbp_q1_mean', np.nan))
    row.setdefault('current_pre_dbp_mean', row.get('pre_dbp_q1_mean', np.nan))
    row.setdefault('current_idh_any', row.get('idh_any_q1', 0))
    row.setdefault('pre_sbp_q1_mean', row.get('current_pre_sbp_mean', 0))
    row.setdefault('pre_dbp_q1_mean', row.get('current_pre_dbp_mean', 0))
    row.setdefault('pre_sbp_q1_std', 0)
    row.setdefault('pre_dbp_q1_std', 0)
    row.setdefault('idh_any_q1', row.get('current_idh_any', 0))
    row.setdefault('idh_count_q1', row.get('current_idh_any', 0))
    for col in ['delta_hb', 'delta_esa_dose', 'delta_eri', 'delta_crp', 'delta_albumin', 'delta_ktv', 'delta_pth', 'pre_sbp_q2_mean', 'pre_dbp_q2_mean', 'idh_any_q2', 'pre_sbp_q3_mean', 'pre_dbp_q3_mean', 'idh_any_q3']:
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


def prepare_phenotype_values(values: dict) -> dict:
    prepared = values.copy()
    crp = to_float(prepared.get('crp'))
    cap = to_float(CLUSTER_META.get('crp_p99_cap'))
    if np.isfinite(crp):
        crp_winsor99 = min(max(crp, 0.0), cap) if np.isfinite(cap) else max(crp, 0.0)
        prepared['crp_winsor99'] = crp_winsor99
        prepared['log_crp_w99'] = np.log1p(crp_winsor99)
    prepared.setdefault('current_pre_sbp_mean', prepared.get('pre_sbp_q1_mean', np.nan))
    prepared.setdefault('current_pre_dbp_mean', prepared.get('pre_dbp_q1_mean', np.nan))
    prepared.setdefault('current_idh_any', prepared.get('idh_any_q1', 0))
    return prepared


def assign_phenotype(values: dict) -> str:
    centroids = np.array(CLUSTER_META.get('centroids_scaled', CLUSTER_META.get('centroids')), dtype=float)
    feature_order = CLUSTER_META['cluster_features']
    name_map = {int(k): v for k, v in CLUSTER_META['cluster_name_map'].items()}
    prepared = prepare_phenotype_values(values)
    imputation = CLUSTER_META.get('imputation_values', {})
    mean = CLUSTER_META.get('scaler_mean', {})
    scale = CLUSTER_META.get('scaler_scale', {})
    arr = []
    for col in feature_order:
        value = to_float(prepared.get(col))
        if not np.isfinite(value):
            value = to_float(imputation.get(col, 0))
        if mean and scale:
            denom = to_float(scale.get(col, 1))
            value = (value - to_float(mean.get(col, 0))) / (denom if denom else 1)
        arr.append(value)
    dists = np.linalg.norm(centroids - np.array(arr, dtype=float), axis=1)
    return name_map[int(np.argmin(dists))]


def review_suggestions(values: dict, phenotype: str, risk_level: str, risk_score: float) -> list[str]:
    suggestions = []
    hb = to_float(values.get('hb'))
    eri = to_float(values.get('eri'))
    crp = to_float(values.get('crp'))
    albumin = to_float(values.get('albumin'))
    ktv = to_float(values.get('ktv'))
    urr = to_float(values.get('urr'))
    pth = to_float(values.get('pth'))
    calcium = to_float(values.get('calcium'))
    phosphorus = to_float(values.get('phosphorus'))
    pre_sbp = to_float(values.get('current_pre_sbp_mean'))
    pre_dbp = to_float(values.get('current_pre_dbp_mean'))
    current_idh = int(to_float(values.get('current_idh_any')) == 1)

    if risk_level in {'High', 'Very high'}:
        suggestions.append(f'Risk context: predicted ESA low-response risk is {risk_score:.3f} ({risk_level}); review reversible drivers before reflexive ESA dose escalation.')
    elif risk_level == 'Intermediate':
        suggestions.append(f'Risk context: predicted ESA low-response risk is {risk_score:.3f} ({risk_level}); prioritize modifiable contributors before routine ESA adjustment.')
    else:
        suggestions.append(f'Risk context: predicted ESA low-response risk is {risk_score:.3f} ({risk_level}); if clinical low response persists, check whether recent events or missing labs explain discordance.')

    if phenotype == 'MBD-dominant':
        suggestions.append('Phenotype-specific focus: MBD-dominant; prioritize CKD-MBD optimization using PTH, phosphorus, calcium, and MBD medication adherence before ESA intensification.')
    elif phenotype == 'Inflammation–underdialysis':
        suggestions.append('Phenotype-specific focus: Inflammation–underdialysis; look for inflammatory/nutritional burden and inadequate delivered dialysis before ESA intensification.')
    elif phenotype == 'IDH-dominant':
        suggestions.append('Phenotype-specific focus: IDH-dominant; review intradialytic hypotension tolerance, ultrafiltration burden, target dry weight, antihypertensive timing, and cardiac status.')
    else:
        suggestions.append(f'Phenotype-specific focus: {phenotype}; align the review with the dominant abnormal clinical domain.')

    patient_findings = []
    if np.isfinite(hb) and hb < 100:
        patient_findings.append(f'Hb {hb:.1f} g/L is low; confirm Hb trend, bleeding/hemolysis, iron status, inflammation, and recent ESA exposure.')
    if np.isfinite(eri) and eri > 15:
        patient_findings.append(f'ERI {eri:.1f} is high; treat reversible resistance drivers before further dose escalation when possible.')
    if np.isfinite(crp) and crp > 5:
        patient_findings.append(f'CRP {crp:.1f} is elevated; assess infection, vascular access inflammation, recent hospitalization, and chronic inflammatory burden.')
    if np.isfinite(albumin) and albumin < 35:
        patient_findings.append(f'Albumin {albumin:.1f} g/L is low; assess protein-energy wasting, inflammation, appetite, and dialysis adequacy.')
    if (np.isfinite(ktv) and ktv < 1.2) or (np.isfinite(urr) and urr < 65):
        ktv_text = f'{ktv:.2f}' if np.isfinite(ktv) else 'missing'
        urr_text = f'{urr:.1f}' if np.isfinite(urr) else 'missing'
        patient_findings.append(f'Dialysis adequacy is borderline/low (Kt/V {ktv_text}, URR {urr_text}%); review delivered treatment time, blood flow, dialyzer, access function, and missed or shortened sessions.')
    if np.isfinite(pth) and pth > 300:
        patient_findings.append(f'PTH {pth:.1f} is elevated; review secondary hyperparathyroidism control and MBD therapy.')
    if np.isfinite(phosphorus) and phosphorus > 1.78:
        patient_findings.append(f'Phosphorus {phosphorus:.2f} mmol/L is elevated; review dietary phosphate load, binder use, and dialysis phosphate removal.')
    if np.isfinite(calcium) and (calcium < 2.10 or calcium > 2.50):
        patient_findings.append(f'Calcium {calcium:.2f} mmol/L is outside the usual target range; interpret PTH and phosphate management in that context.')
    if current_idh:
        bp_text = ''
        if np.isfinite(pre_sbp) and np.isfinite(pre_dbp):
            bp_text = f' with pre-dialysis BP {pre_sbp:.0f}/{pre_dbp:.0f} mmHg'
        patient_findings.append(f'Current-quarter intradialytic hypotension is present{bp_text}; review ultrafiltration rate, dry weight, interdialytic weight gain, antihypertensive timing, dialysate temperature/sodium, and cardiac tolerance.')
    if values.get('iron_use_flag', 0) == 0:
        patient_findings.append('No current iron use is recorded; check ferritin and TSAT before attributing low response to ESA resistance alone.')

    if patient_findings:
        suggestions.extend(f'Patient-specific driver: {finding}' for finding in patient_findings)
    else:
        suggestions.append('Patient-specific driver: no single entered variable crosses the built-in review thresholds; use longitudinal Hb, ESA dose, ERI, iron indices, and recent clinical events to refine the assessment.')

    if phenotype == 'MBD-dominant':
        suggestions.append('Practical checklist: verify PTH/phosphorus/calcium trends, phosphate binder and vitamin D/calcimimetic adherence, dialysate calcium, and timing for reassessing Hb/ERI after MBD optimization.')
    elif phenotype == 'Inflammation–underdialysis':
        suggestions.append('Practical checklist: screen for infection/access inflammation, nutrition-inflammation syndrome, delivered Kt/V/URR, session time, blood flow, dialyzer performance, access recirculation, and treatment adherence.')
    elif phenotype == 'IDH-dominant':
        suggestions.append('Practical checklist: review interdialytic weight gain, ultrafiltration rate, target dry weight, antihypertensive schedule, dialysate cooling/sodium strategy, and need for cardiac evaluation.')

    return suggestions


def predict_case(values: dict) -> dict:
    df = build_input_dataframe(values)
    row = df.iloc[0].to_dict()
    prob = float(MODEL.predict_proba(df)[0, 1])
    phenotype = assign_phenotype(row)
    risk_level = assign_risk_level(prob)
    return {
        'risk_score': prob,
        'risk_level': risk_level,
        'phenotype': phenotype,
        'suggestions': review_suggestions(row, phenotype, risk_level, prob),
    }
