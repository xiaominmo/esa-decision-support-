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

PRIORITY_ORDER = {'High': 0, 'Moderate': 1, 'Context': 2}


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


def derive_case_features(values: dict) -> dict:
    esa_dose = float(values.get('esa_dose', 0.0) or 0.0)
    hb = float(values.get('hb', 0.0) or 0.0)
    weight = float(values.get('dry_weight', 0.0) or 0.0)
    route = values.get('esa_route')
    eq_esa_dose = esa_dose if route == 'Subcutaneous' else esa_dose * 2 / 3
    eri = np.nan
    if hb > 0 and weight > 0:
        eri = eq_esa_dose / weight / (hb / 10.0)
    return {'eq_esa_dose': eq_esa_dose, 'eri': eri}


def make_review_item(priority: str, title: str, rationale: str, actions: list[str]) -> dict:
    return {
        'priority': priority,
        'title': title,
        'rationale': rationale,
        'actions': actions,
    }


def build_review_summary(risk_level: str, phenotype: str, items: list[dict]) -> str:
    if items:
        top_priority = items[0]['priority']
        return (
            f"Predicted risk is {risk_level.lower()} and the case pattern is closest to "
            f"'{phenotype}'. Prioritize the {top_priority.lower()} review items below before "
            "assuming isolated ESA dose escalation is the best next step."
        )
    return (
        f"Predicted risk is {risk_level.lower()} with a phenotype pattern closest to "
        f"'{phenotype}'. No major rule-based flags were triggered from the limited web inputs, "
        "so trend data, iron studies, bleeding history, and recent intercurrent illness still "
        "need clinician review."
    )


def review_suggestions(values: dict, risk_level: str, phenotype: str) -> dict:
    derived = derive_case_features(values)
    eq_esa_dose = derived['eq_esa_dose']
    eri = derived['eri']
    items = []

    crp = float(values.get('crp', 0.0) or 0.0)
    if crp > 5:
        items.append(
            make_review_item(
                'High',
                'Inflammatory burden may be contributing to ESA hyporesponse',
                f"CRP is {crp:.1f}, above the review threshold of 5.",
                [
                    'Look for reversible inflammatory sources such as vascular access infection, intercurrent illness, or tissue inflammation.',
                    'Interpret CRP together with albumin trend and recent dialysis events before further ESA escalation.',
                    'If inflammation is active, recheck Hb response after the reversible driver is treated.',
                ],
            )
        )

    albumin = float(values.get('albumin', 0.0) or 0.0)
    if albumin < 35:
        items.append(
            make_review_item(
                'High',
                'Nutritional or catabolic stress deserves targeted review',
                f"Albumin is {albumin:.1f} g/L, below the review threshold of 35 g/L.",
                [
                    'Review recent oral intake, weight trend, catabolic illness, and gastrointestinal losses.',
                    'Low albumin together with elevated CRP should raise concern for combined inflammation and malnutrition.',
                    'Consider correcting reversible nutritional barriers before attributing poor response only to ESA dosing.',
                ],
            )
        )

    ktv = float(values.get('ktv', 0.0) or 0.0)
    urr = float(values.get('urr', 0.0) or 0.0)
    if ktv < 1.2 or urr < 65:
        items.append(
            make_review_item(
                'High',
                'Delivered dialysis adequacy may be limiting anemia control',
                f"Kt/V is {ktv:.2f} and URR is {urr:.1f}%; the review targets here are Kt/V >= 1.2 and URR >= 65%.",
                [
                    'Review delivered treatment time, missed or shortened sessions, blood flow, dialyzer performance, and access function.',
                    'If adequacy is persistently low, address that before further increasing ESA exposure.',
                    'Repeat anemia assessment after adequacy issues are corrected to see whether dose demand falls.',
                ],
            )
        )

    pth = float(values.get('pth', 0.0) or 0.0)
    phosphorus = float(values.get('phosphorus', 0.0) or 0.0)
    if pth > 300 or phosphorus > 1.8:
        items.append(
            make_review_item(
                'High',
                'CKD-MBD control may be worsening ESA responsiveness',
                f"PTH is {pth:.0f} and phosphorus is {phosphorus:.2f}; these values suggest CKD-MBD review is warranted.",
                [
                    'Review phosphate binder adherence, dialysis phosphate clearance, vitamin D strategy, and calcimimetic use if applicable.',
                    'Interpret PTH together with calcium and phosphorus trends instead of a single isolated value.',
                    'Correcting uncontrolled CKD-MBD may reduce the need for further ESA escalation.',
                ],
            )
        )

    idh_any_q1 = int(values.get('idh_any_q1', 0) or 0)
    idh_count_q1 = int(values.get('idh_count_q1', 0) or 0)
    sbp = float(values.get('pre_sbp_q1_mean', 0.0) or 0.0)
    dbp = float(values.get('pre_dbp_q1_mean', 0.0) or 0.0)
    if idh_any_q1 == 1 or idh_count_q1 > 0 or sbp < 110 or dbp < 60:
        rationale_bits = []
        if idh_any_q1 == 1 or idh_count_q1 > 0:
            rationale_bits.append(f"IDH flag/count: {idh_any_q1}/{idh_count_q1}")
        if sbp < 110:
            rationale_bits.append(f"pre-dialysis SBP {sbp:.0f}")
        if dbp < 60:
            rationale_bits.append(f"pre-dialysis DBP {dbp:.0f}")
        items.append(
            make_review_item(
                'High',
                'Hemodynamic instability may be interfering with effective treatment delivery',
                '; '.join(rationale_bits) if rationale_bits else 'Hemodynamic review triggered by the current input pattern.',
                [
                    'Review ultrafiltration strategy, dry-weight target, and antihypertensive timing.',
                    'Check whether hypotension is shortening sessions or reducing delivered dialysis dose.',
                    'If instability is recurrent, reassess anemia management after hemodynamic tolerance improves.',
                ],
            )
        )

    hb = float(values.get('hb', 0.0) or 0.0)
    iron_use_flag = int(values.get('iron_use_flag', 0) or 0)
    if iron_use_flag == 0 and (hb < 100 or eq_esa_dose >= 10000):
        items.append(
            make_review_item(
                'Moderate',
                'Iron availability should be checked before assuming ESA resistance',
                f"Iron use is currently unchecked, while hemoglobin is {hb:.1f} g/L and estimated equivalent ESA dose is {eq_esa_dose:.0f} IU/week.",
                [
                    'Review ferritin, transferrin saturation, recent blood loss, and access-related blood loss.',
                    'If iron indices are borderline, correct iron deficiency before making large ESA dose increases.',
                    'Interpret iron need in the context of inflammation because functional iron deficiency can coexist with elevated CRP.',
                ],
            )
        )

    if hb < 100 and (eq_esa_dose >= 10000 or (not np.isnan(eri) and eri >= 15)):
        eri_text = 'not available' if np.isnan(eri) else f'{eri:.2f}'
        items.append(
            make_review_item(
                'Moderate',
                'Current ESA exposure is high relative to the observed hemoglobin response',
                f"Hemoglobin is {hb:.1f} g/L, equivalent ESA dose is {eq_esa_dose:.0f} IU/week, and estimated ERI is {eri_text}.",
                [
                    'Confirm that the recorded ESA route and dose are accurate before interpreting the result.',
                    'Review reversible drivers first instead of escalating ESA dose in isolation.',
                    'Use repeat Hb trend and interval clinical events to judge whether poor response is persistent or transient.',
                ],
            )
        )

    if phenotype == 'Hemodynamic instability / mixed phenotype':
        items.append(
            make_review_item(
                'Context',
                'Phenotype context: hemodynamic instability / mixed phenotype',
                'The clustering pattern suggests blood-pressure tolerance and mixed competing drivers may be part of the low-response profile.',
                [
                    'Prioritize intradialytic tolerance, session completion, and reversible inflammatory or nutritional triggers.',
                ],
            )
        )
    elif phenotype == 'CKD-MBD dominant phenotype':
        items.append(
            make_review_item(
                'Context',
                'Phenotype context: CKD-MBD dominant phenotype',
                'The clustering pattern is most consistent with mineral-bone disorder burden as a leading contributor.',
                [
                    'Review CKD-MBD trend data and treatment adherence before escalating anemia-directed therapy alone.',
                ],
            )
        )
    else:
        items.append(
            make_review_item(
                'Context',
                'Phenotype context: dialysis inadequacy / high-burden phenotype',
                'The clustering pattern suggests treatment burden and dialysis delivery should be reviewed carefully.',
                [
                    'Check adequacy, missed sessions, access performance, and intercurrent illness together rather than focusing on a single laboratory value.',
                ],
            )
        )

    items.sort(key=lambda item: PRIORITY_ORDER[item['priority']])
    return {
        'summary': build_review_summary(risk_level, phenotype, items),
        'items': items,
    }


def predict_case(values: dict) -> dict:
    df = build_input_dataframe(values)
    prob = float(MODEL.predict_proba(df)[0, 1])
    risk_level = assign_risk_level(prob)
    phenotype = assign_phenotype(values)
    review = review_suggestions(values, risk_level, phenotype)
    return {
        'risk_score': prob,
        'risk_level': risk_level,
        'phenotype': phenotype,
        'review_summary': review['summary'],
        'suggestions': review['items'],
    }
