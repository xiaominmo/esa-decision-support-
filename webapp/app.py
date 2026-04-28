import streamlit as st
from inference import predict_case

st.set_page_config(page_title='ESA Decision Support', layout='wide')
st.title('ESA Low-Response Decision Support Prototype')
st.caption('Local prototype based on the offline-trained model and revised 2026-04-27 phenotype centroids.')

with st.sidebar:
    st.header('Patient-quarter input')
    age = st.number_input('Age', min_value=18.0, value=60.0)
    dialysis_age = st.number_input('Dialysis age (months)', min_value=3.0, value=24.0)
    hb = st.number_input('Hemoglobin (g/L)', min_value=40.0, value=100.0)
    esa_dose = st.number_input('ESA weekly dose', min_value=0.0, value=10000.0)
    esa_route = st.selectbox('ESA route', ['Subcutaneous', 'Intravenous'])
    dry_weight = st.number_input('Dry weight (kg)', min_value=20.0, value=60.0)
    crp = st.number_input('CRP', min_value=0.0, value=5.0)
    albumin = st.number_input('Albumin', min_value=10.0, value=35.0)
    ktv = st.number_input('Kt/V', min_value=0.1, value=1.2)
    urr = st.number_input('URR', min_value=1.0, value=65.0)
    pth = st.number_input('PTH', min_value=0.0, value=300.0)
    calcium = st.number_input('Calcium', min_value=0.0, value=2.2)
    phosphorus = st.number_input('Phosphorus', min_value=0.0, value=1.8)
    potassium = st.number_input('Potassium', min_value=0.0, value=4.8)
    sodium = st.number_input('Sodium', min_value=80.0, value=138.0)
    creatinine = st.number_input('Creatinine', min_value=0.0, value=800.0)
    iron_use_flag = 1 if st.checkbox('Iron use', value=True) else 0
    hif_use_flag = 1 if st.checkbox('HIF use', value=False) else 0
    current_pre_sbp_mean = st.number_input('Current-quarter pre-dialysis SBP', min_value=50.0, value=145.0)
    current_pre_dbp_mean = st.number_input('Current-quarter pre-dialysis DBP', min_value=30.0, value=80.0)
    current_idh_any = 1 if st.checkbox('Current-quarter intradialytic hypotension', value=False) else 0
    submitted = st.button('Run prediction')

if submitted:
    values = {
        'age': age,
        'dialysis_age': dialysis_age,
        'hb': hb,
        'esa_dose': esa_dose,
        'esa_route': esa_route,
        'esa_unit': 'IU',
        'dry_weight': dry_weight,
        'crp': crp,
        'albumin': albumin,
        'ktv': ktv,
        'urr': urr,
        'pth': pth,
        'calcium': calcium,
        'phosphorus': phosphorus,
        'potassium': potassium,
        'sodium': sodium,
        'creatinine': creatinine,
        'iron_use_flag': iron_use_flag,
        'hif_use_flag': hif_use_flag,
        'current_pre_sbp_mean': current_pre_sbp_mean,
        'current_pre_dbp_mean': current_pre_dbp_mean,
        'current_idh_any': current_idh_any,
        'pre_sbp_q1_mean': current_pre_sbp_mean,
        'pre_dbp_q1_mean': current_pre_dbp_mean,
        'idh_any_q1': current_idh_any,
    }
    result = predict_case(values)
    c1, c2, c3 = st.columns(3)
    c1.metric('Risk score', f"{result['risk_score']:.3f}")
    c2.metric('Risk level', result['risk_level'])
    c3.metric('Phenotype', result['phenotype'])

    st.subheader('Review suggestions')
    for item in result['suggestions']:
        st.write(f'- {item}')

    st.subheader('Input summary')
    st.json(values)
else:
    st.info('Enter patient-quarter data in the sidebar and click Run prediction.')
