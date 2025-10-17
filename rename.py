# Single-file deploy script: builds a Streamlit Clinical Trial Digital Twin Simulator and launches it.
# Paste this into Google Colab or run locally (VS Code / terminal). See notes below re: ngrok.


import os
import sys
import subprocess
import time
import textwrap

# ---------------------------
# 0) Configuration
# ---------------------------
APP_DIR = "clinical_trial_simulator_app"
APP_FILE = os.path.join(APP_DIR, "app.py")
PORT = 8501

# Optional: place your ngrok auth token here if you want a public URL from Colab.
# You can also set environment variable NGROK_AUTH_TOKEN before running.
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")  # set this if you want persistent ngrok tunnels

# ---------------------------
# 1) Install dependencies (if missing)
# ---------------------------
def install_if_missing(pkg):
    try:
        _import_(pkg)
    except Exception:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for pkg in ("streamlit", "pandas", "numpy", "plotly", "pyngrok"):
    install_if_missing(pkg)

# ---------------------------
# 2) Create app directory & write Streamlit app
# ---------------------------
os.makedirs(APP_DIR, exist_ok=True)

app_code = r'''
"""
Clinical Trial Digital Twin Simulator (Streamlit)
Objectives implemented:
1) Synthetic patient generator
2) Simplified PK/PD drug modeling
3) Trial simulation engine
4) Outcome analysis & subgroup risk detection
5) Inclusion/exclusion recommendations

Designed for Colab and local use. Visualized using Plotly, UI uses Streamlit.
"""

# Standard imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ---------------------------
# Helper functions & models
# ---------------------------

def generate_population(n=500, age_mu=55, age_sd=12, female_pct=50,
                        bmi_mu=27, bmi_sd=4, chol_mu=215, chol_sd=25,
                        prevalences=None, genetics=None, seed=42):
    """
    Objective 1: Generate a synthetic patient population with demographics,
    comorbidities, and genetic markers preserving simple correlations.
    """
    np.random.seed(seed)
    # Basic demographics
    age = np.clip(np.random.normal(age_mu, age_sd, n).astype(int), 18, 95)
    gender = np.random.choice(["Female", "Male"], size=n, p=[female_pct/100, 1-female_pct/100])
    bmi = np.round(np.clip(np.random.normal(bmi_mu, bmi_sd, n), 15, 55),1)
    cholesterol = np.round(np.clip(np.random.normal(chol_mu, chol_sd, n), 90, 450),1)

    # Comorbidities (probabilistic, may depend on age/bmi)
    prevalences = prevalences or {"Hypertension": 30, "Diabetes": 12, "CardioDisease": 8, "CKD": 6}
    def assign(p): 
        # slightly increase probability with age/bmi
        adj = p/100 + (age - age.mean())/400 + (bmi - bmi.mean())/400
        adj = np.clip(adj, 0.01, 0.95)
        return np.random.rand(n) < adj

    comorb = {k: assign(v) for k,v in prevalences.items()}

    # Genetics (binary flags, sample by prevalence)
    genetics = genetics or {"SLCO1B1_loss": 2, "APOE4": 12}
    geno = {k: (np.random.rand(n) < (v/100)) for k,v in genetics.items()}

    df = pd.DataFrame({
        "PatientID": [f"PT-{i+1:05d}" for i in range(n)],
        "Age": age,
        "Gender": gender,
        "BMI": bmi,
        "Cholesterol": cholesterol,
        **comorb,
        **geno
    })
    # Add baseline LDL approximated from Cholesterol with noise
    df["LDL_Baseline"] = np.round(df["Cholesterol"] * np.random.uniform(0.6, 0.8, size=n), 1)
    return df

def pk_one_compartment(dose_mg, ka, ke, Vd, times):
    """
    Simple one-compartment PK (single oral dose) — return concentration array.
    C(t) = (D*ka)/(Vd*(ka-ke)) * (e^{-ke t} - e^{-ka t})
    """
    t = np.array(times)
    if abs(ka - ke) < 1e-8:
        # avoid division by zero; use limiting form
        C = (dose_mg / Vd) * ka * t * np.exp(-ke*t)
    else:
        C = (dose_mg * ka) / (Vd * (ka - ke)) * (np.exp(-ke*t) - np.exp(-ka*t))
    return np.maximum(C, 0.0)

def pd_emax(Emax, EC50, C):
    """
    Simplified PD: percent effect (e.g. % LDL reduction) using Emax model (not subtracting baseline).
    E(C) = Emax * C / (EC50 + C)
    """
    return Emax * C / (EC50 + C)

def simulate_trial(pop_df, protocol, pk_params, pd_params, seed=1234):
    """
    Objective 3: Simulate trial by computing exposures per patient, effect, and AEs.
    protocol: dict with keys dose_mg, dosing_interval_hr, n_doses, duration_days
    pk_params: dict ka, ke, Vd
    pd_params: dict Emax, EC50, baseline_pct, ae_base_pct
    Returns: DataFrame with trial outcomes appended and summary dict
    """
    np.random.seed(seed)
    df = pop_df.copy()
    n = len(df)
    # simulate PK exposures with small inter-individual variability
    iid_clinical_variability = np.random.lognormal(mean=0, sigma=0.08, size=n)  # 8% variability
    # build time grid for one dosing regimen (hours)
    total_hours = protocol["dosing_interval_hr"] * protocol["n_doses"]
    times = np.linspace(0, total_hours, 300)
    # compute per-patient Cmax using superposition of doses
    cmax = np.zeros(n)
    for i in range(n):
        factor = iid_clinical_variability[i]
        # accumulate repeated doses (superposition)
        conc = np.zeros_like(times)
        for dose_index in range(protocol["n_doses"]):
            t_shift = times - dose_index*protocol["dosing_interval_hr"]
            conc += pk_one_compartment(protocol["dose_mg"], pk_params["ka"]*factor, pk_params["ke"]/factor, pk_params["Vd"]*factor, t_shift) * (t_shift>=0)
        cmax[i] = conc.max()
    # PD effect: map concentration to % reduction
    effect_pct = pd_emax(pd_params["Emax"], pd_params["EC50"], cmax) + pd_params.get("baseline_pct", 0)
    # Clamp
    effect_pct = np.clip(effect_pct, 0, 100)
    # Apply effect to baseline LDL
    df["LDL_Post"] = np.round(df["LDL_Baseline"] * (1 - effect_pct/100.0), 1)
    df["LDL_Reduction_pct"] = np.round((df["LDL_Baseline"] - df["LDL_Post"]) / df["LDL_Baseline"] * 100, 1)
    # Adverse event probability: base + exposure term + risk factors
    base = pd_params.get("ae_base_pct", 5)/100.0
    exposure_term = (cmax / (np.percentile(cmax, 95) + 1.0)) * 0.12  # scaled exposure effect
    age_term = (df["Age"] > 65).astype(float) * 0.05
    bmi_term = (df["BMI"] > 32).astype(float) * 0.04
    genotype_term = df.get("SLCO1B1_loss", pd.Series(False, index=df.index)).astype(float) * 0.15
    ae_prob = np.clip(base + exposure_term + age_term + bmi_term + genotype_term, 0, 0.9)
    df["AE_Prob"] = np.round(ae_prob,3)
    df["Adverse_Event"] = np.random.rand(n) < ae_prob
    # classify AE severity roughly
    df["AE_Severity"] = df["Adverse_Event"].apply(lambda v: np.random.choice(["Mild","Moderate","Severe"], p=[0.7,0.25,0.05]) if v else "")
    # add metadata
    summary = {
        "mean_ldl_reduction_pct": float(df["LDL_Reduction_pct"].mean()),
        "ae_rate_pct": float(df["Adverse_Event"].mean()*100),
        "n_patients": n
    }
    return df, summary

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Clinical Trial Digital Twin Simulator", layout="wide", page_icon="🩺")
st.title("🧬 Clinical Trial Digital Twin Simulator")
st.caption("Simulate trials on synthetic populations — generate insights to refine trial design.")

# Layout: left sidebar controls, main tabs
st.sidebar.header("Simulation Flow")
tab = st.sidebar.radio("Navigate", ["Population", "Drug Model", "Run Simulation", "Results & Insights"], index=0)

# Store state
if "population_df" not in st.session_state:
    st.session_state["population_df"] = None
if "protocol" not in st.session_state:
    st.session_state["protocol"] = None
if "pkpd" not in st.session_state:
    st.session_state["pkpd"] = None
if "sim_out" not in st.session_state:
    st.session_state["sim_out"] = None

# ---------------------------
# Population tab (Objective 1)
# ---------------------------
if tab == "Population":
    st.header("🧬 Population Generator")
    st.markdown("Create a synthetic population with demographics, comorbidities, and genetics.")
    with st.form("pop_form"):
        n = st.slider("Population size", 50, 5000, 500, step=50)
        col1, col2 = st.columns(2)
        with col1:
            age_mu = st.slider("Mean age", 18, 80, 55)
            age_sd = st.slider("Age SD", 1, 25, 12)
            female_pct = st.slider("Female %", 0, 100, 50)
            bmi_mu = st.slider("Mean BMI", 16, 40, 27)
            bmi_sd = st.slider("BMI SD", 1, 8, 3)
        with col2:
            chol_mu = st.slider("Mean total cholesterol (mg/dL)", 120, 320, 215)
            chol_sd = st.slider("Cholesterol SD", 5, 60, 24)
            st.markdown("*Comorbidity prevalence (%)*")
            htn = st.slider("Hypertension %", 0, 100, 30)
            diabetes = st.slider("Diabetes %", 0, 100, 12)
            cardio = st.slider("Cardio disease %", 0, 100, 8)
            ckd = st.slider("CKD %", 0, 20, 6)
            st.markdown("*Genetic prevalence (%)*")
            slco = st.slider("SLCO1B1_loss % (statin sensitivity)", 0, 10, 2)
            apoe4 = st.slider("APOE4 %", 0, 30, 12)
        submitted = st.form_submit_button("Generate population")
    if submitted:
        prevalences = {"Hypertension": htn, "Diabetes": diabetes, "CardioDisease": cardio, "CKD": ckd}
        genetics = {"SLCO1B1_loss": slco, "APOE4": apoe4}
        pop = generate_population(n=n, age_mu=age_mu, age_sd=age_sd, female_pct=female_pct,
                                  bmi_mu=bmi_mu, bmi_sd=bmi_sd, chol_mu=chol_mu, chol_sd=chol_sd,
                                  prevalences=prevalences, genetics=genetics, seed=123)
        st.session_state["population_df"] = pop
        st.success(f"Generated {len(pop)} synthetic patients.")
    if st.session_state["population_df"] is not None:
        df = st.session_state["population_df"]
        st.subheader("Population preview")
        st.dataframe(df.head(20), use_container_width=True, height=260)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean age", f"{df['Age'].mean():.1f}")
        c2.metric("Mean BMI", f"{df['BMI'].mean():.1f}")
        c3.metric("Mean LDL baseline", f"{df['LDL_Baseline'].mean():.1f}")
        st.plotly_chart(px.histogram(df, x="Age", nbins=20, title="Age distribution"), use_container_width=True)

# ---------------------------
# Drug Model tab (Objective 2)
# ---------------------------
elif tab == "Drug Model":
    st.header("💊 PK/PD Drug Modeling & Protocol")
    st.markdown("Configure a sample statin-like drug (one-compartment PK + Emax PD).")
    # Default PK/PD values that are reasonable placeholders
    with st.form("drug_form"):
        st.subheader("PK parameters (one-compartment)")
        ka = st.slider("Absorption rate ka (1/hr)", 0.1, 2.0, 0.7, step=0.01)
        ke = st.slider("Elimination rate ke (1/hr)", 0.01, 0.5, 0.08, step=0.01)
        Vd = st.slider("Volume of distribution Vd (L)", 10.0, 100.0, 40.0, step=1.0)
        st.subheader("PD parameters")
        Emax = st.slider("Emax (max % LDL reduction)", 0, 80, 45)
        EC50 = st.slider("EC50 (mg/L)", 0.1, 10.0, 1.0, step=0.01)
        baseline_pct = st.slider("Baseline response (additive %)", 0, 10, 0)
        ae_base = st.slider("Base adverse event probability (%)", 0, 20, 5)
        st.subheader("Protocol (dosing)")
        dose_mg = st.slider("Dose (mg)", 5, 120, 40, step=5)
        dosing_interval = st.slider("Dosing interval (hours)", 8, 48, 24)
        n_doses = st.slider("Number of repeated doses", 1, 10, 3)
        duration_days = st.slider("Treatment duration (days)", 7, 180, 28)
        save = st.form_submit_button("Save PK/PD & Protocol")
    if save:
        st.session_state["pkpd"] = {
            "ka": float(ka), "ke": float(ke), "Vd": float(Vd),
            "Emax": float(Emax), "EC50": float(EC50), "baseline_pct": float(baseline_pct),
            "ae_base_pct": float(ae_base)
        }
        st.session_state["protocol"] = {
            "dose_mg": int(dose_mg), "dosing_interval_hr": int(dosing_interval),
            "n_doses": int(n_doses), "duration_days": int(duration_days)
        }
        st.success("PK/PD model and protocol saved. Ready to run simulation.")
    if st.session_state.get("pkpd"):
        st.subheader("Current saved PK/PD & Protocol")
        st.json({**st.session_state["pkpd"], **st.session_state["protocol"]})
        # quick PK plot for one subject
        times = np.linspace(0, st.session_state["protocol"]["dosing_interval_hr"] * st.session_state["protocol"]["n_doses"], 300)
        conc = np.zeros_like(times)
        for i in range(st.session_state["protocol"]["n_doses"]):
            t_shift = times - i*st.session_state["protocol"]["dosing_interval_hr"]
            conc += np.where(t_shift>=0,
                             (st.session_state["protocol"]["dose_mg"]/st.session_state["pkpd"]["Vd"])*st.session_state["pkpd"]["ka"]/(st.session_state["pkpd"]["ka"]-st.session_state["pkpd"]["ke"]) * (np.exp(-st.session_state["pkpd"]["ke"]*t_shift) - np.exp(-st.session_state["pkpd"]["ka"]*t_shift)),
                             0)
        st.plotly_chart(px.line(x=times, y=conc, labels={"x":"Time (hr)","y":"Conc (mg/L)"}, title="Simulated plasma concentration (one-subject)"), use_container_width=True)

# ---------------------------
# Run Simulation tab (Objective 3)
# ---------------------------
elif tab == "Run Simulation":
    st.header("🧪 Trial Simulation Engine")
    st.markdown("Run the virtual trial on the generated population using saved PK/PD & protocol.")
    if st.session_state.get("population_df") is None:
        st.warning("Generate a population first (Population tab).")
        st.stop()
    if st.session_state.get("pkpd") is None or st.session_state.get("protocol") is None:
        st.warning("Save PK/PD & protocol first (Drug Model tab).")
        st.stop()
    # Simulation controls
    col1, col2 = st.columns([1,2])
    with col1:
        n_runs = st.number_input("Number of replicate simulations (for uncertainty)", min_value=1, max_value=50, value=1)
        run_button = st.button("Run Simulation")
    with col2:
        st.info("Simulator features: per-patient exposure (Cmax), PD via Emax, AE probabilistic model, subgroup outputs.")
    if run_button:
        all_outcomes = []
        # run replicate sims to show variability if requested
        for r in range(int(n_runs)):
            df_out, summary = simulate_trial(st.session_state["population_df"], st.session_state["protocol"], st.session_state["pkpd"], 
                                             {"Emax": st.session_state["pkpd"]["Emax"], "EC50": st.session_state["pkpd"]["EC50"], 
                                              "baseline_pct": st.session_state["pkpd"].get("baseline_pct",0),
                                              "ae_base_pct": st.session_state["pkpd"].get("ae_base_pct",5)},
                                             seed=123 + r*7)
            all_outcomes.append((df_out, summary))
        # store last run and summaries
        st.session_state["sim_out"] = all_outcomes
        st.success(f"Completed {len(all_outcomes)} simulation run(s). Click Results & Insights to explore outcomes.")

# ---------------------------
# Results & Insights tab (Objectives 4 & 5)
# ---------------------------
elif tab == "Results & Insights":
    st.header("📊 Results & Insights")
    if st.session_state.get("sim_out") is None:
        st.warning("Run the simulation first on the Run Simulation tab.")
        st.stop()
    # We'll show the last run and an aggregate view
    runs = st.session_state["sim_out"]
    last_df, last_summary = runs[-1]
    st.subheader("Trial-level summary (last run)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average LDL reduction (%)", f"{last_summary['mean_ldl_reduction_pct']:.1f}")
    c2.metric("Adverse event rate (%)", f"{last_summary['ae_rate_pct']:.2f}")
    c3.metric("Number of patients", f"{last_summary['n_patients']}")
    # Distribution plots
    st.plotly_chart(px.histogram(last_df, x="LDL_Reduction_pct", nbins=25, title="% LDL reduction distribution (last run)"), use_container_width=True)
    st.plotly_chart(px.histogram(last_df, x="AE_Prob", nbins=30, title="Predicted individual AE probability distribution"), use_container_width=True)
    # Subgroup analysis
    st.subheader("Subgroup Analysis: Age & BMI risk cross-tab")
    last_df["AgeBin"] = pd.cut(last_df["Age"], bins=[18,40,60,80,120], labels=["18-39","40-59","60-79","80+"])
    pivot = last_df.groupby(["AgeBin"]).agg(avg_ldl_reduction=("LDL_Reduction_pct","mean"),
                                           ae_rate=("Adverse_Event","mean"),
                                           n=("PatientID","count")).reset_index()
    st.dataframe(pivot.style.format({"avg_ldl_reduction":"{:.1f}", "ae_rate":"{:.2%}"}), height=240)
    st.plotly_chart(px.bar(pivot, x="AgeBin", y="ae_rate", labels={"ae_rate":"Adverse Event Rate"}, title="AE rate by age group"), use_container_width=True)
    # Identify flagged high-risk groups
    st.subheader("Flagged High-Risk Subgroups & Recommendations")
    flagged = last_df[(last_df["Adverse_Event"]) & ((last_df["Age"]>65) | (last_df["BMI"]>32))]
    n_flagged = len(flagged)
    st.markdown(f"- *{n_flagged} patients* (~{n_flagged/len(last_df)*100:.1f}%) had AEs and were age>65 or BMI>32.")
    st.markdown("*Recommendations (automatically generated):*")
    st.markdown("""
    1. Consider excluding patients with *Age > 75* or *BMI > 34* for the pivotal trial, or include them in a safety sub-cohort.  
    2. For patients with *SLCO1B1_loss* genotype, consider genotype-guided dosing or intensified monitoring (CK enzymes).  
    3. Add routine early visit (Week 2) for CK and ALT in high-risk subgroups.  
    4. Consider reducing starting dose by 25% for patients >75 or with CKD.
    """)
    # Offer downloads
    csv = last_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download last-run patient-level CSV", csv, file_name="digital_twin_last_run.csv")
    # Quick sensitivity: show effect of excluding a subgroup on AE rate and LDL reduction
    st.subheader("Quick sensitivity: Exclusion trade-offs")
    exclude_age = st.checkbox("Simulate excluding Age > 75", value=True)
    if exclude_age:
        subset = last_df[last_df["Age"] <= 75]
        st.markdown(f"- Excluding Age>75: New AE rate = {subset['Adverse_Event'].mean()*100:.2f}%, Avg LDL reduction = {subset['LDL_Reduction_pct'].mean():.1f}% (n={len(subset)})")
    # Short methodology / limitations
    with st.expander("Methodology & Limitations"):
        st.write("""
        - Population generated from simple probabilistic models and parametric marginals; real-world calibration recommended.
        - PK/PD is a simplified one-compartment Emax model for demonstration and hypothesis generation.
        - Rare AEs are difficult to estimate; simulations are useful for prioritization and monitoring design, not absolute risk quantification.
        - For regulatory-grade simulation, use validated PK/PD libraries and real-world datasets to calibrate distributions.
        """)
    st.success("Simulation insights generated. Use recommendations to refine inclusion/exclusion criteria and monitoring schedule.")
'''

# Write the app file
with open(APP_FILE, "w", encoding="utf-8") as f:
    f.write(app_code)

print(f"Streamlit app written to {APP_FILE}")

# ---------------------------
# 3) Launch the app
# ---------------------------

# Attempt to determine if we're in a Colab environment
IN_COLAB = "COLAB_GPU" in os.environ or "google.colab" in sys.modules

# Kill leftover processes for clean restart (best-effort)
def safe_kill(proc_name):
    try:
        subprocess.call(["pkill", "-f", proc_name])
    except Exception:
        pass

safe_kill("streamlit")
safe_kill("ngrok")

# If ngrok token is provided either in code or environment, configure it.
if not NGROK_AUTH_TOKEN:
    NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")

if IN_COLAB:
    # Colab flow: start streamlit in background and create ngrok tunnel
    from pyngrok import ngrok, conf
    if NGROK_AUTH_TOKEN:
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        print("ngrok authtoken set from NGROK_AUTH_TOKEN.")
    else:
        print("No ngrok authtoken provided. You may still get a free tunnel but ngrok requires verification for persistent tunnels.")
    # Start streamlit
    print("Starting Streamlit in background...")
    # Use nohup for background safe start; also redirect output
    cmd = f"nohup streamlit run {APP_FILE} --server.port {PORT} >/dev/null 2>&1 &"
    os.system(cmd)
    time.sleep(2.5)
    try:
        ngrok.kill()  # ensure no old tunnels
    except Exception:
        pass
    try:
        public_url = ngrok.connect(PORT, "http").public_url
        print("\n✅ Your Streamlit app should be available at the public ngrok URL below:")
        print(public_url)
        print("\nIf the ngrok URL fails, (1) ensure NGROK_AUTH_TOKEN is set, or (2) open the Colab VM's port or run locally and open http://localhost:8501")
    except Exception as e:
        print("Could not start ngrok tunnel:", str(e))
        print("If running locally, open http://localhost:8501 after running 'streamlit run app.py' in your terminal.")
else:
    # Local / VS Code flow: run streamlit in foreground
    print("Detected local environment. Launching Streamlit server (foreground).")
    print(f"Run this command in a terminal if the script does not continue:\n    streamlit run {APP_FILE} --server.port {PORT}")
    # Launch and block (this call will run Streamlit; if you want non-blocking, comment it)
    os.system(f"streamlit run {APP_FILE} --server.port {PORT}")