import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.base import Controller, Action
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim

st.set_page_config(
    page_title="Type 1 Diabetes Simulation Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS to hide the sidebar collapse button
st.markdown(
    """
    <style>
    [data-testid="collapsedControl"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩸 Type 1 Diabetes Virtual Simulator")
st.markdown("Adjust the hyperparameters and select patients below to view predictive glucose curves using **simglucose**.")

class DelayedInjectionController(Controller):
    def __init__(self, init_state, injection_hour, bolus_units):
        self.init_state = init_state
        self.state = init_state
        self.step_count = 0
        self.injection_hour = injection_hour
        self.bolus_units = bolus_units

    def policy(self, observation, reward, done, **info):
        self.state = observation
        current_time_hours = (self.step_count * 3) / 60.0
        basal_rate = 0
        bolus = 0
        if abs(current_time_hours - self.injection_hour) < 0.01:
            bolus = self.bolus_units
        action = Action(basal=basal_rate, bolus=bolus)
        self.step_count += 1
        return action

    def reset(self):
        self.state = self.init_state
        self.step_count = 0

# --- Interactive Sidebar ---
st.sidebar.markdown("### 🎯 Simulation Parameters")

# Group patient settings side-by-side
col1, col2 = st.sidebar.columns(2)
patient_group = col1.selectbox("Cohort:", ["Adolescents", "Adults", "Children"])
patient_num = col2.number_input("Profile #:", 1, 10, 2)
patient_prefix = {"Adolescents": "adolescent", "Adults": "adult", "Children": "child"}[patient_group]
PATIENT_ID = f"{patient_prefix}#{int(patient_num):03d}"

# Dynamically display the weight of the selected patient in the sidebar
try:
    selected_patient = T1DPatient.withName(PATIENT_ID)
    patient_weight = selected_patient._params.BW
    st.sidebar.info(f"**⚖️ Patient Weight:** {patient_weight:.1f} kg")
except Exception as e:
    st.sidebar.warning("Could not load patient weight.")

INITIAL_BG = st.sidebar.number_input("Target Initial BG (mg/dL):", min_value=50, max_value=500, value=458)

st.sidebar.markdown("#### 🍔 Meal Event")
m_col1, m_col2 = st.sidebar.columns(2)
MEAL_HOUR = m_col1.number_input("Time (hr):", 0.0, 8.0, 6.0, step=0.5)
MEAL_CARBS = m_col2.number_input("Carbs (g):", 0, 150, 60)

st.sidebar.markdown("#### 💉 Insulin Bolus")
i_col1, i_col2 = st.sidebar.columns(2)
INJECTION_HOUR = i_col1.number_input("Time (hr):", 0.0, 8.0, 0.75, step=0.25)
INJECTION_UNITS = i_col2.number_input("Dose (U):", 0, 30, 7)

st.sidebar.markdown("---")
if st.sidebar.button("⚡ Run Virtual Simulation", type="primary", use_container_width=True):
    with st.spinner("Calculating patient metabolic trace..."):
        try:
            start_time = datetime(2026, 1, 1, 0, 0, 0)
            patient = T1DPatient.withName(PATIENT_ID)
            original_weight = patient._params.BW

            col1, col2 = st.columns(2)
            col1.metric("Selected Cohort Profile", PATIENT_ID.upper())
            col2.metric("Patient Weight", f"{original_weight:.2f} kg")

            custom_init_state = patient.init_state.copy()
            current_initial_bg = custom_init_state[3] / patient._params.Vg
            bg_ratio = INITIAL_BG / current_initial_bg

            custom_init_state[3] *= bg_ratio
            custom_init_state[4] *= bg_ratio
            custom_init_state[12] *= bg_ratio
            patient._init_state = custom_init_state

            sensor = CGMSensor.withName('Dexcom', seed=1)
            pump = InsulinPump.withName('Insulet')

            scenario_events = []
            if MEAL_CARBS > 0:
                scenario_events.append((MEAL_HOUR, MEAL_CARBS))

            scenario = CustomScenario(start_time=start_time, scenario=scenario_events)
            env = T1DSimEnv(patient, sensor, pump, scenario)

            controller = DelayedInjectionController(
                init_state=0,
                injection_hour=INJECTION_HOUR,
                bolus_units=INJECTION_UNITS
            )

            s1 = SimObj(env, controller, timedelta(hours=8), animate=False, path='./results')
            results = sim(s1)
            df = pd.concat([results])

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axhspan(70, 180, color='green', alpha=0.15, label='Target Range (70-180 mg/dL)')
            ax.axhspan(0, 70, color='red', alpha=0.1, label='Hypoglycemia (<70 mg/dL)')
            upper_limit = max(350, df['BG'].max() + 20)
            ax.axhspan(180, upper_limit, color='orange', alpha=0.1, label='Hyperglycemia (>180 mg/dL)')

            ax.plot(df.index, df['BG'], label='Blood Glucose Trace', color='black', linewidth=3)

            meal_time = start_time + timedelta(hours=MEAL_HOUR)
            inject_time = start_time + timedelta(hours=INJECTION_HOUR)

            if MEAL_CARBS > 0:
                ax.axvline(x=meal_time, color='orange', linestyle='--', linewidth=2, label=f'Food Intake ({MEAL_CARBS}g at {MEAL_HOUR}h)')
            ax.axvline(x=inject_time, color='purple', linestyle=':', linewidth=2, label=f'Insulin Dose ({INJECTION_UNITS}U at {INJECTION_HOUR}h)')

            ax.set_xlabel('Simulation Time Elapsed')
            ax.set_ylabel('Blood Glucose (mg/dL)')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle=':', alpha=0.6)

            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
        except Exception as e:
            st.error(f"Simulation error: {str(e)}")
else:
    st.info("👈 Choose parameters in the sidebar panel and click 'Run Virtual Simulation' to launch visualization data.")
