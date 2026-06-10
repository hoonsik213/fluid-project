import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Photoresist Jet Simulator", layout="wide")

st.title("Photoresist Dispensing Jet Simulator")
st.write("Simple Rayleigh-Plateau instability model for a photoresist liquid jet.")

# -----------------------------
# Input section
# -----------------------------
st.sidebar.header("Input values")

radius_mm = st.sidebar.slider("Nozzle radius r0 (mm)", 0.1, 2.0, 0.5, 0.1)
velocity = st.sidebar.slider("Jet velocity U (m/s)", 0.5, 10.0, 3.0, 0.5)
viscosity = st.sidebar.slider("Viscosity μ (Pa·s)", 0.001, 0.100, 0.020, 0.001)
surface_tension = st.sidebar.slider("Surface tension γ (N/m)", 0.010, 0.080, 0.030, 0.005)
density = st.sidebar.slider("Density ρ (kg/m³)", 800, 1300, 1000, 50)
distance_cm = st.sidebar.slider("Distance to wafer L (cm)", 1.0, 20.0, 5.0, 0.5)

# Unit conversion
r0 = radius_mm / 1000      # mm to m
D = 2 * r0
L = distance_cm / 100      # cm to m

# -----------------------------
# Basic dimensionless numbers
# -----------------------------
Re = density * velocity * D / viscosity
Ca = viscosity * velocity / surface_tension
We = density * velocity**2 * D / surface_tension

# -----------------------------
# Rayleigh-Plateau calculation
# -----------------------------
# x = k*r0
x = np.linspace(0.01, 1.4, 300)

# Inviscid approximation of growth rate
# omega^2 = gamma/(rho*r0^3) * x^2 * (1-x^2)
omega_sq = (surface_tension / (density * r0**3)) * (x**2) * (1 - x**2)

# negative part is stable, so set to zero for plotting growth rate
omega = np.sqrt(np.maximum(omega_sq, 0))

max_index = np.argmax(omega)
x_max = x[max_index]
omega_max = omega[max_index]

# wavelength = 2*pi/k = 2*pi*r0/x
most_unstable_wavelength = 2 * np.pi * r0 / x_max

# simple breakup time estimation
# small initial disturbance assumed
initial_disturbance = 0.01 * r0
target_disturbance = r0

if omega_max > 0:
    breakup_time = np.log(target_disturbance / initial_disturbance) / omega_max
    breakup_distance = velocity * breakup_time
else:
    breakup_time = np.inf
    breakup_distance = np.inf

time_to_wafer = L / velocity

# -----------------------------
# Results
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Reynolds number", f"{Re:.1f}")
col2.metric("Capillary number", f"{Ca:.3f}")
col3.metric("Weber number", f"{We:.1f}")

col4, col5, col6 = st.columns(3)

col4.metric("Max growth rate", f"{omega_max:.2f} 1/s")
col5.metric("Most unstable wavelength", f"{most_unstable_wavelength*1000:.2f} mm")

if np.isfinite(breakup_distance):
    col6.metric("Estimated breakup distance", f"{breakup_distance*100:.2f} cm")
else:
    col6.metric("Estimated breakup distance", "Stable")

st.divider()

# -----------------------------
# Decision message
# -----------------------------
if breakup_distance < L:
    st.error("Prediction: The jet may break up before reaching the wafer.")
else:
    st.success("Prediction: The jet is expected to reach the wafer before breakup.")

st.write(f"Time to reach wafer: **{time_to_wafer:.4f} s**")
st.write(f"Estimated breakup time: **{breakup_time:.4f} s**")

# -----------------------------
# Graph 1: growth rate spectrum
# -----------------------------
st.subheader("Growth-rate spectrum")

fig, ax = plt.subplots()
ax.plot(x, omega)
ax.axvline(x_max, linestyle="--", label="Most unstable mode")
ax.set_xlabel("Dimensionless wavenumber, k r0")
ax.set_ylabel("Growth rate, omega (1/s)")
ax.set_title("Rayleigh-Plateau growth rate")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Graph 2: simple jet shape visualization
# -----------------------------
st.subheader("Simple jet disturbance shape")

z = np.linspace(0, L, 400)
k_max = x_max / r0

# disturbance grows with distance using t = z/U
amp = initial_disturbance * np.exp(omega_max * z / velocity)
amp = np.minimum(amp, r0)

jet_radius = r0 + amp * np.sin(k_max * z)

fig2, ax2 = plt.subplots()
ax2.plot(z * 100, jet_radius * 1000, label="Upper radius")
ax2.plot(z * 100, -jet_radius * 1000, label="Lower radius")
ax2.set_xlabel("Distance from nozzle (cm)")
ax2.set_ylabel("Jet radius (mm)")
ax2.set_title("Simplified liquid jet disturbance")
ax2.grid(True)
ax2.legend()

st.pyplot(fig2)

# -----------------------------
# Short explanation
# -----------------------------
st.subheader("Model explanation")

st.write("""
This simulator uses a simplified Rayleigh-Plateau instability model.
A cylindrical liquid jet becomes unstable when the disturbance wavelength is larger than the jet circumference scale.
The growth-rate equation is used to find the most unstable mode and estimate the breakup distance.

In this simplified version, the viscosity effect is mainly reflected through dimensionless numbers such as Re and Ca.
The result is intended for process-level understanding, not exact industrial prediction.
""")
st.divider()

st.subheader("Validation View")

if viscosity > 0.03:
    st.success(
        "Validation: Higher viscosity increases jet stability and delays breakup. "
        "This trend agrees with Rayleigh-Plateau theory."
    )
else:
    st.info(
        "Validation: Lower viscosity allows disturbances to grow faster, "
        "which can lead to earlier breakup."
    )

st.divider()

st.subheader("Design Recommendation")

if breakup_distance > L:
    st.success(
        "Recommended operating condition: Current settings are suitable "
        "for delivering the jet to the wafer."
    )
else:
    st.warning(
        "Recommendation: Increase viscosity, reduce nozzle radius, "
        "or increase surface tension to improve stability."
    )
