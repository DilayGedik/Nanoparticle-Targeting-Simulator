import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import streamlit as st
from simulation import TargetingConfig, simulate_targeting, trajectories_to_dataframe, final_summary

st.set_page_config(page_title="Nanoparticle Tumor Targeting",page_icon="🎯",layout="wide")
st.title("Nanoparticle Tumor Targeting")
st.caption("Interactive stochastic model of transport, receptor binding, internalization, and clearance.")

with st.sidebar:
    st.header("Particles")
    count = st.slider("Particle count",50,600,220,10)
    diffusion = st.slider("Diffusion coefficient (µm²/s)",0.5,20.0,4.5,0.5)
    drift = st.slider("Convective drift (µm/s)",0.0,20.0,5.0,0.5)
    st.header("Targeting")
    radius = st.slider("Tumor radius (µm)",80.0,260.0,170.0,5.0)
    receptors = st.slider("Relative receptor density",0.1,3.0,1.0,0.1)
    kon = st.slider("Binding rate k_on (1/s)",0.01,1.00,0.40,0.01)
    koff = st.slider("Unbinding rate k_off (1/s)",0.0,0.20,0.035,0.005)
    internal = st.slider("Internalization rate (1/s)",0.0,0.20,0.060,0.005)
    healthy = st.slider("Healthy binding scale",0.0,0.50,0.08,0.01)
    clearance = st.slider("Clearance rate (1/s)",0.0,0.020,0.003,0.001)
    duration = st.slider("Duration (s)",40.0,500.0,220.0,10.0)
    run = st.button("Run simulation",type="primary",use_container_width=True)

config = TargetingConfig(particle_count=count,diffusion_um2_s=diffusion,
    drift_velocity_um_s=drift,tumor_radius_um=radius,
    receptor_density_relative=receptors,kon_s=kon,koff_s=koff,
    internalization_rate_s=internal,healthy_binding_scale=healthy,
    clearance_rate_s=clearance,duration_s=duration)

if run or "target_result" not in st.session_state:
    with st.spinner("Simulating particle transport and binding..."):
        st.session_state.target_result = simulate_targeting(config)
        st.session_state.target_config = config

result = st.session_state.target_result
active = st.session_state.target_config
metrics = result["metrics"]
summary = final_summary(result)

if active != config:
    st.info("Parameters changed. Click Run simulation to update.")

frame = st.slider("Time point",0,len(result["times_s"])-1,len(result["times_s"])-1)
t = float(result["times_s"][frame])

a,b,c,d = st.columns(4)
a.metric("Time",f"{t:.1f} s")
b.metric("Tumor targeting efficiency",f"{summary['targeting_efficiency_percent']:.1f}%")
c.metric("Internalized",int(metrics['internalized'].iloc[-1]))
d.metric("Cleared",int(metrics['cleared'].iloc[-1]))

tabs = st.tabs(["Particle map","Trajectories","State kinetics","Binding analysis","Export & assumptions"])
names = ["Free","Bound","Internalized","Cleared"]

with tabs[0]:
    fig,ax = plt.subplots(figsize=(11,7))
    p = result["positions_um"][frame]
    s = result["states"][frame]
    for code,name in enumerate(names):
        mask = s==code
        if mask.any():
            ax.scatter(p[mask,0],p[mask,1],s=24,alpha=.75,label=name)
    ax.add_patch(Circle((active.tumor_center_x_um,active.tumor_center_y_um),
        active.tumor_radius_um,fill=False,linewidth=2.5))
    ax.set(xlim=(0,active.width_um),ylim=(0,active.height_um),
        xlabel="x (µm)",ylabel="y (µm)",title=f"Nanoparticle states at t = {t:.1f} s")
    ax.set_aspect("equal"); ax.legend()
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    if st.button("Generate targeting GIF"):
        fig,ax = plt.subplots(figsize=(9,6))
        ax.set_xlim(0,active.width_um); ax.set_ylim(0,active.height_um); ax.set_aspect("equal")
        ax.add_patch(Circle((active.tumor_center_x_um,active.tumor_center_y_um),
            active.tumor_radius_um,fill=False,linewidth=2))
        scatters=[ax.scatter([],[],s=24,label=n) for n in names]
        ax.legend()
        def update(i):
            pp=result["positions_um"][i]; ss=result["states"][i]
            for code,sc in enumerate(scatters):
                sc.set_offsets(pp[ss==code])
            ax.set_title(f"Nanoparticle targeting: t = {result['times_s'][i]:.1f} s")
            return scatters
        stride=max(1,len(result["times_s"])//120)
        ani=FuncAnimation(fig,update,frames=range(0,len(result["times_s"]),stride),interval=70)
        path="nanoparticle_targeting.gif"
        ani.save(path,writer=PillowWriter(fps=12))
        plt.close(fig)
        data=open(path,"rb").read()
        st.image(data)
        st.download_button("Download GIF",data=data,file_name=path,mime="image/gif")

with tabs[1]:
    fig,ax = plt.subplots(figsize=(11,7))
    stride=max(1,len(result["times_s"])//180)
    for pid in range(min(active.particle_count,100)):
        xy=result["positions_um"][::stride,pid]
        ax.plot(xy[:,0],xy[:,1],linewidth=.7,alpha=.45)
    ax.add_patch(Circle((active.tumor_center_x_um,active.tumor_center_y_um),
        active.tumor_radius_um,fill=False,linewidth=2.5))
    ax.set(xlim=(0,active.width_um),ylim=(0,active.height_um),
        xlabel="x (µm)",ylabel="y (µm)",title="Representative nanoparticle trajectories")
    ax.set_aspect("equal")
    st.pyplot(fig,use_container_width=True); plt.close(fig)

with tabs[2]:
    st.line_chart(metrics.set_index("time_s")[["free","bound","internalized","cleared"]])
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(metrics["time_s"],metrics["particles_in_tumor"],label="Particles in tumor")
    ax.plot(metrics["time_s"],metrics["tumor_internalized"],label="Tumor-internalized")
    ax.set(title="Tumor accumulation over time",xlabel="Time (s)",ylabel="Particle count")
    ax.grid(True,alpha=.25); ax.legend()
    st.pyplot(fig,use_container_width=True); plt.close(fig)

with tabs[3]:
    f=metrics.iloc[-1]
    chart=pd.DataFrame({"State":names,"Particles":[int(f["free"]),int(f["bound"]),int(f["internalized"]),int(f["cleared"])]})
    st.bar_chart(chart.set_index("State"))
    x,y,z=st.columns(3)
    x.metric("Particles in tumor",int(f["particles_in_tumor"]))
    y.metric("Tumor-internalized",int(f["tumor_internalized"]))
    z.metric("Overall internalization",f"{summary['internalization_percent']:.1f}%")

with tabs[4]:
    st.dataframe(metrics,use_container_width=True)
    st.download_button("Download metrics CSV",metrics.to_csv(index=False).encode(),
        "nanoparticle_targeting_metrics.csv","text/csv")
    traj=trajectories_to_dataframe(result)
    st.download_button("Download trajectories CSV",traj.to_csv(index=False).encode(),
        "nanoparticle_targeting_trajectories.csv","text/csv")
    st.markdown('''### Assumptions
- Homogeneous 2D domain
- Brownian diffusion plus constant drift
- Higher tumor receptor density increases binding
- Bound particles can unbind or internalize
- Free particles can be cleared
- No receptor saturation, immune uptake, or vascular extravasation

This is a portfolio and early-design model, not a clinical prediction tool.''')
