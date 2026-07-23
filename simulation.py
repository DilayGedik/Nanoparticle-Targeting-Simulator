from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TargetingConfig:
    width_um: float = 1200.0
    height_um: float = 800.0
    particle_count: int = 220
    diffusion_um2_s: float = 4.5
    drift_velocity_um_s: float = 5.0
    tumor_center_x_um: float = 850.0
    tumor_center_y_um: float = 400.0
    tumor_radius_um: float = 170.0
    receptor_density_relative: float = 1.0
    kon_s: float = 0.40
    koff_s: float = 0.035
    internalization_rate_s: float = 0.06
    healthy_binding_scale: float = 0.08
    clearance_rate_s: float = 0.003
    duration_s: float = 220.0
    dt_s: float = 0.25
    seed: int = 11

def inside_tumor(x, y, c):
    return (x-c.tumor_center_x_um)**2 + (y-c.tumor_center_y_um)**2 <= c.tumor_radius_um**2

def reflect(v, upper):
    v = np.where(v < 0, -v, v)
    v = np.where(v > upper, 2*upper-v, v)
    return np.clip(v, 0, upper)

def simulate_targeting(c):
    rng = np.random.default_rng(c.seed)
    steps = int(np.ceil(c.duration_s/c.dt_s)) + 1
    times = np.arange(steps) * c.dt_s
    pos = np.zeros((steps, c.particle_count, 2), dtype=np.float32)
    state = np.zeros((steps, c.particle_count), dtype=np.int8)
    pos[0,:,0] = rng.uniform(40,120,c.particle_count)
    pos[0,:,1] = rng.uniform(0.15*c.height_um,0.85*c.height_um,c.particle_count)
    sigma = np.sqrt(2*c.diffusion_um2_s*c.dt_s)

    for k in range(1, steps):
        pos[k] = pos[k-1]
        state[k] = state[k-1]
        free = state[k-1] == 0
        bound = state[k-1] == 1

        if free.any():
            idx = np.where(free)[0]
            motion = rng.normal(0,sigma,size=(len(idx),2))
            motion[:,0] += c.drift_velocity_um_s*c.dt_s
            pos[k,idx] += motion
            pos[k,idx,0] = reflect(pos[k,idx,0],c.width_um)
            pos[k,idx,1] = reflect(pos[k,idx,1],c.height_um)

            in_tumor = inside_tumor(pos[k,idx,0],pos[k,idx,1],c)
            local_rate = np.where(in_tumor,c.kon_s*c.receptor_density_relative,
                                  c.kon_s*c.healthy_binding_scale)
            bind = rng.random(len(idx)) < (1-np.exp(-local_rate*c.dt_s))
            state[k,idx[bind]] = 1

            clear = rng.random(len(idx)) < (1-np.exp(-c.clearance_rate_s*c.dt_s))
            clear_idx = idx[clear & ~bind]
            state[k,clear_idx] = 3

        if bound.any():
            idx = np.where(bound)[0]
            internalize = rng.random(len(idx)) < (1-np.exp(-c.internalization_rate_s*c.dt_s))
            state[k,idx[internalize]] = 2
            remain = ~internalize
            if remain.any():
                rem_idx = idx[remain]
                unbind = rng.random(len(rem_idx)) < (1-np.exp(-c.koff_s*c.dt_s))
                state[k,rem_idx[unbind]] = 0

    rows = []
    for i,t in enumerate(times):
        st = state[i]
        tumor = inside_tumor(pos[i,:,0],pos[i,:,1],c)
        rows.append({
            "time_s": t,
            "free": int((st==0).sum()),
            "bound": int((st==1).sum()),
            "internalized": int((st==2).sum()),
            "cleared": int((st==3).sum()),
            "particles_in_tumor": int((tumor & (st!=3)).sum()),
            "tumor_internalized": int((tumor & (st==2)).sum())
        })
    return {"times_s":times,"positions_um":pos,"states":state,"metrics":pd.DataFrame(rows)}

def trajectories_to_dataframe(result):
    p = result["positions_um"]
    s = result["states"]
    t = result["times_s"]
    nsteps,npart,_ = p.shape
    labels = np.array(["free","bound","internalized","cleared"])
    return pd.DataFrame({
        "time_s": np.repeat(t,npart),
        "particle_id": np.tile(np.arange(npart),nsteps),
        "x_um": p[:,:,0].reshape(-1),
        "y_um": p[:,:,1].reshape(-1),
        "state": labels[s.reshape(-1)]
    })

def final_summary(result):
    f = result["metrics"].iloc[-1]
    total = float(f["free"]+f["bound"]+f["internalized"]+f["cleared"])
    return {
        "targeting_efficiency_percent": 100*float(f["tumor_internalized"])/max(total,1),
        "internalization_percent": 100*float(f["internalized"])/max(total,1),
        "clearance_percent": 100*float(f["cleared"])/max(total,1),
        "tumor_presence_percent": 100*float(f["particles_in_tumor"])/max(total,1)
    }
