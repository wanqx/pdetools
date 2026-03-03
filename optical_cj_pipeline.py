"""Frozen optical-to-CJ-pressure pipeline.

This module freezes the current implementation used in `plotlcq.ipynb`:
Areal_2light (channels 0/1) -> TOF velocity -> M_D -> optical-only CJ pressure,
with Areal_4highf_p plateau used for validation only.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def calc_detonation_velocity_cfd(
    time,
    light,
    sensor_distance=0.1,
    valid_cols=(0, 1),
    expected_speed=2000.0,
    speed_bounds=(1000.0, 4000.0),
    cfd_fraction=0.1,
    smooth_window_us=20.0,
    pre_window_us=80.0,
    post_window_us=200.0,
    min_event_gap_us=300.0,
    detect_merge_gap_us=300.0,
    coarse_delay_margin_us=20.0,
    downstream_fallback_search=True,
    deriv_sigma=6.0,
    min_snr=8.0,
    apply_velocity_filter=True,
    expected_speed_tolerance=0.35,
    hampel_window=2,
    hampel_nsigma=2.0,
    return_debug=False,
):
    """
    Compute detonation-wave velocity from two optical channels using
    a robust timing method: derivative trigger + CFD (constant-fraction discrimination).

    Why this method:
    - Peak-time is biased when two sensors have different response shapes.
    - CFD timing is widely used in time-of-flight diagnostics because timing is less
      sensitive to amplitude/pulse-shape differences.

    Parameters
    ----------
    time : (N,) array
        Time axis in seconds.
    light : (N, M) array
        Optical intensity matrix.
    sensor_distance : float
        Distance between the two sensors, in meters. (100 mm -> 0.1 m)
    valid_cols : tuple
        Column indices of upstream/downstream sensors in `light`.
    expected_speed : float
        Expected detonation speed for matching, m/s. (about 2000 m/s here)
    speed_bounds : tuple
        Allowed speed range (v_min, v_max), m/s.
    cfd_fraction : float
        Constant-fraction level in [0, 1], recommended 0.05~0.3 for leading-edge timing.
    smooth_window_us : float
        Moving-average smoothing window, microseconds.
    pre_window_us/post_window_us : float
        Windows around trigger for local baseline/peak search, microseconds.
    min_event_gap_us : float
        Minimum gap used for final event de-duplication, microseconds.
    detect_merge_gap_us : float
        Merge gap for coarse candidate clustering inside one channel, microseconds.
    coarse_delay_margin_us : float
        Extra margin for coarse candidate pairing window, microseconds.
        Final validity is still checked by CFD delay bounds.
    downstream_fallback_search : bool
        If True, when no downstream coarse candidate is found in the pairing window,
        perform local derivative-based fallback search in that window.
    deriv_sigma : float
        Robust derivative threshold factor (median + k * MAD).
    min_snr : float
        Minimum local SNR required for CFD timing.
    apply_velocity_filter : bool
        If True, apply robust outlier filtering to per-event velocity.
    expected_speed_tolerance : float or None
        Relative tolerance around expected_speed used as an outlier gate.
        Example: 0.35 means keep around expected_speed +/-35%.
    hampel_window : int
        Half-window size for Hampel outlier detection on velocity series.
    hampel_nsigma : float
        Hampel threshold in sigma units.
    return_debug : bool
        If True, include intermediate arrays/thresholds.

    Returns
    -------
    result : dict
        Contains per-event time delay and velocity statistics.
    """
    time = np.asarray(time).squeeze()
    light = np.asarray(light)

    if time.ndim != 1:
        raise ValueError("time must be 1D")
    if light.ndim != 2:
        raise ValueError("light must be 2D, shape (N, M)")
    if light.shape[0] != time.shape[0]:
        raise ValueError("time and light must have the same length N")
    if len(valid_cols) != 2:
        raise ValueError("valid_cols must contain exactly two channel indices")

    up_col, dn_col = valid_cols
    y_up = light[:, up_col].astype(float)
    y_dn = light[:, dn_col].astype(float)

    dt_arr = np.diff(time)
    if np.any(dt_arr <= 0):
        raise ValueError("time must be strictly increasing")
    dt = float(np.median(dt_arr))

    if not (0.0 < cfd_fraction < 1.0):
        raise ValueError("cfd_fraction must be in (0, 1)")

    v_min, v_max = sorted([float(speed_bounds[0]), float(speed_bounds[1])])
    if v_min <= 0.0:
        raise ValueError("speed_bounds must be positive")

    delay_min = sensor_distance / v_max
    delay_max = sensor_distance / v_min
    margin_s = max(0.0, float(coarse_delay_margin_us) * 1e-6)
    if expected_speed is None or expected_speed <= 0:
        expected_delay = 0.5 * (delay_min + delay_max)
    else:
        expected_delay = sensor_distance / float(expected_speed)

    def us_to_samples(us, min_n=3):
        n = int(round((us * 1e-6) / dt))
        return max(min_n, n)

    smooth_n = us_to_samples(smooth_window_us, min_n=3)
    if smooth_n % 2 == 0:
        smooth_n += 1
    pre_n = us_to_samples(pre_window_us, min_n=8)
    post_n = us_to_samples(post_window_us, min_n=12)
    merge_gap_n = us_to_samples(detect_merge_gap_us, min_n=2)
    guard_n = max(1, us_to_samples(5.0, min_n=1))

    def moving_average(y, n):
        if n <= 2:
            return y.copy()
        k = np.ones(int(n), dtype=float) / float(n)
        return np.convolve(y, k, mode="same")

    def detect_candidates(y):
        ys = moving_average(y, smooth_n)
        dy = np.gradient(ys, dt)
        dy_abs = np.abs(dy)

        med = np.median(dy_abs)
        mad = np.median(np.abs(dy_abs - med))
        sigma = 1.4826 * mad + 1e-12
        thr = med + deriv_sigma * sigma

        hot = np.flatnonzero(dy_abs > thr)
        if hot.size == 0:
            return {
                "smoothed": ys,
                "derivative": dy,
                "threshold": thr,
                "candidates": np.array([], dtype=int),
            }

        splits = np.where(np.diff(hot) > 1)[0] + 1
        groups = np.split(hot, splits)

        coarse = []
        for g in groups:
            idx = int(g[np.argmax(dy_abs[g])])
            coarse.append(idx)

        coarse = sorted(coarse)
        filtered = []
        if coarse:
            cluster = [coarse[0]]
            for idx in coarse[1:]:
                if idx - cluster[-1] <= merge_gap_n:
                    cluster.append(idx)
                else:
                    best_idx = max(cluster, key=lambda k: dy_abs[k])
                    filtered.append(best_idx)
                    cluster = [idx]
            best_idx = max(cluster, key=lambda k: dy_abs[k])
            filtered.append(best_idx)

        return {
            "smoothed": ys,
            "derivative": dy,
            "threshold": thr,
            "candidates": np.asarray(filtered, dtype=int),
        }

    def cfd_time(y_raw, idx_center):
        n = y_raw.shape[0]
        left = max(0, idx_center - pre_n)
        right = min(n - 1, idx_center + post_n)
        baseline_guard_n = max(guard_n, pre_n // 4)
        pre_right = max(left + 3, idx_center - baseline_guard_n)
        pre_right = min(pre_right, right)

        baseline_seg = y_raw[left:pre_right]
        if baseline_seg.size < 3:
            return None, None

        baseline = float(np.median(baseline_seg))
        noise = float(1.4826 * np.median(np.abs(baseline_seg - baseline)) + 1e-12)

        # Infer polarity locally, not globally. This is important when sensors
        # have different transfer characteristics.
        post_seg = y_raw[pre_right:right + 1] - baseline
        if post_seg.size < 3:
            return None, None
        peak_pos = float(np.max(post_seg))
        peak_neg = float(np.min(post_seg))
        polarity = 1 if peak_pos >= abs(peak_neg) else -1

        signed = polarity * (y_raw - baseline)
        start = max(left + 1, pre_right)
        if start >= right:
            return None, None

        seg = signed[start:right + 1]
        if seg.size < 3:
            return None, None

        peak_rel = int(np.argmax(seg))
        peak_idx = start + peak_rel
        amp = float(signed[peak_idx])
        if amp <= min_snr * noise:
            return None, None

        level = cfd_fraction * amp
        rise = signed[start:peak_idx + 1]
        crossed = np.flatnonzero(rise >= level)
        if crossed.size == 0:
            return None, None

        k = start + int(crossed[0])
        if k <= 0:
            t_cross = float(time[k])
        else:
            y0, y1 = float(signed[k - 1]), float(signed[k])
            t0, t1 = float(time[k - 1]), float(time[k])
            if y1 == y0:
                t_cross = t1
            else:
                t_cross = t0 + (level - y0) * (t1 - t0) / (y1 - y0)

        meta = {
            "baseline": baseline,
            "noise": noise,
            "amplitude": amp,
            "snr": amp / noise,
            "peak_idx": peak_idx,
            "polarity": polarity,
            "cfd_level": baseline + polarity * level,
        }
        return t_cross, meta

    det_up = detect_candidates(y_up)
    det_dn = detect_candidates(y_dn)

    cand_up = det_up["candidates"]
    cand_dn = det_dn["candidates"]
    used_dn = np.zeros(cand_dn.size, dtype=bool)

    event_time = []
    t_up_list = []
    t_dn_list = []
    dt_list = []
    vel_list = []
    delay_err_list = []
    snr_up_list = []
    snr_dn_list = []

    for idx_u in cand_up:
        t_u_coarse = float(time[idx_u])

        pool = []
        for j, idx_d in enumerate(cand_dn):
            if used_dn[j]:
                continue
            t_d_coarse = float(time[idx_d])
            if (t_u_coarse + delay_min - margin_s) <= t_d_coarse <= (t_u_coarse + delay_max + margin_s):
                pool.append(j)

        j_best = None
        idx_d = None

        if pool:
            j_best = min(
                pool,
                key=lambda j: abs((float(time[cand_dn[j]]) - t_u_coarse) - expected_delay),
            )
            idx_d = int(cand_dn[j_best])
        elif downstream_fallback_search:
            # Fallback: search strongest local derivative in downstream within delay window.
            s_left_t = t_u_coarse + delay_min - margin_s
            s_right_t = t_u_coarse + delay_max + margin_s
            i0 = int(np.searchsorted(time, s_left_t, side="left"))
            i1 = int(np.searchsorted(time, s_right_t, side="right")) - 1
            i0 = max(0, min(i0, time.size - 1))
            i1 = max(0, min(i1, time.size - 1))
            if i1 > i0 + 2:
                dy_dn_abs = np.abs(det_dn["derivative"][i0:i1 + 1])
                idx_d = int(i0 + np.argmax(dy_dn_abs))

        if idx_d is None:
            continue

        t_u, meta_u = cfd_time(y_up, int(idx_u))
        t_d, meta_d = cfd_time(y_dn, int(idx_d))
        if (t_u is None) or (t_d is None):
            continue

        dt_evt = float(t_d - t_u)
        if not (delay_min <= dt_evt <= delay_max):
            continue

        v_evt = sensor_distance / dt_evt
        if j_best is not None:
            used_dn[j_best] = True

        event_time.append(0.5 * (t_u + t_d))
        t_up_list.append(t_u)
        t_dn_list.append(t_d)
        dt_list.append(dt_evt)
        vel_list.append(v_evt)
        delay_err_list.append(abs(dt_evt - expected_delay))
        snr_up_list.append(meta_u["snr"])
        snr_dn_list.append(meta_d["snr"])

    event_time = np.asarray(event_time, dtype=float)
    t_up_arr = np.asarray(t_up_list, dtype=float)
    t_dn_arr = np.asarray(t_dn_list, dtype=float)
    dt_arr = np.asarray(dt_list, dtype=float)
    vel_arr_raw = np.asarray(vel_list, dtype=float)
    delay_err_arr = np.asarray(delay_err_list, dtype=float)
    snr_up_arr = np.asarray(snr_up_list, dtype=float)
    snr_dn_arr = np.asarray(snr_dn_list, dtype=float)

    def robust_filter_velocity(v_raw):
        v = np.asarray(v_raw, dtype=float).copy()
        n = v.size
        outlier = np.zeros(n, dtype=bool)
        if n == 0:
            return v, outlier

        # Gate 1: expected-speed envelope (when expected_speed is provided).
        if (
            expected_speed is not None
            and expected_speed > 0.0
            and expected_speed_tolerance is not None
            and expected_speed_tolerance > 0.0
        ):
            tol = float(expected_speed_tolerance) * float(expected_speed)
            outlier |= np.abs(v - float(expected_speed)) > tol

        # Gate 2: Hampel filter for local isolated spikes/dips.
        hw = int(max(1, hampel_window))
        ns = float(max(0.5, hampel_nsigma))
        for i in range(n):
            if not np.isfinite(v[i]):
                outlier[i] = True
                continue
            lo = max(0, i - hw)
            hi = min(n, i + hw + 1)
            win = v[lo:hi]
            win = win[np.isfinite(win)]
            if win.size < 3:
                continue
            med = float(np.median(win))
            mad = float(np.median(np.abs(win - med)))
            sigma_h = 1.4826 * mad
            if sigma_h <= 1e-12:
                continue
            if abs(v[i] - med) > ns * sigma_h:
                outlier[i] = True

        if not np.any(outlier):
            return v, outlier

        # Replace outliers by local robust center to preserve cycle count/time mapping.
        v_fixed = v.copy()
        good = ~outlier & np.isfinite(v_fixed)
        global_med = float(np.median(v_fixed[good])) if np.any(good) else np.nan
        for i in np.where(outlier)[0]:
            lo = max(0, i - hw)
            hi = min(n, i + hw + 1)
            local = v_fixed[lo:hi]
            local_good = local[np.isfinite(local)]
            if local_good.size > 0:
                v_fixed[i] = float(np.median(local_good))
            else:
                v_fixed[i] = global_med
        return v_fixed, outlier

    # Final event de-duplication by time gap (independent from candidate merge gap)
    min_evt_gap_s = max(0.0, float(min_event_gap_us) * 1e-6)
    if event_time.size > 1 and min_evt_gap_s > 0.0:
        order = np.argsort(event_time)
        event_time = event_time[order]
        t_up_arr = t_up_arr[order]
        t_dn_arr = t_dn_arr[order]
        dt_arr = dt_arr[order]
        vel_arr_raw = vel_arr_raw[order]
        delay_err_arr = delay_err_arr[order]
        snr_up_arr = snr_up_arr[order]
        snr_dn_arr = snr_dn_arr[order]

        keep = []
        n_evt = event_time.size
        s = 0
        while s < n_evt:
            e = s + 1
            while e < n_evt and (event_time[e] - event_time[e - 1] <= min_evt_gap_s):
                e += 1
            j_rel = int(np.argmin(delay_err_arr[s:e]))
            keep.append(s + j_rel)
            s = e

        keep = np.asarray(keep, dtype=int)
        event_time = event_time[keep]
        t_up_arr = t_up_arr[keep]
        t_dn_arr = t_dn_arr[keep]
        dt_arr = dt_arr[keep]
        vel_arr_raw = vel_arr_raw[keep]
        delay_err_arr = delay_err_arr[keep]
        snr_up_arr = snr_up_arr[keep]
        snr_dn_arr = snr_dn_arr[keep]

    if apply_velocity_filter:
        vel_arr, vel_outlier = robust_filter_velocity(vel_arr_raw)
    else:
        vel_arr = vel_arr_raw.copy()
        vel_outlier = np.zeros(vel_arr.size, dtype=bool)

    if vel_arr.size > 0:
        median_v = float(np.median(vel_arr))
        mean_v = float(np.mean(vel_arr))
        std_v = float(np.std(vel_arr, ddof=1)) if vel_arr.size > 1 else 0.0
    else:
        median_v = np.nan
        mean_v = np.nan
        std_v = np.nan

    result = {
        "method": "Derivative trigger + Constant Fraction Discrimination (CFD)",
        "distance_m": float(sensor_distance),
        "event_time_s": event_time,
        "t_upstream_s": t_up_arr,
        "t_downstream_s": t_dn_arr,
        "delta_t_s": dt_arr,
        "velocity_mps_raw": vel_arr_raw,
        "velocity_mps": vel_arr,
        "velocity_is_outlier": vel_outlier,
        "snr_upstream": snr_up_arr,
        "snr_downstream": snr_dn_arr,
        "n_events": int(vel_arr.size),
        "velocity_median_mps": median_v,
        "velocity_mean_mps": mean_v,
        "velocity_std_mps": std_v,
        "speed_bounds_mps": (v_min, v_max),
        "expected_speed_mps": expected_speed,
        "is_near_2000mps": bool(np.isfinite(median_v) and (1500.0 <= median_v <= 2600.0)),
    }

    if return_debug:
        result["debug"] = {
            "dt_s": dt,
            "delay_min_s": delay_min,
            "delay_max_s": delay_max,
            "expected_delay_s": expected_delay,
            "smooth_samples": smooth_n,
            "pre_samples": pre_n,
            "post_samples": post_n,
            "detect_merge_samples": merge_gap_n,
            "det_up": det_up,
            "det_dn": det_dn,
        }

    return result


def calc_detonation_velocity_from_areal(Areal_1time, Areal_2light, sensor_distance=0.1, **kwargs):
    """
    Convenience wrapper for your current data layout:
    - use Areal_2light[:, 0] and Areal_2light[:, 1]
    - ignore Areal_2light[:, 2] (empty sampling)
    """
    return calc_detonation_velocity_cfd(
        time=Areal_1time,
        light=Areal_2light,
        sensor_distance=sensor_distance,
        valid_cols=(0, 1),
        **kwargs,
    )


# Example (not auto-executed):
# result = calc_detonation_velocity_from_areal(
#     Areal_1time,
#     Areal_2light,
#     sensor_distance=0.1,
#     expected_speed=2000.0,
#     speed_bounds=(1200.0, 3500.0),
#     cfd_fraction=0.5,
#     return_debug=True,
# )
# print("n_events:", result["n_events"])
# print("median velocity (m/s):", result["velocity_median_mps"])



def build_detonation_cycle_dataset(
    Areal_1time,
    Areal_2light,
    areal_data=None,
    sensor_distance=0.1,
    slice_window_us=(150.0, 250.0),
    event_time_mode="midpoint",
    velocity_kwargs=None,
    auto_cycle_gap_factor=0.35,
    auto_cycle_dedup=True,
    auto_startup_trim=True,
    startup_max_trim=1,
    startup_velocity_dev_tol=0.25,
    startup_gap_dev_tol=0.30,
    startup_head_time_factor=0.25,
    startup_isolated_gap_factor=2.5,
    auto_low_light_trim=True,
    low_light_ratio=0.20,
    low_light_balance_min=0.35,
    low_light_min_snr=10.0,
    low_light_ref_window=2,
):
    """
    Build cycle-level detonation dataset from optical TOF velocity result.

    Returned structure includes:
    - identified cycle count
    - per-cycle sliced Areal_* data (ready to plot)
    - per-cycle velocity
    - per-cycle feature time

    Parameters
    ----------
    Areal_1time : array, shape (N,)
    Areal_2light : array, shape (N, M)
    areal_data : dict or None
        Optional mapping like {"Areal_4highf_p": arr, ...}. If None, it auto-collects
        variables from globals() whose names start with "Areal_" and length N.
    sensor_distance : float
        Sensor spacing in meters (100 mm -> 0.1 m).
    slice_window_us : float or (float, float)
        (pre_us, post_us) around cycle feature time.
    event_time_mode : str
        "midpoint" (recommended) or "light_peak".
    velocity_kwargs : dict or None
        Extra kwargs passed to calc_detonation_velocity_from_areal().
    auto_cycle_gap_factor : float
        Automatic de-dup gap as a fraction of the median cycle period.
        Events closer than this gap are treated as likely repeated triggers.
    auto_cycle_dedup : bool
        If True, perform cycle-level automatic de-duplication using optical quality
        and outlier status only (no pressure input).
    auto_startup_trim : bool
        If True, trim the first detected event when it is likely a startup false trigger.
    startup_max_trim : int
        Maximum number of leading events to trim.
    startup_velocity_dev_tol : float
        Relative tolerance for first-event velocity deviation from robust core speed.
    startup_gap_dev_tol : float
        Relative tolerance for first-event period-gap deviation from robust period.
    startup_head_time_factor : float
        First-event age threshold as a fraction of median cycle period.
    startup_isolated_gap_factor : float
        If first gap is larger than this factor times median period and first event is
        near recording start, it is treated as a startup false trigger.
    auto_low_light_trim : bool
        If True, remove cycles that are likely low-light false triggers.
    low_light_ratio : float
        Threshold ratio to local optical amplitude reference.
    low_light_balance_min : float
        Minimum acceptable channel-amplitude balance for low-light cycle rejection.
    low_light_min_snr : float
        Minimum per-event SNR used by low-light rejection.
    low_light_ref_window : int
        Half-window size (in cycles) for local amplitude reference.
    """
    time = np.asarray(Areal_1time).squeeze()
    light = np.asarray(Areal_2light)

    if time.ndim != 1:
        raise ValueError("Areal_1time must be 1D")
    if light.ndim != 2 or light.shape[0] != time.size:
        raise ValueError("Areal_2light must be 2D with matching first dimension")

    if velocity_kwargs is None:
        velocity_kwargs = {}

    vel_result = calc_detonation_velocity_from_areal(
        Areal_1time=time,
        Areal_2light=light,
        sensor_distance=sensor_distance,
        **velocity_kwargs,
    )

    t_evt = np.asarray(vel_result["event_time_s"], dtype=float)
    t_up = np.asarray(vel_result["t_upstream_s"], dtype=float)
    t_dn = np.asarray(vel_result["t_downstream_s"], dtype=float)
    v_evt_raw = np.asarray(vel_result.get("velocity_mps_raw", vel_result["velocity_mps"]), dtype=float)
    v_evt = np.asarray(vel_result["velocity_mps"], dtype=float)
    v_evt_outlier = np.asarray(
        vel_result.get("velocity_is_outlier", np.zeros(v_evt.shape, dtype=bool)),
        dtype=bool,
    )
    dt_evt = np.asarray(vel_result["delta_t_s"], dtype=float)
    snr_up = np.asarray(vel_result.get("snr_upstream", np.full(v_evt.shape, np.nan)), dtype=float)
    snr_dn = np.asarray(vel_result.get("snr_downstream", np.full(v_evt.shape, np.nan)), dtype=float)

    # Cycle-level auto de-duplication for repeated close optical triggers.
    # This makes cycle indexing more robust across cases with dirty light signals.
    if auto_cycle_dedup and t_evt.size >= 4:
        order = np.argsort(t_evt)
        t_evt = t_evt[order]
        t_up = t_up[order]
        t_dn = t_dn[order]
        v_evt_raw = v_evt_raw[order]
        v_evt = v_evt[order]
        v_evt_outlier = v_evt_outlier[order]
        dt_evt = dt_evt[order]
        snr_up = snr_up[order]
        snr_dn = snr_dn[order]

        d_evt = np.diff(t_evt)
        period_s = float(np.median(d_evt)) if d_evt.size > 0 else np.nan
        min_gap_s = float(max(0.0, auto_cycle_gap_factor)) * period_s if np.isfinite(period_s) else np.nan

        if np.isfinite(min_gap_s) and min_gap_s > 0.0:
            expected_v = float(vel_result.get("expected_speed_mps", np.nan))
            keep = []
            s_idx = 0
            n_evt = t_evt.size

            while s_idx < n_evt:
                e_idx = s_idx + 1
                while e_idx < n_evt and (t_evt[e_idx] - t_evt[e_idx - 1] <= min_gap_s):
                    e_idx += 1

                sl = slice(s_idx, e_idx)
                q_snr = np.minimum(snr_up[sl], snr_dn[sl])
                q_snr = np.nan_to_num(q_snr, nan=0.0, posinf=0.0, neginf=0.0)
                good_flag = (~v_evt_outlier[sl]).astype(float)

                if np.isfinite(expected_v) and expected_v > 1e-9:
                    near_expected = -np.abs(v_evt[sl] - expected_v) / expected_v
                else:
                    near_expected = np.zeros_like(good_flag)

                score = 2.5 * good_flag + 0.7 * np.log1p(np.clip(q_snr, 0.0, None)) + 0.8 * near_expected
                j_pick = s_idx + int(np.argmax(score))
                keep.append(j_pick)
                s_idx = e_idx

            keep = np.asarray(keep, dtype=int)
            t_evt = t_evt[keep]
            t_up = t_up[keep]
            t_dn = t_dn[keep]
            v_evt_raw = v_evt_raw[keep]
            v_evt = v_evt[keep]
            v_evt_outlier = v_evt_outlier[keep]
            dt_evt = dt_evt[keep]
            snr_up = snr_up[keep]
            snr_dn = snr_dn[keep]

    # Trim likely startup false trigger(s): optical-only, no pressure usage.
    if auto_startup_trim and t_evt.size >= 4 and int(startup_max_trim) > 0:
        max_trim = int(max(0, startup_max_trim))
        n_trim = 0

        while t_evt.size >= 4 and n_trim < max_trim:
            d_evt = np.diff(t_evt)
            period_s = float(np.median(d_evt)) if d_evt.size > 0 else np.nan
            if not (np.isfinite(period_s) and period_s > 0.0):
                break

            good_core = (~v_evt_outlier) & np.isfinite(v_evt)
            if np.sum(good_core) >= 3:
                vv = np.sort(v_evt[good_core])
                ql, qh = np.percentile(vv, [10.0, 90.0])
                core = vv[(vv >= ql) & (vv <= qh)]
                v_ref = float(np.median(core)) if core.size > 0 else float(np.median(vv))
            else:
                v_ref = float(vel_result.get("expected_speed_mps", np.nan))
                if not np.isfinite(v_ref):
                    v_ref = float(np.nanmedian(v_evt[np.isfinite(v_evt)])) if np.any(np.isfinite(v_evt)) else np.nan

            if not (np.isfinite(v_ref) and v_ref > 1e-12):
                break

            first_gap = float(t_evt[1] - t_evt[0])
            first_v = float(v_evt[0])
            first_out = bool(v_evt_outlier[0])
            first_age = float(t_evt[0] - time[0])

            vel_dev = abs(first_v - v_ref) / v_ref
            gap_dev = abs(first_gap - period_s) / period_s

            # Robust startup false-trigger criterion:
            # 1) Quality-based trim (legacy): flagged outlier + inconsistent speed/gap.
            trim_by_quality = first_out and (
                (vel_dev > float(startup_velocity_dev_tol))
                or (gap_dev > float(startup_gap_dev_tol))
            )

            # 2) Time-structure trim (new): event appears almost at acquisition start
            # and is isolated from the next event by an abnormally long gap.
            trim_by_startup_isolation = (
                (first_age <= float(startup_head_time_factor) * period_s)
                and (first_gap >= float(startup_isolated_gap_factor) * period_s)
            )

            if trim_by_quality or trim_by_startup_isolation:
                t_evt = t_evt[1:]
                t_up = t_up[1:]
                t_dn = t_dn[1:]
                v_evt_raw = v_evt_raw[1:]
                v_evt = v_evt[1:]
                v_evt_outlier = v_evt_outlier[1:]
                dt_evt = dt_evt[1:]
                snr_up = snr_up[1:]
                snr_dn = snr_dn[1:]
                n_trim += 1
            else:
                break

    # Trim low-light false-trigger cycles (pure optical):
    # reject events with very weak light amplitude relative to local reference,
    # combined with poor channel balance / low SNR / velocity outlier.
    opt_amp_evt = np.full(t_evt.shape, np.nan, dtype=float)
    opt_bal_evt = np.full(t_evt.shape, np.nan, dtype=float)
    opt_low_light_evt = np.zeros(t_evt.shape, dtype=bool)

    if auto_low_light_trim and t_evt.size >= 3:
        for i, tf in enumerate(t_evt):
            i0 = int(np.searchsorted(time, tf - 0.30e-3, side="left"))
            i1 = int(np.searchsorted(time, tf + 0.35e-3, side="right")) - 1
            i0 = max(0, min(i0, time.size - 1))
            i1 = max(0, min(i1, time.size - 1))
            if i1 <= i0 + 6:
                continue

            t_rel_ms = (time[i0:i1 + 1] - tf) * 1e3
            seg = np.asarray(light[i0:i1 + 1, :2], dtype=float)
            up = seg[:, 0]
            dn = seg[:, 1]

            base_mask = (t_rel_ms >= -0.25) & (t_rel_ms <= -0.05)
            peak_mask = (t_rel_ms >= -0.02) & (t_rel_ms <= 0.20)
            if np.sum(base_mask) < 4 or np.sum(peak_mask) < 4:
                continue

            up_a = float(np.nanmax(up[peak_mask]) - np.nanmedian(up[base_mask]))
            dn_a = float(np.nanmax(dn[peak_mask]) - np.nanmedian(dn[base_mask]))
            amp_i = max(up_a, dn_a)
            opt_amp_evt[i] = amp_i

            if up_a > 0.0 and dn_a > 0.0 and np.isfinite(up_a) and np.isfinite(dn_a):
                opt_bal_evt[i] = float(np.exp(-abs(np.log(up_a / dn_a))))

        # local rolling amplitude reference
        w = int(max(1, low_light_ref_window))
        amp_ref_evt = np.full(opt_amp_evt.shape, np.nan, dtype=float)
        for i in range(opt_amp_evt.size):
            lo = max(0, i - w)
            hi = min(opt_amp_evt.size, i + w + 1)
            seg = opt_amp_evt[lo:hi]
            seg = seg[np.isfinite(seg)]
            if seg.size > 0:
                amp_ref_evt[i] = float(np.median(seg))

        global_ref = float(np.nanmedian(opt_amp_evt[np.isfinite(opt_amp_evt)])) if np.any(np.isfinite(opt_amp_evt)) else np.nan
        if np.isfinite(global_ref):
            amp_ref_evt = np.where(np.isfinite(amp_ref_evt), amp_ref_evt, global_ref)

        low_amp = np.isfinite(opt_amp_evt) & np.isfinite(amp_ref_evt) & (opt_amp_evt < float(low_light_ratio) * amp_ref_evt)
        snr_evt = np.minimum(snr_up, snr_dn)

        weak_quality = (
            (snr_evt < float(low_light_min_snr))
            | (~np.isfinite(opt_bal_evt))
            | (opt_bal_evt < float(low_light_balance_min))
            | v_evt_outlier
        )

        opt_low_light_evt = low_amp & weak_quality

        if np.any(opt_low_light_evt):
            keep = ~opt_low_light_evt
            # keep at least a minimal cycle set to avoid over-trimming pathological cases
            if np.sum(keep) >= 3:
                t_evt = t_evt[keep]
                t_up = t_up[keep]
                t_dn = t_dn[keep]
                v_evt_raw = v_evt_raw[keep]
                v_evt = v_evt[keep]
                v_evt_outlier = v_evt_outlier[keep]
                dt_evt = dt_evt[keep]
                snr_up = snr_up[keep]
                snr_dn = snr_dn[keep]
                opt_amp_evt = opt_amp_evt[keep]
                opt_bal_evt = opt_bal_evt[keep]
                opt_low_light_evt = opt_low_light_evt[keep]
            else:
                opt_low_light_evt[:] = False

    if np.isscalar(slice_window_us):
        pre_us = float(slice_window_us)
        post_us = float(slice_window_us)
    else:
        pre_us, post_us = float(slice_window_us[0]), float(slice_window_us[1])

    dt = float(np.median(np.diff(time)))
    pre_n = max(1, int(round(pre_us * 1e-6 / dt)))
    post_n = max(1, int(round(post_us * 1e-6 / dt)))

    if areal_data is None:
        areal_data = {}
        for name, val in globals().items():
            if not name.startswith("Areal_"):
                continue
            arr = np.asarray(val)
            if arr.ndim >= 1 and arr.shape[0] == time.size:
                areal_data[name] = arr
    else:
        areal_data = {k: np.asarray(v) for k, v in areal_data.items()}

    def feature_time_for_cycle(i):
        if event_time_mode == "midpoint":
            return float(t_evt[i])
        if event_time_mode == "light_peak":
            i0 = int(np.searchsorted(time, t_up[i]))
            i1 = int(np.searchsorted(time, t_dn[i]))
            left = max(0, min(i0, i1) - 2)
            right = min(time.size - 1, max(i0, i1) + 2)
            seg = np.asarray(light[left:right + 1, :2])
            rel = int(np.argmax(np.max(np.abs(seg), axis=1)))
            return float(time[left + rel])
        raise ValueError("event_time_mode must be 'midpoint' or 'light_peak'")

    cycles = []
    cycle_times = []
    cycle_velocities = []

    for i in range(v_evt.size):
        tf = feature_time_for_cycle(i)
        ic = int(np.searchsorted(time, tf))
        s = max(0, ic - pre_n)
        e = min(time.size - 1, ic + post_n)

        data_slice = {
            "Areal_1time": time[s:e + 1].copy(),
        }
        for name, arr in areal_data.items():
            if arr.ndim >= 1 and arr.shape[0] == time.size:
                data_slice[name] = arr[s:e + 1].copy()

        cycle = {
            "cycle_id": int(i + 1),
            "start_idx": int(s),
            "end_idx": int(e),
            "time_window_s": (float(time[s]), float(time[e])),
            "time_feature_s": float(tf),
            "velocity_mps_raw": float(v_evt_raw[i]),
            "velocity_mps": float(v_evt[i]),
            "velocity_is_outlier": bool(v_evt_outlier[i]),
            "delta_t_s": float(dt_evt[i]),
            "t_upstream_s": float(t_up[i]),
            "t_downstream_s": float(t_dn[i]),
            "optical_amp": float(opt_amp_evt[i]) if i < opt_amp_evt.size and np.isfinite(opt_amp_evt[i]) else np.nan,
            "optical_balance": float(opt_bal_evt[i]) if i < opt_bal_evt.size and np.isfinite(opt_bal_evt[i]) else np.nan,
            "optical_low_light": bool(opt_low_light_evt[i]) if i < opt_low_light_evt.size else False,
            "data": data_slice,
        }
        cycles.append(cycle)
        cycle_times.append(tf)
        cycle_velocities.append(v_evt[i])

    cycle_times = np.asarray(cycle_times, dtype=float)
    cycle_velocities = np.asarray(cycle_velocities, dtype=float)

    if cycle_velocities.size > 0:
        v_median = float(np.median(cycle_velocities))
        v_mean = float(np.mean(cycle_velocities))
        v_std = float(np.std(cycle_velocities, ddof=1)) if cycle_velocities.size > 1 else 0.0
    else:
        v_median = np.nan
        v_mean = np.nan
        v_std = np.nan

    return {
        "method": vel_result["method"],
        "distance_m": float(sensor_distance),
        "n_cycles": int(len(cycles)),
        "cycle_time_s": cycle_times,
        "velocity_mps_raw": v_evt_raw.copy(),
        "velocity_mps": cycle_velocities,
        "velocity_is_outlier": v_evt_outlier.copy(),
        "optical_amp": opt_amp_evt.copy(),
        "optical_balance": opt_bal_evt.copy(),
        "optical_low_light": opt_low_light_evt.copy(),
        "velocity_summary": {
            "median_mps": v_median,
            "mean_mps": v_mean,
            "std_mps": v_std,
        },
        "cycles": cycles,
    }


def plot_detonation_cycles_sci(cycle_result, n_show=4, reference_speed=2000.0):
    """
    SCI-style visualization:
    1) Typical slices for first n_show cycles (light + pressure traces)
    2) All-cycle feature-time vs detonation-velocity plot

    Returns
    -------
    (fig_slices, fig_velocity)
    """
    import matplotlib.pyplot as plt

    cycles = cycle_result.get("cycles", [])
    n_total = len(cycles)
    if n_total == 0:
        raise ValueError("No cycles found in cycle_result")

    n_show = int(max(1, min(n_show, n_total)))

    rc = {
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }

    with plt.rc_context(rc):
        fig_slices, axes = plt.subplots(
            n_show,
            2,
            figsize=(12.0, 2.7 * n_show),
            constrained_layout=True,
            sharex=False,
        )
        if n_show == 1:
            axes = np.array([axes])

        for i in range(n_show):
            cyc = cycles[i]
            data = cyc["data"]
            t = np.asarray(data["Areal_1time"]).squeeze()
            t_ms = (t - cyc["time_feature_s"]) * 1e3

            ax_l = axes[i, 0]
            ax_r = axes[i, 1]

            # Left panel: optical signals
            if "Areal_2light" in data:
                y_light = np.asarray(data["Areal_2light"])
                if y_light.ndim == 1:
                    ax_l.plot(t_ms, y_light, color="#1f77b4", label="Light[0]")
                else:
                    if y_light.shape[1] >= 1:
                        ax_l.plot(t_ms, y_light[:, 0], color="#1f77b4", label="Light[0]")
                    if y_light.shape[1] >= 2:
                        ax_l.plot(t_ms, y_light[:, 1], color="#d62728", label="Light[1]")
            ax_l.axvline(0.0, color="k", lw=1.0, ls="--", alpha=0.6)
            ax_l.set_ylabel("Optical Intensity")
            ax_l.set_xlabel("Time Relative to Cycle Feature (ms)")
            ax_l.grid(True, ls=":", alpha=0.35)
            ax_l.legend(loc="upper right", frameon=False)
            ax_l.set_title(
                f"Cycle {cyc['cycle_id']}  |  v = {cyc['velocity_mps']:.1f} m/s  |  t = {cyc['time_feature_s']:.6f} s"
            )

            # Right panel: pressure-related traces
            has_pressure = False
            if "Areal_4highf_p" in data:
                y4 = np.asarray(data["Areal_4highf_p"]).squeeze()
                ax_r.plot(t_ms, y4, color="#2ca02c", label="Areal_4highf_p")
                has_pressure = True
            if "Areal_5highf_p_osc" in data:
                y5 = np.asarray(data["Areal_5highf_p_osc"])
                if y5.ndim == 1:
                    ax_r.plot(t_ms, y5, color="#9467bd", label="Areal_5highf_p_osc")
                else:
                    ncol = min(3, y5.shape[1])
                    cols = ["#9467bd", "#8c564b", "#17becf"]
                    for c in range(ncol):
                        ax_r.plot(t_ms, y5[:, c], color=cols[c], lw=1.2, alpha=0.9, label=f"Areal_5highf_p_osc[{c}]")
                has_pressure = True

            ax_r.axvline(0.0, color="k", lw=1.0, ls="--", alpha=0.6)
            ax_r.set_ylabel("Pressure / Signal")
            ax_r.set_xlabel("Time Relative to Cycle Feature (ms)")
            ax_r.grid(True, ls=":", alpha=0.35)
            if has_pressure:
                ax_r.legend(loc="upper right", frameon=False)
            ax_r.set_title("Pressure-related Slice")

        # Figure 2: all cycles, feature time vs velocity
        fig_vel, axv = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
        tt = np.asarray(cycle_result["cycle_time_s"], dtype=float)
        vv = np.asarray(cycle_result["velocity_mps"], dtype=float)

        axv.plot(tt, vv, "-o", color="#1f77b4", ms=4.5, lw=1.4, label="Cycle Velocity")
        if reference_speed is not None:
            axv.axhline(float(reference_speed), color="#d62728", ls="--", lw=1.2, label=f"Reference {reference_speed:.0f} m/s")

        if vv.size > 0 and np.isfinite(np.nanmedian(vv)):
            axv.axhline(float(np.nanmedian(vv)), color="#2ca02c", ls=":", lw=1.2, label=f"Median {np.nanmedian(vv):.1f} m/s")

        axv.set_xlabel("Cycle Feature Time (s)")
        axv.set_ylabel("Detonation Velocity (m/s)")
        axv.set_title("Detonation Velocity Evolution")
        axv.grid(True, ls=":", alpha=0.35)
        axv.legend(loc="best", frameon=False)

    return fig_slices, fig_vel


def get_frozen_build_options():
    """Return frozen defaults for cycle building from optical channels.

    Returns
    -------
    dict
        Keyword arguments ready to pass into `build_detonation_cycle_dataset`.
    """
    return {
        "sensor_distance": 0.1,
        "slice_window_us": (560.0, 1650.0),
        "event_time_mode": "midpoint",
        "auto_cycle_dedup": True,
        "auto_cycle_gap_factor": 0.35,
        "auto_startup_trim": True,
        "startup_max_trim": 1,
        "startup_velocity_dev_tol": 0.25,
        "startup_gap_dev_tol": 0.30,
        "auto_low_light_trim": True,
        "low_light_ratio": 0.20,
        "low_light_balance_min": 0.35,
        "low_light_min_snr": 10.0,
        "low_light_ref_window": 2,
        "velocity_kwargs": {
            "expected_speed": 2000.0,
            "speed_bounds": (1000.0, 3500.0),
            "cfd_fraction": 0.1,
            "min_event_gap_us": 1000.0,
            "detect_merge_gap_us": 66.0,
            "expected_speed_tolerance": 0.25,
        },
    }


def estimate_optical_cj_from_cycle_result(cycle_result):
    """Estimate cycle-wise CJ pressure from optical-only velocity path.

    Parameters
    ----------
    cycle_result : dict
        Output of `build_detonation_cycle_dataset`.

    Returns
    -------
    dict
        Diagnostic arrays and summary statistics, including:
        - `cycle_time`, `velocity_for_cj`, `M_D`
        - `p_cj_optical`, `p_cj_measured`, `peak_p_a4`
        - `validation_mae_pa`, `validation_mape`
    """
    # ===== Optical-only CJ Estimation Settings =====
    p0 = 1.0e5
    gamma = 1.4

    # Effective sound speed used in M_D->p_CJ mapping
    # NOTE: this is an "effective" mapping constant for optical TOF in noisy experiments.
    a0_for_cj = 386.0

    # Optical expected-speed regularization (pure optical; no pressure feedback)
    expected_speed_ref = 2000.0
    max_expected_blend = 0.95
    outlier_blend_boost = 0.80

    # Temporal-consistency regularization (pure optical)
    trend_window = 2
    temporal_resid_threshold = 0.08
    temporal_resid_span = 0.25
    temporal_blend_boost = 0.20
    spike_blend_boost = 0.80
    neighbor_coherence_tol = 0.12
    neighbor_dev_threshold = 0.10
    neighbor_dev_span = 0.20
    neighbor_dev_blend_boost = 1.00
    base_ref_mix = 0.30
    trend_dev_gain = 0.70
    trend_dev_threshold = 0.15
    trend_dev_span = 0.20
    quality_mix_boost = 0.08

    # Startup transient handling (pure optical): early cycles are often unstable.
    startup_cycle_count = 4
    startup_blend_boost = 1.10

    # Optical amplitude / balance quality
    amp_ratio_min = 0.20
    amp_quality_min = 0.35
    balance_quality_min = 0.45
    quality_blend_boost = 0.35

    # CJ plateau extraction on Areal_4highf_p (validation only)
    plateau_after_peak_ms = (0.01, 0.08)
    plateau_window_fallback_ms = (0.02, 0.12)

    cycle_time = np.asarray(cycle_result["cycle_time_s"], dtype=float)
    velocity = np.asarray(cycle_result["velocity_mps"], dtype=float)
    velocity_raw = np.asarray(cycle_result.get("velocity_mps_raw", velocity), dtype=float)
    velocity_is_outlier = np.asarray(
        cycle_result.get("velocity_is_outlier", np.zeros_like(velocity, dtype=bool)),
        dtype=bool,
    )
    cycles = cycle_result["cycles"]
    n_cycles = len(cycles)


    def rolling_median(x, window=2):
        x = np.asarray(x, dtype=float)
        n = x.size
        out = np.full(n, np.nan, dtype=float)
        w = int(max(1, window))
        for i in range(n):
            lo = max(0, i - w)
            hi = min(n, i + w + 1)
            seg = x[lo:hi]
            seg = seg[np.isfinite(seg)]
            if seg.size > 0:
                out[i] = float(np.median(seg))
        return out


    # ----- Optical quality metrics (pure optical) -----
    amp_up = np.full(n_cycles, np.nan, dtype=float)
    amp_dn = np.full(n_cycles, np.nan, dtype=float)
    optical_amp = np.full(n_cycles, np.nan, dtype=float)
    amp_balance = np.full(n_cycles, np.nan, dtype=float)

    for i, cyc in enumerate(cycles):
        y_light = np.asarray(cyc["data"].get("Areal_2light", np.array([])), dtype=float)
        if y_light.ndim != 2 or y_light.shape[1] < 2:
            continue

        t = np.asarray(cyc["data"]["Areal_1time"]).squeeze()
        t_ms = (t - cyc["time_feature_s"]) * 1e3
        up = y_light[:, 0]
        dn = y_light[:, 1]

        base_mask = t_ms < -0.10
        peak_mask = (t_ms >= -0.02) & (t_ms <= 0.20)
        if np.sum(base_mask) < 5 or np.sum(peak_mask) < 5:
            continue

        up_base = float(np.nanmedian(up[base_mask]))
        dn_base = float(np.nanmedian(dn[base_mask]))
        up_a = float(np.nanmax(up[peak_mask]) - up_base)
        dn_a = float(np.nanmax(dn[peak_mask]) - dn_base)

        amp_up[i] = up_a
        amp_dn[i] = dn_a
        optical_amp[i] = max(up_a, dn_a)

        if np.isfinite(up_a) and np.isfinite(dn_a) and abs(dn_a) > 1e-12 and up_a > 0.0 and dn_a > 0.0:
            # 1 means best (balanced), -> 0 means highly imbalanced
            amp_balance[i] = float(np.exp(-abs(np.log(up_a / dn_a))))

    # Prefer cycle-level optical metrics from builder (closer to event timing),
    # and fallback to local re-computation when unavailable.
    opt_amp_evt = np.asarray(cycle_result.get("optical_amp", np.full(n_cycles, np.nan)), dtype=float)
    opt_bal_evt = np.asarray(cycle_result.get("optical_balance", np.full(n_cycles, np.nan)), dtype=float)
    if opt_amp_evt.size == n_cycles:
        optical_amp = np.where(np.isfinite(opt_amp_evt), opt_amp_evt, optical_amp)
    if opt_bal_evt.size == n_cycles:
        amp_balance = np.where(np.isfinite(opt_bal_evt), opt_bal_evt, amp_balance)

    amp_ref = float(np.nanmedian(optical_amp[np.isfinite(optical_amp)])) if np.any(np.isfinite(optical_amp)) else np.nan

    # Quantile-based amplitude normalization is more robust across cases than median-only scaling.
    if np.sum(np.isfinite(optical_amp)) >= 4:
        amp_p20, amp_p80 = np.percentile(optical_amp[np.isfinite(optical_amp)], [20.0, 80.0])
        amp_span = max(float(amp_p80 - amp_p20), 1e-12)
        amp_norm = np.clip((optical_amp - amp_p20) / amp_span, 0.0, 1.0)
    else:
        amp_norm = np.full(n_cycles, np.nan, dtype=float)
        if np.isfinite(amp_ref) and amp_ref > 1e-12:
            amp_norm = np.clip(optical_amp / amp_ref, 0.0, 1.0)

    bal_norm = np.clip(np.nan_to_num(amp_balance, nan=0.0), 0.0, 1.0)
    confidence = np.sqrt(np.clip(np.nan_to_num(amp_norm, nan=0.0) * bal_norm, 0.0, 1.0))
    quality_low = (
        (np.nan_to_num(amp_norm, nan=0.0) < amp_quality_min)
        | (bal_norm < balance_quality_min)
        | velocity_is_outlier
    )

    low_optical_amp = np.isfinite(optical_amp) & np.isfinite(amp_ref) & (optical_amp < amp_ratio_min * amp_ref)


    # ----- Optical-only velocity regularization for CJ -----
    v_trend = rolling_median(velocity, window=trend_window)

    for i in range(n_cycles):
        if not np.isfinite(v_trend[i]):
            v_trend[i] = expected_speed_ref

    temporal_residual = np.abs(velocity - v_trend) / np.maximum(v_trend, 1e-12)
    neighbor_coherent = np.zeros(n_cycles, dtype=bool)
    neighbor_residual = np.zeros(n_cycles, dtype=float)
    for i in range(1, n_cycles - 1):
        v_l = velocity[i - 1]
        v_r = velocity[i + 1]
        v_m = float(np.median([v_l, v_r]))
        if np.isfinite(v_l) and np.isfinite(v_r) and np.isfinite(v_m) and v_m > 1e-12:
            neighbor_coherent[i] = (abs(v_l - v_r) / v_m) < neighbor_coherence_tol
            if neighbor_coherent[i]:
                neighbor_residual[i] = abs(velocity[i] - v_m) / v_m

    temporal_spike_flag = (temporal_residual > temporal_resid_threshold) & neighbor_coherent
    neighbor_spike_flag = (neighbor_residual > neighbor_dev_threshold) & neighbor_coherent

    expected_blend = np.clip(max_expected_blend * (1.0 - np.nan_to_num(confidence, nan=0.0)), 0.0, max_expected_blend)
    expected_blend = expected_blend + outlier_blend_boost * velocity_is_outlier.astype(float)
    expected_blend = expected_blend + temporal_blend_boost * np.clip(
        (temporal_residual - temporal_resid_threshold) / max(temporal_resid_span, 1e-12),
        0.0,
        1.0,
    )
    expected_blend = expected_blend + spike_blend_boost * temporal_spike_flag.astype(float)
    expected_blend = expected_blend + quality_blend_boost * quality_low.astype(float)
    expected_blend = expected_blend + neighbor_dev_blend_boost * np.clip(
        (neighbor_residual - neighbor_dev_threshold) / max(neighbor_dev_span, 1e-12),
        0.0,
        1.0,
    )

    n_start = int(max(0, min(startup_cycle_count, n_cycles)))
    if n_start > 0:
        expected_blend[:n_start] = expected_blend[:n_start] + startup_blend_boost

    expected_blend = np.clip(expected_blend, 0.0, 0.98)

    # Robust global optical speed reference (pure optical):
    # use central non-outlier velocity distribution to avoid startup bias.
    good_v = np.isfinite(velocity) & (~velocity_is_outlier)
    if np.sum(good_v) >= 4:
        vv = np.sort(velocity[good_v])
        ql, qh = np.percentile(vv, [10.0, 90.0])
        core = vv[(vv >= ql) & (vv <= qh)]
        v_global_ref = float(np.median(core)) if core.size > 0 else float(np.median(vv))
    else:
        v_global_ref = float(np.nanmedian(velocity[np.isfinite(velocity)]))
    if not np.isfinite(v_global_ref):
        v_global_ref = expected_speed_ref

    trend_dev = np.abs(v_trend - v_global_ref) / max(v_global_ref, 1e-12)
    mix_ref = base_ref_mix + trend_dev_gain * np.clip(
        (trend_dev - trend_dev_threshold) / max(trend_dev_span, 1e-12),
        0.0,
        1.0,
    )
    low_side_flag = (velocity < v_global_ref).astype(float)
    mix_ref = mix_ref + quality_mix_boost * quality_low.astype(float) * low_side_flag
    mix_ref = np.clip(mix_ref, 0.0, 0.95)

    v_target = mix_ref * v_global_ref + (1.0 - mix_ref) * v_trend
    velocity_for_cj = (1.0 - expected_blend) * velocity + expected_blend * v_target


    # ----- Optical speed -> CJ pressure (optical only) -----
    valid_for_cj = np.isfinite(velocity_for_cj) & (~low_optical_amp)
    M_D = np.full(n_cycles, np.nan, dtype=float)
    p_cj_optical = np.full(n_cycles, np.nan, dtype=float)

    if np.any(valid_for_cj):
        M_D_valid = velocity_for_cj[valid_for_cj] / a0_for_cj
        M2_valid = np.maximum(M_D_valid**2, 1.0)
        p_valid = p0 * (1.0 + (2.0 * gamma / (gamma + 1.0)) * (M2_valid - 1.0))
        M_D[valid_for_cj] = M_D_valid
        p_cj_optical[valid_for_cj] = p_valid


    # ----- Measured CJ plateau from Areal_4highf_p (validation only) -----
    def estimate_cj_plateau_from_a4(t_rel_ms, y4):
        if y4.size == 0:
            return np.nan

        y4 = np.asarray(y4, dtype=float)
        finite = np.isfinite(y4)
        if np.sum(finite) < 8:
            return np.nan

        idx_peak = int(np.nanargmax(y4))
        t_peak = float(t_rel_ms[idx_peak])

        w0 = t_peak + plateau_after_peak_ms[0]
        w1 = t_peak + plateau_after_peak_ms[1]
        m = (t_rel_ms >= w0) & (t_rel_ms <= w1)

        if np.sum(m) < 8:
            m = (t_rel_ms >= plateau_window_fallback_ms[0]) & (t_rel_ms <= plateau_window_fallback_ms[1])
        if np.sum(m) < 8:
            return np.nan

        yy = np.asarray(y4[m], dtype=float)
        yy = yy[np.isfinite(yy)]
        if yy.size < 6:
            return np.nan

        ql, qh = np.percentile(yy, [20.0, 80.0])
        core = yy[(yy >= ql) & (yy <= qh)]
        if core.size < 4:
            core = yy

        return float(np.median(core))


    p_cj_measured = np.full(n_cycles, np.nan, dtype=float)
    peak_p_a4 = np.full(n_cycles, np.nan, dtype=float)

    for i, cyc in enumerate(cycles):
        t = np.asarray(cyc["data"]["Areal_1time"]).squeeze()
        t_rel_ms = (t - cyc["time_feature_s"]) * 1e3
        y4 = np.asarray(cyc["data"].get("Areal_4highf_p", np.array([]))).squeeze()

        if y4.size > 0:
            peak_p_a4[i] = float(np.nanmax(y4))
        p_cj_measured[i] = estimate_cj_plateau_from_a4(t_rel_ms, y4)


    # ---------- 图1：保持原有展示形式 ----------
    max_rows_per_fig = 6

    for k0 in range(0, n_cycles, max_rows_per_fig):
        k1 = min(k0 + max_rows_per_fig, n_cycles)
        group = list(range(k0, k1))

        fig, axes = plt.subplots(
            len(group), 1,
            figsize=(10.5, 2.7 * len(group)),
            constrained_layout=True,
            sharex=False,
        )
        if len(group) == 1:
            axes = [axes]

        for ax, i in zip(axes, group):
            cyc = cycles[i]
            t = np.asarray(cyc["data"]["Areal_1time"]).squeeze()
            t_ms = (t - cyc["time_feature_s"]) * 1e3
            y4 = np.asarray(cyc["data"].get("Areal_4highf_p", np.array([]))).squeeze()

            if y4.size > 0:
                ax.plot(t_ms, y4, color="#3E4A89", lw=1.35, label="Measured Areal_4highf_p")

            if np.isfinite(peak_p_a4[i]):
                ax.hlines(
                    peak_p_a4[i], xmin=t_ms[0], xmax=t_ms[-1],
                    colors="#9A9A9A", linestyles=":", lw=1.0,
                    label=f"Measured peak (context) = {peak_p_a4[i]:.1f} Pa",
                )

            if np.isfinite(p_cj_measured[i]):
                ax.hlines(
                    p_cj_measured[i], xmin=t_ms[0], xmax=t_ms[-1],
                    colors="#3B786D", linestyles=":", lw=1.3,
                    label=f"Measured CJ plateau = {p_cj_measured[i]:.1f} Pa",
                )

            if np.isfinite(p_cj_optical[i]):
                ax.hlines(
                    p_cj_optical[i], xmin=t_ms[0], xmax=t_ms[-1],
                    colors="#A94A4A", linestyles="--", lw=1.35,
                    label=f"Optical-estimated CJ = {p_cj_optical[i]:.1f} Pa",
                )
            else:
                ax.text(
                    0.02, 0.90,
                    "Optical CJ skipped (low light amplitude)",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#A94A4A",
                    ha="left", va="top",
                )

            ax.axvline(0.0, color="k", lw=1.0, ls=":", alpha=0.65)
            ax.set_xlabel("Time Relative to Cycle Feature (ms)")
            ax.set_ylabel("Pressure (Pa)")

            if velocity_is_outlier[i]:
                vel_text = f"v(raw->clean)={velocity_raw[i]:.1f}->{velocity[i]:.1f}"
            else:
                vel_text = f"v(clean)={velocity[i]:.1f}"

            if np.isfinite(velocity_for_cj[i]):
                vel_text += f" | v_for_CJ={velocity_for_cj[i]:.1f}"

            vel_text += f" | trend={v_trend[i]:.1f} | v_ref={v_global_ref:.1f} | conf={confidence[i]:.2f}"
            vel_text += f" | blend={expected_blend[i]:.2f} | mix_ref={mix_ref[i]:.2f} | resid={temporal_residual[i]:.2f}"
            if i < startup_cycle_count:
                vel_text += " | startup"
            if temporal_spike_flag[i]:
                vel_text += " | spike"
            if neighbor_spike_flag[i]:
                vel_text += " | nspike"
            if quality_low[i]:
                vel_text += " | qlow"
            if (quality_low[i] and (velocity[i] < v_global_ref)):
                vel_text += " | qmix"

            md_text = f"M_D={M_D[i]:.3f}" if np.isfinite(M_D[i]) else "M_D=N/A"
            if low_optical_amp[i]:
                md_text += " | low-light"

            ax.set_title(
                f"Cycle {cyc['cycle_id']} | t={cyc['time_feature_s']:.6f} s | "
                f"{md_text} | {vel_text}"
            )
            ax.grid(True, ls=":", alpha=0.35)
            ax.legend(loc="upper right", fontsize=8)

        fig.suptitle(f"Cycle-wise Areal_4highf_p vs Optical-only CJ (Cycles {k0+1}-{k1})", y=1.01)
        plt.show()

    # ---------- 图2：CJ 平台压力对比 ----------
    fig_sum, ax_sum = plt.subplots(figsize=(10, 4.8), constrained_layout=True)

    ax_sum.plot(
        cycle_time, p_cj_measured, "-o",
        lw=1.35, ms=4.5, color="#3E4A89",
        label="Measured CJ plateau (Areal_4highf_p)",
    )
    ax_sum.plot(
        cycle_time, p_cj_optical, "-s",
        lw=1.35, ms=4.5, color="#A94A4A",
        label="Optical-estimated CJ",
    )

    ax_sum.set_xlabel("Cycle Feature Time (s)")
    ax_sum.set_ylabel("Pressure (Pa)")
    ax_sum.set_title("CJ Pressure Comparison: Measured Plateau vs Optical-only Estimation")
    ax_sum.grid(True, ls=":", alpha=0.35)
    ax_sum.legend(loc="best")
    plt.show()

    # ---------- 文本诊断（逐周期） ----------
    ratio_meas_to_opt = p_cj_measured / p_cj_optical

    print("Cycle diagnostics (optical-only path, A4 plateau validation):")
    print("idx | time(s) | v_raw | v_clean | v_trend | v_for_CJ | conf | blend | mix_ref | resid | nresid | spike | nspike | qlow | p_CJ_opt(Pa) | p_CJ_meas(Pa) | meas/opt | low_light")
    for i in range(n_cycles):
        print(
            f"{i+1:>3d} | {cycle_time[i]:.6f} | "
            f"{velocity_raw[i]:>7.1f} | {velocity[i]:>7.1f} | {v_trend[i]:>7.1f} | {velocity_for_cj[i]:>8.1f} | "
            f"{confidence[i]:>4.2f} | {expected_blend[i]:>5.2f} | {mix_ref[i]:>7.2f} | {temporal_residual[i]:>5.2f} | {neighbor_residual[i]:>6.2f} | {str(bool(temporal_spike_flag[i])):>5s} | {str(bool(neighbor_spike_flag[i])):>6s} | {str(bool(quality_low[i])):>4s} | "
            f"{p_cj_optical[i]:>11.1f} | {p_cj_measured[i]:>11.1f} | {ratio_meas_to_opt[i]:>7.3f} | {str(bool(low_optical_amp[i])):>9s}"
        )

    # Validation-only summary (not used in optical estimator)
    valid_eval = np.isfinite(p_cj_optical) & np.isfinite(p_cj_measured)
    if np.any(valid_eval):
        err = p_cj_optical[valid_eval] - p_cj_measured[valid_eval]
        mae = float(np.mean(np.abs(err)))
        mape = float(np.mean(np.abs(err) / np.maximum(np.abs(p_cj_measured[valid_eval]), 1e-12)))
        print(f"Validation MAE = {mae:.1f} Pa, MAPE = {mape:.4f}")

    valid_eval = np.isfinite(p_cj_optical) & np.isfinite(p_cj_measured)
    if np.any(valid_eval):
        err = p_cj_optical[valid_eval] - p_cj_measured[valid_eval]
        mae = float(np.mean(np.abs(err)))
        mape = float(np.mean(np.abs(err) / np.maximum(np.abs(p_cj_measured[valid_eval]), 1e-12)))
    else:
        mae = np.nan
        mape = np.nan

    return {
        "cycle_time": cycle_time,
        "velocity_raw": velocity_raw,
        "velocity": velocity,
        "velocity_for_cj": velocity_for_cj,
        "velocity_is_outlier": velocity_is_outlier,
        "confidence": confidence,
        "quality_low": quality_low,
        "expected_blend": expected_blend,
        "mix_ref": mix_ref,
        "temporal_residual": temporal_residual,
        "neighbor_residual": neighbor_residual,
        "temporal_spike_flag": temporal_spike_flag,
        "neighbor_spike_flag": neighbor_spike_flag,
        "M_D": M_D,
        "p_cj_optical": p_cj_optical,
        "p_cj_measured": p_cj_measured,
        "peak_p_a4": peak_p_a4,
        "low_optical_amp": low_optical_amp,
        "validation_mae_pa": mae,
        "validation_mape": mape,
    }


def run_optical_to_cj_pipeline(Areal_1time, Areal_2light, areal_data=None, build_options=None):
    """Run the frozen optical->velocity->CJ pipeline end-to-end.

    Parameters
    ----------
    Areal_1time : array-like
        Time axis in seconds.
    Areal_2light : array-like
        Optical channels, using columns 0 and 1.
    areal_data : dict, optional
        Mapping of Areal_* arrays for slice packaging/validation.
    build_options : dict, optional
        Overrides on top of frozen build defaults.

    Returns
    -------
    dict
        `{"cycle_result": ..., "cj_result": ...}`.
    """
    opts = get_frozen_build_options()
    if build_options:
        merged = dict(opts)
        for k, v in build_options.items():
            if k == "velocity_kwargs" and isinstance(v, dict):
                vv = dict(opts.get("velocity_kwargs", {}))
                vv.update(v)
                merged[k] = vv
            else:
                merged[k] = v
        opts = merged

    cycle_result = build_detonation_cycle_dataset(
        Areal_1time,
        Areal_2light,
        areal_data=areal_data,
        **opts,
    )
    cj_result = estimate_optical_cj_from_cycle_result(cycle_result)
    return {"cycle_result": cycle_result, "cj_result": cj_result}
