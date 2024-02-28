import json
import os
import streamlit as st
from math import sqrt
from statistics import mean
import pandas as pd
import altair as alt

def is_valid_car(w, h):
    if w<10 or h<5:
        # Too small
        return False
    if w>150 or h>50:
        # Too large
        return False
    return True

@st.cache_data
def compute_data():
    path_to_json = "data/"

    min_velocity_per_file = []
    file_specific_data = {}

    for file_name in [file for file in os.listdir(path_to_json)]:
        with open(path_to_json + file_name) as json_file:
            data = json.load(json_file)
            valid_file = True

            # Compute basic series of data that correspond to time points in the video.
            x_points = []
            y_points = []
            times = []

            for r in data:
                if r["frame"] in times:
                    # Multiple detections in the same frame - skip video.
                    valid_file = False
                if r["class"] != 2:
                    # Not a car
                    continue
                
                w = abs(r["x"] - r["x2"])
                h = abs(r["y"] - r["y2"])
                x = (r["x"] + r["x2"])/2
                y = (r["y"] + r["y2"])/2
                t = r["frame"]

                if not is_valid_car(w, h):
                    # Check for sizes that are way off from expected and skip that frame.
                    continue

                x_points.append(x)
                y_points.append(y)
                times.append(t)

            if len(times) < 10:
                # Skip the video if there weren't enough detected frames.
                valid_file = False

            if times and max(times) > 400:
                # Skip the video if the detection is at the very end since it likely gets cut off.
                valid_file = False
            
            if times and min(times) < 50:
                # Skip the video if the detection is at the very beginning since it's likely missing the start.
                valid_file = False
            
            time_gap_threshold = 25
            for t_idx in range(len(times)):
                if t_idx > 0 and times[t_idx] - times[t_idx-1] > time_gap_threshold:
                    # Skip the video if it has a signfiicatn gap between detection times since we don't know how to handle these gaps reliably.
                    valid_file = False

            if not valid_file:
                continue
            
            # Compute derived properties like velocity and distance travelled.
            velocity = []
            distance = []
            smooth_velocity = []
            for t_idx in range(len(times)):
                dx = x_points[t_idx] - x_points[t_idx - 1]
                dy = y_points[t_idx] - y_points[t_idx - 1]
                dt = times[t_idx] - times[t_idx - 1]
                v_x = dx/dt
                v_y = dy/dt
                v = sqrt(pow(v_x, 2) + pow(v_y, 2))
                d = sqrt(pow(dx, 2) + pow(dy, 2))

                velocity.append(v)
                distance.append(d)
                if t_idx > 3:
                    smooth_velocity.append(mean(velocity[-3:-1]))
            
            # Compute and track data across files
            min_velocity_per_file.append((min(smooth_velocity), file_name))
            file_specific_data[file_name] = {
                "x": x_points,
                "y": y_points,
                "v": velocity,
                "d": distance,
                "t": times
            }
    
    sorted_min_velocity_per_file = sorted(min_velocity_per_file)
    min_velocities = [t[0] for t in sorted_min_velocity_per_file]
    file_names = [t[1] for t in sorted_min_velocity_per_file]

    return min_velocities, file_names, file_specific_data

def plot_scatter(x_data, y_data, x_title, y_title, x_domain, y_domain, title):
    title = alt.TitleParams('Velocity of the car through video', anchor='middle')
    plot = alt.Chart(pd.DataFrame({
            x_title: x_data,
            y_title: y_data,
        })).mark_circle(size=60).encode(
            alt.X(f'{x_title}:Q').scale(domain=x_domain),
            alt.Y(f'{y_title}:Q').scale(domain=y_domain),
        tooltip=[x_title, y_title], 
    ).properties(
        title=title
    ).interactive()
    st.altair_chart(plot, use_container_width=True)