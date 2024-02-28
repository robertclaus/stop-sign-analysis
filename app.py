import math
import streamlit as st
from app_utils import compute_data, plot_scatter

min_velocities, sorted_file_names, file_specific_data = compute_data()
car_ids = range(len(min_velocities))

st.title("Analysis of stopping at a stop sign")
st.markdown("This analysis shows how close to a stop cars came at an actual stop sign. You can see that very few cars actually hit a zero velocity. Most slowed to a slower speed, but many went through the interesction at a considerable pace.")

plot_scatter(
    x_data=car_ids, 
    y_data=min_velocities, 
    x_title="Car Id", 
    y_title="Minimum Velocity (px / frame)", 
    x_domain=(0, len(car_ids)), 
    y_domain=(0, math.ceil(max(min_velocities))), 
    title='Stopping Velocity of Cars at a Stop Sign'
)

st.divider()

st.subheader("Individual car data")

car_id = st.selectbox("Car to Inspect", car_ids)
file = sorted_file_names[car_id]

if file:
    col1, col2 = st.columns(2)
    with col1:
        plot_scatter(
            x_data=file_specific_data[file]["x"], 
            y_data=file_specific_data[file]["y"], 
            x_title="X Position (px)", 
            y_title="Y Position (px)", 
            x_domain=(0,512), 
            y_domain=(0, 288), 
            title='Position of car in video'
        )

    with col2:
        plot_scatter(
            x_data=file_specific_data[file]["t"], 
            y_data=file_specific_data[file]["v"], 
            x_title="Time (frames)", 
            y_title="Velocity (px / frame)", 
            x_domain=(0,450), 
            y_domain=(0, 10), 
            title='Velocity of the car through video'
        )
    st.video(f"output/{file[0:-4]}", format="video/mp4", start_time=int(file_specific_data[file]["t"][0]*(60/450)))
