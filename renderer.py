import numpy as np
import matplotlib.pyplot as plt
import fastf1 as f1
import pandas as pd

import time


class TrackRenderer:
    def __init__(self, year: int, track_name: str, session_type: str):
        self.session = f1.get_session(year, track_name, session_type)
        self.session.load()

        self.lap = self.session.laps.pick_fastest()
        self.pos = self.lap.get_pos_data()

        self.circuit_info = self.session.get_circuit_info()

    def rotate(self, xy, *, angle):
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                    [-np.sin(angle), np.cos(angle)]])
        
        return np.matmul(xy, rotation_matrix)
    
    def draw_track(self, ax) -> None:
    
        track = self.pos.loc[:, ("X", "Y")].to_numpy()

        track_angle = self.circuit_info.rotation / 180 * np.pi
        rotated_track = self.rotate(track, angle=track_angle)

        ax.plot(
            rotated_track[:, 0],
            rotated_track[:, 1],
            color="black",
            linewidth=2
        )

        offset_vector = [500, 0]

        for _, corner in self.circuit_info.corners.iterrows():
            txt = f"{corner['Number']}{corner['Letter']}"

            offset_angle = corner["Angle"] / 180 * np.pi
            offset_x, offset_y = self.rotate(
                offset_vector,
                angle=offset_angle
            )

            text_x = corner["X"] + offset_x
            text_y = corner["Y"] + offset_y

            text_x, text_y = self.rotate(
                [text_x, text_y],
                angle=track_angle
            )

            track_x, track_y = self.rotate(
                [corner["X"], corner["Y"]],
                angle=track_angle
            )

            ax.scatter(text_x, text_y, color="grey", s=140)
            ax.plot(
                [track_x, text_x],
                [track_y, text_y],
                color="grey"
            )

            ax.text(
                text_x,
                text_y,
                txt,
                va="center_baseline",
                ha="center",
                size="small",
                color="white"
            )

    def draw_drivers(self, ax, session_time: float):
        track_angle = self.circuit_info.rotation / 180 * np.pi
        drivers = []
        
        for _, driver in self.session.results.iterrows():
            driver_code = driver["Abbreviation"]
            laps = self.session.laps.pick_drivers(driver_code)
            if laps.empty:
                continue

            target_time = pd.to_timedelta(session_time, unit="s")

            active_lap = laps[
                (laps["LapStartTime"] <= target_time) &
                (laps["LapStartTime"] + laps["LapTime"] >= target_time)
            ]

            if active_lap.empty:
                continue

            lap = active_lap.iloc[0]

            
            pos_data = lap.get_pos_data()
            

            telemetry_time = (pos_data["SessionTime"].dt.total_seconds())

            idx = (telemetry_time - session_time).abs().idxmin()

            x = pos_data.loc[idx, "X"]
            y = pos_data.loc[idx, "Y"]

            x, y = self.rotate([x, y], angle=track_angle)

            point = ax.scatter(x, y, s=80)

            ax.text(x, y, driver_code, fontsize=8)

            drivers.append(point)

        return drivers
    
    def draw(self, speed: int) -> None:
        plt.ion()

        fig, ax = plt.subplots()

        self.draw_track(ax)

        ax.set_title(self.session.event["Location"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("equal")

        drivers = None

        t = 1

        while True:

            # remove previous car positions
            if drivers:
                for d in drivers:
                    d.remove()

            drivers = self.draw_drivers(
                ax,
                t * speed
            )

            fig.canvas.draw()
            fig.canvas.flush_events()

            plt.pause(1)

            t += 1