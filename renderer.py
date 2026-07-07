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

        self.driver_cache = {}
        self.cache_driver_positions()

    def init_drivers(self, ax):
        track_angle = self.circuit_info.rotation / 180 * np.pi

        self.driver_artists = {}

        for driver_code, pos_data in self.driver_cache.items():

            
            point = ax.scatter([], [], s=80)

            label = ax.text(0, 0, driver_code, fontsize=8, ha="center")

            self.driver_artists[driver_code] = {
                "point": point,
                "label": label,
                "data": pos_data
            }

    def interpolate_position(self, pos_data, session_time):

        times = pos_data["Time"].values

        index = np.searchsorted(times, session_time)

        # outside telemetry range
        if index == 0 or index >= len(times):
            return None, None

        t1 = times[index - 1]
        t2 = times[index]

        x1 = pos_data.iloc[index - 1]["X"]
        y1 = pos_data.iloc[index - 1]["Y"]

        x2 = pos_data.iloc[index]["X"]
        y2 = pos_data.iloc[index]["Y"]

        # interpolation factor
        ratio = (session_time - t1) / (t2 - t1)

        x = x1 + ratio * (x2 - x1)
        y = y1 + ratio * (y2 - y1)

        return x, y

    def cache_driver_positions(self) -> None:
        for _, driver in self.session.results.iterrows():
            driver_code = driver["Abbreviation"]

            laps = self.session.laps.pick_drivers(driver_code)

            if laps.empty:
                continue

            telemetry = []

            for _, lap in laps.iterrows():
                pos_data = lap.get_pos_data()

                if pos_data.empty:
                    continue
                
                pos_data = pos_data[["SessionTime", "X", "Y"]].copy()

                pos_data["Time"] = (pos_data["SessionTime"].dt.total_seconds())

                telemetry.append(pos_data)

            if telemetry:
                self.driver_cache[driver_code] = pd.concat(telemetry, ignore_index=True)

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

    def update_drivers(self, session_time):

        track_angle = self.circuit_info.rotation / 180 * np.pi
        for driver_code, artist in self.driver_artists.items():

            pos_data = artist["data"]

            x, y = self.interpolate_position(pos_data, session_time)

            if x is None:
                continue

            x, y = self.rotate([x, y], angle=track_angle)

            # move car
            artist["point"].set_offsets([[x, y]])

            # move name
            artist["label"].set_position((x, y))
        
    def draw(self):

        plt.ion()

        fig, ax = plt.subplots()

        self.draw_track(ax)
        self.init_drivers(ax)

        ax.set_title(self.session.event["Location"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("equal")

        times = sorted(set(np.concatenate([data["Time"].values for data in self.driver_cache.values()])))

        for t in times:

            self.update_drivers(t)

            fig.canvas.draw()
            fig.canvas.flush_events()

            plt.pause(0.05)