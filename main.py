import fastf1 as f1
import matplotlib.pyplot as plt
import numpy as np


from renderer import TrackRenderer

if __name__ == "__main__":
    tr = TrackRenderer(2023, "Brasil", "R")
    tr.draw(32)
    

