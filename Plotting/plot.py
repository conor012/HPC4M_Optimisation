"""Plotting Optimiser Trajectories
Given a trajectory csv file, will plot (and optionally save) the  trajectory of the
 optimiser (without a surface)
"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("..\\trajectory1.csv", sep=",", header=None)
df = df.values

# Plot initial
fig, ax = plt.subplots()
ax.set(xlim=[min(df[:, 0]), max(df[:, 0])], ylim=[min(df[:, 1]), max(df[:, 1])])
(approx_line,) = ax.plot(df[0, 0], df[0, 1], alpha=0.5)
(dot,) = ax.plot(df[0, 0], df[0, 1], "o")
ax.plot()


def animate(i):
    dot.set_data(df[i - 1, 0], df[i - 1, 1])
    approx_line.set_data(df[max(0, (i - 50)) : i, 0], df[max(0, (i - 50)) : i, 1])
    return approx_line, dot


ani = animation.FuncAnimation(
    fig, animate, frames=len(df[:, 0]), interval=50, blit=True
)
Writer = animation.writers["ffmpeg"]
writer = Writer(fps=15, bitrate=-1)
# ani.save("trajectory.mp4", writer=writer)

plt.show()
