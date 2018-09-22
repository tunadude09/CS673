from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import time

from matplotlib.widgets import Slider, Button, RadioButtons

def signal(thresh, dist_exp, dist_scale):
    value = 1
    return value / ((dist_scale*t)**dist_exp + 1)*(1/(1+np.exp(-(thresh*dist_scale*t-5))))

axis_color = 'lightgoldenrodyellow'
fig = plt.figure()

# Draw the plot
ax = fig.add_subplot(111)
fig.set_size_inches(20,10)
fig.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
thresh_0 = 20.0
dist_exp_0 = 0.8
dist_scale_0 = 100.0

[line] = ax.plot(t, signal(thresh_0, dist_exp_0, dist_scale_0), linewidth=2, color='red')
ax.set_xlim([0, 0.2])
ax.set_ylim([0.0, 0.7])


# # Draw the plot
# ax = fig.add_subplot(122)
# fig.set_size_inches(20,10)
# fig.subplots_adjust(left=0.25, bottom=0.25)
# #t = np.arange(0.0, 1.0, 0.001)
# #thresh_0 = 20.0
# #dist_exp_0 = 0.8
# #dist_scale_0 = 100.0

# [line2] = ax.plot(t, signal(thresh_0, dist_exp_0, dist_scale_0), linewidth=2, color='red')
# ax.set_xlim([0, 0.2])
# ax.set_ylim([0.0, 0.7])




# Add two sliders for tweaking the parameters
threshold_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axis_color)
threshold_slider = Slider(threshold_slider_ax, 'thres', 1.0, 50.0, valinit=thresh_0)
dist_exp_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axis_color)
dist_exp_slider = Slider(dist_exp_slider_ax, 'd_e', 0.2, 20.0, valinit=dist_exp_0)
dist_scaler_slider_ax  = fig.add_axes([0.25, 0.05, 0.65, 0.03], axisbg=axis_color)
dist_scaler_slider = Slider(dist_scaler_slider_ax, 'd_s', 20.0, 200.0, valinit=dist_scale_0)

def sliders_on_changed(val):
    # time.sleep(0.5)
    line.set_ydata(signal(threshold_slider.val, dist_exp_slider.val, dist_scaler_slider.val))
    fig.canvas.draw_idle()
threshold_slider.on_changed(sliders_on_changed)
dist_exp_slider.on_changed(sliders_on_changed)
dist_scaler_slider.on_changed(sliders_on_changed)


# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    freq_slider.reset()
    amp_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)



plt.show()