import numpy as np
from inner_loop_control import CABS_control,CIBS_control
from plot import CABS_plotting,CIBS_plotting,CIBS_robustness_plot,trajectory_plot
from CABS import *
from CIBS import *
import matplotlib.pyplot as plt

#time arrays
dt = 0.01
T = 50
t = np.arange(0,T,dt)
c1,c2 = 2,2

#CIBS controller
dgd = 0.1
get_CIBS_control(dt,t,c1,c2,dgd,plot=True)
plt.show()
dgd_values = np.arange(0,1,0.1)
get_CIBS_robustness(dt,t,c1,c2,dgd_values)
c_values,dt_values = [.1,0.5,1,2,3,4,5],[0.001,0.005,0.01,0.1]
get_CIBS_sensitivity(dt_values,c_values)
plt.show()
quit()

#CABS controller
gamma = 3
get_CABS_control(dt,t,c1,c2,gamma,plot=True)
plt.show()
c_values,gamma_values = [.1,0.5,1,1.5,2,2.5,3],[.1,0.5,1,1.5,2,2.5]
get_CABS_sensitivity(dt,t,c_values,gamma_values)
plt.show()
quit()

gamma = .3
get_CABS_control(dt,t,c1,c2,gamma,plot=True)
dgd = .10
get_CIBS_control(dt,t,c1,c2,dgd,plot=True)
get_CABS_sensitivity(dt,t,c_values,gamma_values)


#Command Filtered Incremental Backstepping
df1 = CIBS_control(t,c1,c2,0.1)
CIBS_plotting(df1,c1,c2,0.1)
plt.show()
quit()

CIBS_robustness_plot(t,c1,c2,[0.1,0.2,0.4,0.5])

#Command Filtered Adaptive Backstepping
gamma = 10
df2 = CABS_control(t,c1,c2,gamma)
CABS_plotting(df2,c1,c2,gamma)

plt.show()
trajectory_plot(df_CIBS=df1,df_CABS=df2)

import matplotlib.pyplot as plt
plt.show()

