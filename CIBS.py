import numpy as np
import pandas as pd
from sim_model import *

#REST PARAMETERS
g = 32.2
W = 470
V_T = 3150
Iyy = 182.5
q_bar = 6132.8
S = 0.55
c_bar = 0.75
m = W/g
M = 2.7

def get_CIBS_control(dt,t,c1,c2,dgd,plot = False):
    # tracking signal and derivatives
    yr = reference_signal(t)
    yr_dot = np.gradient(yr, dt)
    x1,x2 = 0.,0.
    x2c, x2dot, x2c_dot = 0., 0., 0.
    chi1, chi2 = 0., 0.
    u, u_dot = 0.,0.
    x = []

    for index, t_i in enumerate(t):
        bz, bm, Cz, Cm = aerodynamic_coef(x1, M)
        f1,g1 = (q_bar * S / (m * V_T)) * Cz , 1.
        f2,g2 = (q_bar * S * c_bar / (Iyy)) * Cm, (q_bar * S * c_bar / (Iyy)) * bm

        #get on-line measured angular acceleration
        x2dot_0 = f2 + g2*u

        #get outerloop tracking error
        z1 = x1 - yr[index]
        z1_bar = z1 - chi1
        z2 = x2 - x2c
        z2_bar = z2 - chi2

        #get tuning function (alpha)
        alpha = -c1*z1 - f1 +yr_dot[index]
        x2c_0 = alpha - chi2

        #command filtered virtual control
        x2c,x2c_dot = command_filter_1(dt,np.array([[x2c],[x2c_dot]]),[x2c_0])

        #dynamic surface control z1-sys
        chi1_dot = -c1*chi1+(x2c-x2c_0)
        chi1 = chi1 + chi1_dot*dt

        #command filtered control
        u_0 = u                                                     #current u
        du0 = (1/g2)*(-c2*z2-z1_bar-x2dot_0+x2c_dot)                #incremental u = raw control (u0) - current u (u_0)
        u0 = u_0 + du0                                              #raw control law
        u,u_dot = command_filter_2(dt,np.array([[u],[u_dot]]),[u0]) #command filtered control law

        #dynamic surface control z2-sys
        chi2_dot = -c2*chi2+g2*(u-u0)
        chi2 = chi2 + chi2_dot*dt

        #update states
        x1_dot = f1 + x2
        x2dot = x2dot_0 + g2*(u-u_0)
        x1 = x1 + x1_dot*dt
        x2 = x2 + x2dot*dt

        #store data
        x.append([x1 * 180 / np.pi, x2*180/np.pi, z1, z2, z1_bar, z2_bar, u * 180 / np.pi, 0.5*z1_bar**2 + 0.5*z2_bar**2])

    x = pd.DataFrame(x, columns=["x1","x2", "z1", "z2", "z1_bar", "z2_bar", "u","V2"], index=t).astype(float)

    if plot:
        fig = result_plot(t,yr,x)
        fig.suptitle(f"CIBS controller (c1 = {c1}, c2 = {c2}, \u0394Gd/Gd = {dgd})")

    from sklearn.metrics import r2_score
    try:
        R2 = r2_score(x["x1"], yr*180/np.pi)
    except:
        R2  = np.nan
    return R2,x["u"].abs().max()


def get_CIBS_robustness(dt,t,c1,c2,dgd_values):
    R2 = []
    U = []
    for dgd in dgd_values:
        r2,u_max = get_CIBS_control(dt,t,c1,c2,dgd)
        R2.append(r2)
        U.append(u_max)

    fig = CIBS_robust_plot(dgd_values,R2,U)
    fig.suptitle(f"CIBS controller robustness (c1 = {c1}, c2 = {c2})")

def get_CIBS_sensitivity(dt_values,c_values):
    df = pd.DataFrame([],index = dt_values, columns = c_values)
    for dt in dt_values:
        t = np.arange(0,50,dt)
        for c in c_values:
            df.loc[dt,c] = get_CIBS_control(dt,t,c,c,0.)
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    sns.heatmap(df.astype(float).iloc[::-1,:],ax=ax, annot=True,cmap='RdYlGn')
    ax.set_ylabel("time step size [s]")
    ax.set_xlabel("C = C1 = C2")
    fig.suptitle("CIBS R-squared value of tracking")



