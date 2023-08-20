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

# TRUE AERODYNAMIC PARAMETER VALUES
theta_real = np.array([202.1, -246.3, -37.56, 0, 71.51 * M, 10.01 * M, 10.81 * M + 51.27])

def get_CABS_control(dt,t,c1,c2,gamma,plot=False):
    yr = reference_signal(t)
    yr_dot = np.gradient(yr,dt)

    #initialise variables
    x1,x2 = 0.,0.
    x2c,x2c_dot = 0.,0.
    chi1,chi2 = 0.,0.
    theta_est = 1.05*np.array(theta_real)
    theta_est_f2 = theta_est[:-1].reshape((6,1))
    theta_est_g2 = theta_est[-1].reshape((1,1))
    u,u_dot = 0.,0.

    x = []
    theta = []

    for index, t_i in enumerate(t):
        bz, bm, Cz, Cm = aerodynamic_coef(x1, M)

        #get on-line measured acceleration
        Az = (q_bar*S/(m))*(Cz + bz*u)
        f1 = Az/V_T

        #regressor functions for uncertainties in aerodynamic model
        phi_f2, phi_g2 = phi(x1)
        f2 = phi_f2@theta_est_f2
        g2 = phi_g2@theta_est_g2

        #get outerloop tracking error
        z1 = x1 - yr[index]

        #get tuning function (alpha)
        alpha = -c1*z1-f1 +yr_dot[index]
        x2c_0 = alpha - chi2

        #command filtered virtual control
        x2c,x2c_dot = command_filter_1(dt,np.array([[x2c],[x2c_dot]]),[x2c_0])
        chi1_dot = -c1*chi1+(x2c-x2c_0)
        z1_bar = z1 - chi1

        #command filtered u
        z2 = x2 - x2c
        u0 = ((1/g2)*(-c2*z2-z1_bar-f2+x2c_dot))[0]
        u,u_dot = command_filter_2(dt,np.array([[u],[u_dot]]),u0)
        chi2_dot = (-c2*chi2 +g2*(u-u0))[0]
        z2_bar = z2 - chi2

        #updating estimations and state values
        theta_est_f2_dot = (phi_f2*(gamma)*z2_bar).reshape((6,1))
        theta_est_g2_dot = (phi_g2*(gamma)*z2_bar*u).reshape((1,1))
        theta_est_f2 = theta_est_f2 + theta_est_f2_dot*dt
        theta_est_g2 = theta_est_g2 + theta_est_g2_dot*dt

        x1_dot = f1 + x2
        x2_dot = (phi_f2@theta_est_f2+(phi_g2@theta_est_g2)*u)[0]
        x1 = x1 + x1_dot*dt
        x2 = (x2 + x2_dot*dt)[0]
        chi1 = chi1 + chi1_dot*dt
        chi2 = (chi2 + chi2_dot*dt)[0]

        x.append([x1 * 180 / np.pi, x2 * 180 / np.pi, z1, z2, z1_bar, z2_bar, u * 180 / np.pi,
                  0.5 * z1_bar ** 2 + 0.5 * z2_bar ** 2])
        theta_est = np.append(theta_est_f2.flatten(),theta_est_g2.flatten())
        theta.append(theta_est)


    x = pd.DataFrame(x, columns=["x1", "x2", "z1", "z2", "z1_bar", "z2_bar", "u", "V2"], index=t).astype(float)
    theta = pd.DataFrame(theta,index = t).astype(float)

    if plot:
        fig = result_plot(t,yr,x)
        fig.suptitle(f"CABS controller (c1 = {c1}, c2 = {c2}, gamma = {gamma})")

        fig = CABS_params_plot(t,theta,theta_real)
        fig.suptitle(f"CABS parameter estimation (c1 = {c1}, c2 = {c2}, gamma = {gamma})")


    from sklearn.metrics import r2_score
    try:
        R2 = r2_score(x["x1"], yr*180/np.pi)
    except:
        R2  = np.nan
    return R2

def get_CABS_sensitivity(dt,t,c_values,gamma_values):
    df = pd.DataFrame([],index = gamma_values, columns = c_values)
    for c in c_values:
        for gamma in gamma_values:
            df.loc[gamma,c] = get_CABS_control(dt,t,c,c,gamma)

    import seaborn as sns
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    sns.heatmap(df.astype(float).iloc[::-1,:],ax=ax, annot=True,cmap='RdYlGn')
    ax.set_ylabel("Gamma")
    ax.set_xlabel("C = C1 = C2")
    fig.suptitle("CABS R-squared value of tracking")


