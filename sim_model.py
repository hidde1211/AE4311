import numpy as np
import pandas as pd

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
def aerodynamic_coef(alpha,M):
    bz = 1.62*M-6.72
    bm = 10.81*M+51.27

    phi_z1 = -288.7*(alpha**3)+50.32*alpha*np.abs(alpha)-23.89*alpha
    phi_z2 = -13.53*alpha*np.abs(alpha)+4.185*alpha
    phi_m1 = 202.1*(alpha**3)-246.3*alpha*np.abs(alpha)-37.56*alpha
    phi_m2 = 71.51*alpha*np.abs(alpha)+10.01*alpha

    Cz = phi_z1 + phi_z2*M
    Cm = phi_m1 + phi_m2*M
    return bz,bm,Cz,Cm

def aerodyn_plot():
    alpha_range = (np.pi/180)*np.arange(-10,10+0.1,0.1)

    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,2)

    df = pd.DataFrame([],columns = ["bz","bm","Cz","Cm"])
    M0 = 2
    for index,alpha in enumerate(alpha_range):
        df.loc[index,:] = aerodynamic_coef(alpha,M0)

    ax[0].plot((180/np.pi)*alpha_range,df["bz"],c = "tab:blue",label=f"bz (M = {M0})")
    ax[0].plot((180/np.pi)*alpha_range,df["Cz"],c = "tab:orange",label=f"Cz (M = {M0})")

    ax[1].plot((180/np.pi)*alpha_range,df["bm"],c = "tab:blue",label=f"bm (M = {M0})")
    ax[1].plot((180/np.pi)*alpha_range,df["Cm"],c = "tab:orange",label=f"Cm (M = {M0})")

    M0 = 3
    for index,alpha in enumerate(alpha_range):
        df.loc[index,:] = aerodynamic_coef(alpha,M0)

    ax[0].plot((180/np.pi)*alpha_range,df["bz"],c = "tab:blue",label=f"bz (M = {M0})",linestyle = "--")
    ax[0].plot((180/np.pi)*alpha_range,df["Cz"],c = "tab:orange",label=f"Cz (M = {M0})",linestyle = "--")

    ax[1].plot((180/np.pi)*alpha_range,df["bm"],c = "tab:blue",label=f"bm (M = {M0})",linestyle = "--")
    ax[1].plot((180/np.pi)*alpha_range,df["Cm"],c = "tab:orange",label=f"Cm (M = {M0})",linestyle = "--")

    ax[0].legend()
    ax[0].set_ylabel("Aerodynamic force coefficient []")
    ax[1].set_ylabel("Aerodynamic moment coefficient []")
    ax[0].set_xlabel("angle of attack [deg]")
    ax[1].set_xlabel("angle of attack [deg]")

    ax[1].legend()
    fig.suptitle("Non-linear aerodynamic derivatives")
    plt.show()

def reference_signal(t):
    return 13*np.pi/180*np.cos(0.5*t)

def command_filter_1(dt,x,x0):
    wn,damp = 20,0.7
    # Define the system matrices
    A = np.array([[0., 1], [-wn**2, -2*damp*wn]])
    B = np.array([[0], [wn**2]])

    x_dot = (A@x + (B@x0).reshape((2,1))).flatten()
    y1,y2 = x[0] + x_dot[0] * dt, x[1] + x_dot[1]*dt
    return y1[0],y2[0]

def command_filter_2(dt,x,x0):
    wn,damp = 20,0.7
    # Define the system matrices
    A = np.array([[0., 1], [-wn**2, -2*damp*wn]])
    B = np.array([[0], [wn**2]])

    M = 12*np.pi/180

    Smx = x0
    if Smx[0]>M:
        Smx = [M]

    if Smx[0]<-M:
        Smx = [-M]

    x_dot = (A@x + (B@Smx).reshape((2,1))).flatten()
    y1,y2 = x[0] + x_dot[0] * dt, x[1] + x_dot[1]*dt
    return y1[0],y2[0]

def get_Az(x1,M,u):
    bz,bm,Cz,Cm = aerodynamic_coef(x1,M)
    Az = (q_bar*S/(m*V_T*V_T))*(Cz+bz*u)
    return Az

def phi(x1):
    phi_f2 = (q_bar * S * c_bar / (Iyy))*np.array([x1**3,x1*np.abs(x1),x1,x1**3,x1*np.abs(x1),x1])
    phi_g2 = (q_bar * S * c_bar / (Iyy))*np.array([1])
    return phi_f2.reshape((1,6)),phi_g2.reshape((1,1))

def result_plot(t,yr,x):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(2,2)
    x["x1"].plot(ax=ax[0,0],label = "x1")
    ax[0,0].plot(t,yr*180/np.pi,label = "yr")
    ax[0,0].legend()
    ax[0,0].set_xlabel("time [s]")
    ax[0,0].set_ylabel("angle of attack [deg]")

    x["x2"].plot(ax=ax[1,0],label = "x2")
    ax[1,0].legend()
    ax[1,0].set_xlabel("time [s]")
    ax[1,0].set_ylabel("pitch rate [deg/s]")

    x["u"].plot(ax=ax[0,1])
    ax[0,1].set_xlabel("time [s]")
    ax[0,1].set_ylabel("control input [deg]")

    x[["z1","z2","z1_bar","z2_bar"]].plot(ax=ax[1,1])
    ax[1,1].set_xlabel("time [s]")
    ax[1,1].set_ylabel("errors []")
    ax[1,1].legend()
    return fig


def CABS_params_plot(t,theta,true_theta):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    param = [r"$\theta$ f2,1",r"$\theta$ f2,2",r"$\theta$ f2,3",r"$\theta$ f2,4",r"$\theta$ f2,5",r"$\theta$ f2,6",r"$\theta$ g2,1"]
    for index,theta_i in enumerate(true_theta):
        ax.plot(t,(np.ones(len(t))*theta_i-theta.iloc[:,index])/theta_i,label=param[index])
    ax.legend()
    ax.set_ylabel("parameter estimation error [%]")
    ax.set_xlabel("time [s]")
    return fig

def CIBS_robust_plot(x,y,y2):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,2)
    ax[0].plot(x,y)
    ax[0].scatter(x,y)
    ax[0].set_ylabel("x1 tracking r-squared value [-]")
    ax[0].set_xlabel("$\Delta$Gd/Gd [%]")

    ax[1].plot(x,y2)
    ax[1].scatter(x,y2)
    ax[1].set_ylabel("max absolute control deflection [deg]")
    ax[1].set_xlabel("$\Delta$Gd/Gd [%]")
    return fig



