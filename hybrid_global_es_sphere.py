from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['text.usetex'] = True

e1 = np.array([1,0,0])
e2 = np.array([0,1,0])
e3 = np.array([0,0,1])

def hodge(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def cost_func(x):
    return 1-x.dot(np.array([0,0,1]))

def synerigistic_diffeomorphism(x,q,gamma):
    J = cost_func(x)
    if J<=gamma:
        z = x
    else:
        if q==1:
            z = expm(+0.5*0.5*hodge(e1+e2)/np.sqrt(2)*(J-gamma)**2).dot(x)
        elif q==2:
            z = expm(-0.5*0.5*hodge(e1+e2)/np.sqrt(2)*(J-gamma)**2).dot(x)
        else:
            z = x
    return z

def decision_func(x,gamma):
    J1 = cost_func(synerigistic_diffeomorphism(x,1,gamma))
    J2 = cost_func(synerigistic_diffeomorphism(x,2,gamma))
    
    J = np.array([J1,J2])
    
    return np.min(J)

def jump_condition(t,X,gamma,delta):
    Jq = cost_func(synerigistic_diffeomorphism(X[0:3],X[3],gamma))
    m  = decision_func(X[0:3],gamma)
    return Jq-m-delta

def jump_map(X,gamma):
    J1 = cost_func(synerigistic_diffeomorphism(X[0:3],1,gamma))
    J2 = cost_func(synerigistic_diffeomorphism(X[0:3],2,gamma))
    
    J = np.array([J1,J2])
    
    X[3] =np.where(J==np.min(J))+1
    X[4] = X[4] + 1
    return X

def flow_map(t,X,omegas,epsilon,gamma):
    x = X[0:3]
    q = X[3]
    
    K = 10
    
    J = cost_func(synerigistic_diffeomorphism(x,q,gamma))
    u1 = np.sqrt(2*np.pi*omegas[0])*np.cos(2*np.pi*omegas[0]*t/epsilon**2+K*J)/np.sqrt(K)/epsilon
    u2 = np.sqrt(2*np.pi*omegas[1])*np.cos(2*np.pi*omegas[1]*t/epsilon**2+K*J)/np.sqrt(K)/epsilon
    u3 = np.sqrt(2*np.pi*omegas[2])*np.cos(2*np.pi*omegas[2]*t/epsilon**2+K*J)/np.sqrt(K)/epsilon

    dxdt = u1*hodge(e1).dot(x) + u2*hodge(e2).dot(x) + u3*hodge(e3).dot(x)
    return np.concatenate((dxdt,np.array([0]),np.array([0])))

if __name__ == '__main__':
    
    gamma = 1
    delta = 1/8
    epsilon = 1/np.sqrt(5)
    omegas = np.array([1.00,2.00,1.50])
    
    x0 = -e3

    Xi = np.concatenate((x0,[2],[0]))
    Ti = 0
    Tf = 12

    Tout  = np.array([Ti])
    Xout  = Xi.reshape(([5,1]))

    Teout = np.array([])
    Xeout = np.array([[]])

    jump_trigger = lambda t, X: jump_condition(t,X,gamma,delta)
    jump_trigger.direction = 1
    jump_trigger.terminal = True

    if jump_trigger(0,Xi)>0:
        Xi = jump_map(Ti,Xi,gamma)
        print(Xi)

    while Tout[-1]<Tf:

        T = np.linspace(Ti,Tf,int(1e4))
        sol = solve_ivp(lambda t, X: flow_map(t, X, omegas,epsilon,gamma), [T[0], T[-1]], Xi, 'RK45', T, dense_output=True,events=jump_trigger)
        Tout = np.append(Tout,sol.t)
        Xout = np.append(Xout,sol.y,axis=1)
        print(sol.t_events)
        if np.any(sol.t_events[0]):
            Ti = sol.t_events[0][0]
            Xi = sol.y_events[0][0]

            Teout = np.append(Teout,sol.t_events[0])
            Xeout = np.append(Xeout,sol.y_events[0])
            
            print(Xi)
            Xi = jump_map(Ti,Xi,gamma)
            print(Xi)

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    norm = cm.colors.Normalize(vmin=np.nanmin(1-z), vmax=np.nanmax(1-z)+0.125)
    cdata = cm.gray(norm(1-z))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    
    cax1 = ax1.plot_surface(x, y, z, rstride=1, cstride=1, alpha=1, cmap=cm.gray, facecolors=cdata, linewidth=0.5,zorder=1,antialiased=False, norm=norm)
    plt.colorbar(cax1, orientation='horizontal',fraction = 0.05, norm = norm)
    ax1.plot3D(Xout[0,:],Xout[1,:],Xout[2,:],linewidth=1.5,zorder=10,color=mcolors.XKCD_COLORS[f'xkcd:red'].upper(),label='$x(t)$')
    
    ax1.plot(Xout[0, :], Xout[2, :],zdir='y',color=mcolors.XKCD_COLORS[f'xkcd:red'].upper(),zs=+1.5)
    ax1.scatter(Xi[0], Xi[2], zdir='y',marker='o', s=200, color='black', zs=+1.5,label='$x(0)$')
    ax1.scatter(0, 1, zdir='y',marker='*', s=200, color='black', zs=+1.5,label='$x^*$')
    ax1.plot(Xout[1, :], Xout[2, :], zdir='x',color=mcolors.XKCD_COLORS[f'xkcd:red'].upper(),zs=-1.5)
    ax1.scatter(Xi[1], Xi[2], zdir='x',marker='o', s=200, color='black', zs=-1.5)
    ax1.scatter(0, 1, zdir='x',marker='*', s=200, color='black', zs=-1.5)

    ax1.set_xlim([-1.5,1])
    ax1.set_ylim([-1,1.5])
    ax1.set_zlim([-1,1])
    ax1.tick_params(axis='both', labelsize=15)
    ax1.set_xlabel('$x_1$', fontsize=20)
    ax1.set_ylabel('$x_2$', fontsize=20)
    ax1.set_zlabel('$x_3$', fontsize=20)
    ax1.legend(fontsize=20, ncol=3)

    ax1.set_box_aspect((2.5,2.5,2))
    ax1.view_init(elev=10, azim=-45)


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(3,1,1)
    ax2.plot(Tout, Xout[0, :],color=mcolors.XKCD_COLORS[f'xkcd:blue'].upper(),linewidth=2)
    ax2.xaxis.set_ticklabels([])
    ax2.tick_params(axis='y', labelsize=30)
    ax2.set_xlim([Tout[0],Tout[-1]])
    ax2.set_ylabel('$x_1(t)$', fontsize=30)
    ax3 = fig2.add_subplot(3,1,2)
    ax3.plot(Tout, Xout[1, :],color=mcolors.XKCD_COLORS[f'xkcd:blue'].upper(),linewidth=2)
    ax3.xaxis.set_ticklabels([])
    ax3.set_xlim([Tout[0],Tout[-1]])
    ax3.tick_params(axis='y', labelsize=30)
    ax3.set_ylabel('$x_2(t)$', fontsize=30)
    ax4 = fig2.add_subplot(3,1,3)
    ax4.plot(Tout, Xout[2, :],color=mcolors.XKCD_COLORS[f'xkcd:blue'].upper(),linewidth=2)
    ax4.tick_params(axis='both', labelsize=30)
    ax4.set_xlim([Tout[0],Tout[-1]])
    ax4.set_xlabel('$t$', fontsize=30)
    ax4.set_ylabel('$x_3(t)$', fontsize=30)

    plt.show()


