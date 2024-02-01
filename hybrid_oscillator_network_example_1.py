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


def jump_condition(t,X):
    return X[-3]-1

def jump_map(X,num_of_modes):
    X[-4] = 2*np.random.rand()+0.5
    X[-3] = 0
    X[-2] = np.random.choice(np.arange(num_of_modes)+1)
    X[-1] = X[-1] + 1
    return X

def flow_map(t,X,omegas,epsilon,num_of_agents,Modes):
    Mode = Modes[int(X[-2]-1)]
    K = 10
    A = Mode["adj"]
    B = Mode["dir"]
    dXdt = np.array([])
    for i in range(num_of_agents):
        J_i = 0
        x_i = X[2*i:2*i+2]
        for j in list(range(i))+list(range(i+1,num_of_agents)):
            x_j = X[2*j:2*j+2]
            J_i = J_i + 0.5*A[i,j]*np.linalg.norm(x_i-x_j)**2
        dxi_dt = np.array([x_i[1],-x_i[0]])*(2*np.pi*0.25+B[i]*np.sqrt(2*np.pi*omegas[i])*np.cos(2*np.pi*omegas[i]*t/epsilon**2+K*J_i)/np.sqrt(K)/epsilon)
        dXdt = np.append(dXdt,dxi_dt)
    dXdt = np.append(dXdt,np.array([0,X[-4],0,0]))
    return dXdt



if __name__ == '__main__':
    
    num_of_graphs = 1
    num_of_directions = 4
    num_of_agents = 2

    num_of_modes = num_of_graphs*num_of_directions

    Adjs = []
    
    i = 0
    while i<num_of_graphs:
        A = np.random.randint(0,high=2,size=(num_of_agents,num_of_agents))
        A = A*A.transpose()
        A = A - np.diag(np.diag(A))
        D = np.diag(np.sum(A,axis=1))
        L = D-A
        eigvals,eigvecs = np.linalg.eig(L)
        eigvals = np.sort(eigvals)
        if eigvals[1]>0:
            Adjs.append(A)
            i = i+1
    
    B = np.array([[+1,+1],[-1,+1],[+1,-1],[-1,-1]])

    Modes = []
    for i in range(num_of_graphs):
        for j in range(num_of_directions):
            Mode = {}
            Mode["adj"]=Adjs[i]
            Mode["dir"] = B[j,:]
            Modes.append(Mode)
    
    epsilon = 1/np.sqrt(5)
    omegas = np.linspace(0,1,num_of_agents)+1
    
    x0 = np.array([])
    theta_s = np.random.rand()*np.pi
    for i in range(num_of_agents):
        thetai = i*np.pi/num_of_agents
        x0 = np.append(x0,np.array([np.cos(thetai+theta_s),np.sin(thetai+theta_s)]))

    Xi = np.concatenate((x0,[np.random.rand()+0.5],[0],[np.random.choice([1,2,3,4])],[0]))
    Ti = 0
    Tf = 10

    Tout  = np.array([Ti])
    Xout  = Xi.reshape(([2*num_of_agents+4,1]))

    Teout = np.array([])
    Xeout = np.array([[]])

    jump_trigger = lambda t, X: jump_condition(t,X)
    jump_trigger.direction = 1
    jump_trigger.terminal = True

    while Tout[-1]<Tf:
        if jump_trigger(Ti,Xi)>0:
            Xi = jump_map(Xi,num_of_modes)

        T = np.linspace(Ti,Tf,int(1e4))
        sol = solve_ivp(lambda t, X: flow_map(t, X, omegas,epsilon,num_of_agents,Modes), [T[0], T[-1]], Xi, 'RK45', T, dense_output=True,events=jump_trigger)
        Tout = np.append(Tout,sol.t)
        Xout = np.append(Xout,sol.y,axis=1)
        if np.any(sol.t_events[0]):
            Ti = sol.t_events[0][0]
            Xi = sol.y_events[0][0]

            Teout = np.append(Teout,sol.t_events[0])
            Xeout = np.append(Xeout,sol.y_events[0])
            
            Xi = jump_map(Xi,num_of_modes)

    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    U, V = np.meshgrid(u,v)
    c = np.sqrt(1-np.cos(U-V))**2

    norm = cm.colors.Normalize(vmin=np.nanmin(c), vmax=np.nanmax(c))
    cdata = cm.gray(norm(c))

    x = np.outer((2+np.cos(u)), np.cos(v))
    y = np.outer((2+np.cos(u)), np.sin(v))
    z = np.outer(np.sin(u),np.ones(np.shape(v)))
    theta1 = np.arctan2(Xout[0,:],Xout[1,:])
    theta2 = np.arctan2(Xout[2,:],Xout[3,:])

    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    
    cax1 = ax1.contourf(U,V,c,cmap=cm.gray,levels=40,antialiased=False)
    plt.colorbar(cax1, orientation='vertical',fraction = 0.0451, norm = norm)
    ax1.contour(U,V,c,cmap=cm.gray,levels=40,antialiased=False)
    ax1.plot(theta1+np.pi,theta2+np.pi,linewidth=3,color=mcolors.XKCD_COLORS[f'xkcd:red'].upper(),label='$x(t)$')
    ax1.scatter(theta1[0]+np.pi, theta2[1]+np.pi, marker='o', s=200, color='black', label='$x(0)$',zorder=10)

    ax1.set_xlim([0,2*np.pi])
    ax1.set_ylim([0,2*np.pi])
    ax1.set_aspect(aspect=1)
    
    ax1.tick_params(axis='both', labelsize=20)
    ax1.set_xlabel('$\\xi_1$', fontsize=30)
    ax1.set_ylabel('$\\xi_2$', fontsize=30)


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(2,1,1)
    ax2.plot(Tout, Xout[0, :],color=mcolors.XKCD_COLORS[f'xkcd:blue'].upper(),linewidth=2)
    ax2.plot(Tout, Xout[2, :],color=mcolors.XKCD_COLORS[f'xkcd:red'].upper(),linewidth=2)
    ax2.xaxis.set_ticklabels([])
    ax2.set_xlim([Tout[0],Tout[-1]])
    ax2.set_ylabel('$x^i_1(t)$', fontsize=20)
    
    ax3 = fig2.add_subplot(2,1,2)
    ax3.plot(Tout, Xout[1, :],color=mcolors.XKCD_COLORS[f'xkcd:blue'].upper(),linewidth=2)
    ax3.plot(Tout, Xout[3, :],color=mcolors.XKCD_COLORS[f'xkcd:red'].upper(),linewidth=2)
    ax3.set_xlim([Tout[0],Tout[-1]])
    ax3.set_ylabel('$x^i_2(t)$', fontsize=20)

    fig3 = plt.figure()
    ax4 = fig3.add_subplot()
    ax4.plot(Xout[-2, :],Tout,color=mcolors.XKCD_COLORS[f'xkcd:blue'].upper(),linewidth=10)
    ax4.set_xticks([1,2,3,4])
    ax4.set_xlim([0.5,4.5])
    ax4.set_ylim([Tout[0],Tout[-1]])
    ax4.set_xlabel('$z_1(t)$', fontsize=30)
    ax4.set_ylabel('$t$', fontsize=30)
    ax4.tick_params(axis='both', labelsize=20)
    ax4.grid(True)

    plt.show()


