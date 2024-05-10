clc, clear, clf;
% q0 = 10*(rand(2,1)-0.5);
q0 = [-4;4];
epsilon = 1/sqrt(2*pi*5);

rng(5)

Ti = 0;
Tf = 15;
Nf = 1000;
refine = 4;

T1i = Ti;
T2i = Ti;

X1i = [q0;[1;0];2*pi*0.125;0;2;0];
X2i = X1i;

T1out = T1i;
T1eout = [];
T2out = T2i;
T2eout = [];

X1out = X1i;
X1eout = [];
X2out = X2i;
X2eout = [];


options = odeset('RelTol',1e-7,'AbsTol',1e-7,'Events',@jump_func_trig,'Refine',refine);
% 
% figure(1);
% ax = axes;
% box on
Lambdas = [];
mu = 0.25;
while (T1out(end)<Tf)&&(X1out(4,end)<=Nf)
    
    lambda = max(2*pi*0.25*rand,2*pi*0.005);
    Lambdas = [Lambdas;lambda];
    [T1,X1,T1e,X1e,~] = ode45(@(t,x)liebra_seeking(t,x,epsilon,lambda,mu),[T1i,Tf],X1i,options);

    T1out = [T1out;T1(1:end)];
    T1eout = [T1eout;T1e];
    X1out = [X1out,X1(1:end,:)'];
    X1eout = [X1eout;X1e'];

%     plot3(T1,X1(:,end),X1(:,1),'r','LineWidth',2), drawnow
%     if ~ishold
%       hold on
%     end
    if ~isempty(T1e)
        X1i = jump_func_org(T1e,X1(end,:)',mu);
    end
    options = odeset(options,'InitialStep',T1(end)-T1(end-refine),'MaxStep',T1(end)-T1(1));
    T1i = T1(end);
end
xmin = min(X1out(1,:))-1;
xmax = max(X1out(1,:))+1;
ymin = min(X1out(2,:))-1;
ymax = max(X1out(2,:))+1;

%%
q0 = [-4;-4];
T2i = Ti;

X2i = [q0;[1;0];2*pi*0.125;0;2;0];


T2out = T2i;
T2eout = [];

X2out = X2i;
X2eout = [];
Lambdas = [];

mu = 0.03;

while (T2out(end)<Tf)&&(X2out(4,end)<=Nf)
    
    lambda = max(2*pi*5*rand,2*pi*5);
    Lambdas = [Lambdas;lambda];
    [T2,X2,T2e,X2e,~] = ode45(@(t,x)liebra_seeking(t,x,epsilon,lambda,mu),[T2i,Tf],X2i,options);

    T2out = [T2out;T2(1:end)];
    T1eout = [T2eout;T2e];
    X2out = [X2out,X2(1:end,:)'];
    X2eout = [X2eout;X2e'];

    if ~isempty(T2e)
        X2i = jump_func_org(T2e,X2(end,:)',mu);
    end
    options = odeset(options,'InitialStep',T2(end)-T2(end-refine),'MaxStep',T2(end)-T2(1));
    T2i = T2(end);
end

x = linspace(min(xmin,min(X2out(1,:))-1)-1,max(xmax,max(X2out(1,:))+1)+1,1e3);
y = linspace(min(ymin,min(X2out(2,:))-1)-1,max(ymax,max(X2out(2,:))+1)+1,1e3);
[xx,yy]= meshgrid(x,y);
zz = 0.5*(xx.^2+yy.^2);
%%
% figure(2); 
% % subplot(4,4,1:4),plot(T1out,X1out(2,:),'b','LineWidth',1.5),hold on;
% xlabel(ax1,"$t$", "Interpreter","latex")
% ax1.FontSize = 15;
% ylabel(ax1,"$|x(t)-x_p^*|$", "Interpreter","latex")
% % legend({'$x_1$','$x_2$'},"Interpreter","latex","Location","northoutside","Orientation","horizontal","FontSize",15)
% % legend({'$x_1(t)$','$x_2(t)$'},"Interpreter","latex","FontSize",15)
% subplot(4,4,[8 12 16]),plot(X1out(end-1,:),T1out,'b','LineWidth',2), grid on, grid minor, ylim([T1out(1) T1out(end)]), hold on;
% ax2 = gca;
% xlabel(ax2,"$q(t)$", "Interpreter","latex")
% ylabel(ax2,"$t$", "Interpreter","latex")
% ax2.FontSize = 15;
% subplot(4,4,[5:7,9:11,13:15]), plot(X1out(1,:),X1out(2,:),'b','LineWidth',1.5), axis equal,  grid on, hold on;
% ax3 = gca;
% ax3.FontSize = 15;
% xlabel(ax3,"$x_1$","Interpreter","latex");
% ylabel(ax3,"$x_2$","Interpreter","latex");
%%
figure(1); 
subplot(4,4,1:4),plot(T1out,sqrt(X1out(1,:).^2+X1out(2,:).^2),'b','LineWidth',3), grid on, grid minor, xlim([T2out(1) T2out(end)]), hold on;
ax1 = gca;
subplot(4,4,1:4,ax1),plot(ax1,T2out,sqrt(X2out(1,:).^2+X2out(2,:).^2),'r','LineWidth',3), hold off;
% ax1 = gca;
xlabel(ax1,"$t$", "Interpreter","latex")
ylabel(ax1,"$|x(t)-x_p^*|$", "Interpreter","latex")
ax1.FontSize = 20;
subplot(4,4,[8 12 16]),plot(X1out(end-1,:),T1out,'b','LineWidth',7), grid on, grid minor, ylim([T1out(1) T1out(end)]), hold on;
ax2 = gca;
subplot(4,4,[8 12 16],ax2),plot(ax2,X2out(end-1,:),T2out,'r','LineWidth',7), xlim([0.75,3.25]),hold off;
xlabel(ax2,"$q(t)$", "Interpreter","latex")
ylabel(ax2,"$t$", "Interpreter","latex")
ax2.FontSize = 20;
subplot(4,4,[5:7,9:11,13:15]),contourf(xx,yy,zz,20,'LineWidth',1,'LineStyle','none'), colormap gray, axis equal, hold on;
ax3 = gca;
plot(ax3,X1out(1,:),X1out(2,:),'b','LineWidth',3);
plot(ax3,X2out(1,:),X2out(2,:),'r','LineWidth',3),hold off;
ax3.FontSize = 20;
xlabel(ax3,"$x_1$","Interpreter","latex");
ylabel(ax3,"$x_2$","Interpreter","latex");
colorbar
%%
clc

idx1=~(X1out(end-1,:)==3);
trapz(T1out,idx1)
X1out(end,end)

idx2=~(X2out(end-1,:)==3);
trapz(T2out,idx2)
X2out(end,end)