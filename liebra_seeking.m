function dxdt = liebra_seeking(t,x,epsilon,lambda,mu)
    p = x(1:2,1);
    o = x(3:4,1);
    q = x(end-1,1);
    lambda = x(end-3,1);
    dqdt = sqrt(2)*cos(t/epsilon^2+(q-2)*objective_function(t,p))/epsilon*o;
    dodt = [0 1;-1 0]*o/epsilon;

    dxdt = [dqdt;dodt;0;lambda;0;0];
end