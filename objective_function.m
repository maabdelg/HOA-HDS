function J = objective_function(t,x)
%     J = 10/(1+0.025*(x(1)^2+x(2)^2));
%     J = -norm(x(1:2,1))*tanh(norm(x(1:2,1)));
    J = 0.5*norm(x)^2;
    % J = 2.5*norm(x)*tanh(norm(x));
end