function [value, isterminal, direction] = jump_func_trig(~,x)
%     value = x(3,1)-1;
%     isterminal = 1;
%     direction = 1;

    value = [x(end-2,1)-1];
    isterminal = [1];
    direction = [1];
end