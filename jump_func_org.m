function xp = jump_func_org(~,x,mu)
    xp = x;
    xp(end-2,1) = 0;
    idx = 1:3;
    idx(idx==xp(end-1,1))=[];
    xp(end-1,1) = idx(randi(2));
    if xp(end-1,1)==1
        xp(end-3,1)=max(2*pi*mu*rand,2*pi*mu);
    elseif xp(end-1,1)==2
        xp(end-3,1)=max(2*pi*0.15*rand,2*pi*0.05);
    elseif xp(end-1,1)==3
        xp(end-3,1)=max(2*pi*0.10*rand,2*pi*0.05);
    end
    xp(end,1) = xp(end,1) + 1;
end