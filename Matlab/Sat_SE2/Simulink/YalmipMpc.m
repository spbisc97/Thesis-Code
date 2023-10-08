function u=YalmipMpc(w,ref)
arguments
    w (6,1)
    ref (6,1)
end
%YALMIPMPC Summary of this function goes here

%   Detailed explanation goes here
persistent Controller
if isempty(Controller)
    
    global A B C D Q R N
    yalmip('clear');
    p=c2d(ss(A,B,C,D),0.05);
    Ad=p.A;
    Bd=p.B;
    Cd=p.C;
    Dd=p.D;
    
    [n,m]=size(Bd);
    [n,p]=size(Cd);
    
    x = sdpvar(repmat(n,1,N+1),repmat(1,1,N+1));
    u = sdpvar(repmat(m,1,N),repmat(1,1,N));
    r=sdpvar(n,1);
    
    constraints = [];
    objective = 0;
    for k = 1:N
        objective = objective + (r-C*x{k})'*Q*(r-C*x{k})+u{k}'*R*u{k};
        constraints = [constraints, x{k+1} == Ad*x{k}+Bd*u{k}];
        %constraints = [constraints, -1e-2 <= u{k}<= 1e-2];
        %objective=objective+1e100*blackbox([x{k}(3);u{k}(1);u{k}(2)],nlconst);
        %constraints=[constraints,0.1<=blackbox([x{k}(3),u{k}(1),u{k}(2)],nlconst)<=0.1]
    end
    objective = objective + (r-C*x{N+1})'*Q*(r-C*x{N+1});
    ops=sdpsettings('verbose',0);
    ops.solver='gurobi+';
    ops.allownonconvex = 1;
    
    Controller = optimizer(constraints,objective,ops,{x{1},r},u{1});
    
end
[u,err]=Controller(w,ref);
% if err
%     warning('The problem is infeasible');
% end



end