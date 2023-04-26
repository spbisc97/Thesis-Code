function [dy,u] = Cheaser(t,y,y_goal) %#codegen
    %disp(t)
    if ~exist('y','var')
        Return_lqr_values()
        return;
    end

    y=y(:);


    u_tranlational=translation_control(y(1:6),y_goal(1:6));%later mass will have to be passed
    u_attitude=attitude_control(y(7:13),y_goal(7:13));%later mass will have to be passed

    
    [dy,u]=Sat_dyn(t,y,u_tranlational,u_attitude);
    %u=[u_tranlational;u_attitude];
end




%% Control Functions
function [u,e]=translation_control(y,y_goal)
    %u=0;e=0;
    %e=y_goal(1:6)-y(1:6);
    [u,e]=lqr_control_tra(y(1:6),y_goal(1:6));
end

function [u,e]=attitude_control(y,y_goal)
    %u=[0;0;0];
    e=[0;0;0;0;0;0;0];
    u=quat_err_rate(y,y_goal);

end

%% Translation Helping Functions
function [u,e]=lqr_control_tra(y,y_goal)
    e=y_goal(1:6)-y(1:6);
    u=get_k_tra(y,[0;0;0])*(e);
    
end

function K=get_k_tra(y,u)
    [A,B]=get_AB_tra(y,u);
    K=lqr(A,B,get_Q_tra(),get_R_tra());
    %pause
%    K=[...
%    99.9999446137011        -0.107264213962748      3.44547485353584e-14           89.498578587015     -8.00054467120503e-15      2.95056468803997e-14;...
%    0.107264213962668          99.9999446137011     -6.04285394796429e-14       2.1316282072803e-14           89.498578587015     -9.52252925846075e-15;...
% -1.4019034670654e-14      7.00884783457919e-14                       100     -9.53985852357144e-15      7.69306494634369e-14          89.4985478828506];
end

function [A,B]=get_AB_tra(y,u)
    %Init Vars using     %F_LV_LH
    parameters=Sat_params();
    mu=parameters.mu;
    rt= parameters.rt ;%m r of the targhet
    n=parameters.n;
    y(7)=10;

    A = [0, 0, 0, 1, 0, 0 %
        0, 0, 0, 0, 1, 0 %
        0, 0, 0, 0, 0, 1 %
        3 * n ^ 2, 0, 0, 0, 2 * n, 0 %
        0, 0, 0, -2 * n, 0, 0 %
        0, 0, -n ^ 2, 0, 0, 0]; %

    B = [0, 0, 0
        0, 0, 0
        0, 0, 0
        1 / (parameters.mass + y(7)), 0, 0
        0, 1 / (parameters.mass + y(7)), 0
        0, 0, 1 / (parameters.mass + y(7))];

    A = A(1:6, 1:6);
    B = B(1:6, :);

end



function Q=get_Q_tra()
    Q=diag([1e1,1e1,1e1,0.1,0.1,0.1]*1e-3);
end
function R=get_R_tra()
    R=diag([1,1,1])*1e14;
end
function Return_lqr_values()
    [A,B]=get_AB_tra([0;0;0;0;0;0;10],[0;0;0]);
    K=lqr(A,B,get_Q_tra(),get_R_tra());
    %string([K,[";...";";...";";];"]])
    NoInput="No Input has been provided, the K lqr values are returned ";
    disp(NoInput)
    format longg
    disp(K)
end

%% Attitude Helping Functions
function [u,e]=quat_err_rate(y,y_goal)
    q_e=quat_tracking_error(y(1:4),y_goal(1:4));
    e=[q_e;0;0;0];
    u=-q_e(2:4)/(1-norm(q_e(2:4))^2)-y(5:7);
    u=u/100;
end

function mat=omega(q)
    % mat for crossproduct btw quaternions
    % q(1) is the scalar part

    mat=[q(1),-q(2),-q(3),-q(4);...
        q(2),q(1),q(4),-q(3);...
        q(3),-q(4),q(1),q(2);...
        q(4),q(3),-q(2),q(1)];
end

function e=quat_tracking_error(q,q_d)
    q_d_inv=quatinv(q_d')';
    e=omega(q_d_inv)*q;
    e=e(:);
    if norm(e)>1.1
        disp("error norm error")
        disp(norm(e))
    end
end





