function [dy,u] = Chaser(t,y,y_goal) %#codegen
    %disp(t)
    coder.extrinsic('exist');
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
    %[A,B]=get_AB_tra(y,u);
    %[A,B]=sat_jacobian(y,u);
    %K=simple_lqr(A,B,get_Q_tra(),get_R_tra());
    %pause
     K=[...
        3.10446912836289e-08     -2.73186895986547e-11     -9.88240420260271e-20   2.69229716706869e-06      1.31826306669043e-05      9.67820398309341e-17
        5.86774799334853e-07     -2.58150711097142e-10     -8.02539314140983e-21         4.39421022232945e-07      0.000249254005185124     -5.73167583337792e-18
       -2.95196688268913e-18     -1.17680560270348e-23      9.01309369777884e-14         9.68285856369672e-16     -1.67494659966785e-15      2.68563510472645e-06];
       end

function [A,B]=get_AB_tra(y,u)
    %Init Vars using     %F_LV_LH
    p=Sat_params();
    n=sqrt(p.mu/p.rt^3);
    total_mass=10+p.mass;

    A = [0, 0, 0, 1, 0, 0 %
        0, 0, 0, 0, 1, 0 %
        0, 0, 0, 0, 0, 1 %
        3 * n ^ 2, 0, 0, 0, 2 * n, 0 %
        0, 0, 0, -2 * n, 0, 0 %
        0, 0, -n ^ 2, 0, 0, 0]; %

    B = [0, 0, 0
        0, 0, 0
        0, 0, 0
        1 / total_mass, 0, 0
        0, 1 / total_mass, 0
        0, 0, 1 / total_mass];

    A = A(1:6, 1:6);
    B = B(1:6, :);

end



function Q=get_Q_tra()
    Q=diag([3e1,2e1,1e1,0.1,0.1,0.1]*1e-2);
end
function R=get_R_tra()
    R=diag([1e1,3e2,1])*1e16;
end
function Return_lqr_values()
    %[A,B]=get_AB_tra([0;0;0;0;0;0;10],[0;0;0]);
    [A,B]=sat_jacobian([0;0;0;0;0;0;10],[0;0;0]);
    coder.extrinsic('lqr');
    K=lqr(A,B,get_Q_tra(),get_R_tra());
    %string([K,[";...";";...";";];"]])
    NoInput="No Input has been provided, the K lqr values are returned ";
    disp(NoInput)
    coder.extrinsic('format');
    format longg
    disp(K)
end

%% Attitude Helping Functions
function [u,e]=quat_err_rate(y,y_goal)
    q_e=quat_tracking_error(y(1:4),y_goal(1:4));
    e=[q_e;0;0;0];
    parameters=Sat_params();    
    p=-q_e(2:4)/(1-norm(q_e(2:4))^2);
    d=y_goal(5:7)-y(5:7);
    kp=0.1*parameters.invI;
    kd=0.05*parameters.invI;
    u=kp*p+kd*d;
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





