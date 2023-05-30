function [u,e]=Translation_control(t,y,y_goal_traj,tspan)
    %u=0;e=0;
    %e=y_goal(1:6)-y(1:6);

    if ~find(tspan==t)
        disp("error")
        pause()
    end
    y_goal=interp1(tspan,y_goal_traj,t).';
    %disp([y,y_goal])
    [u,e]=lqr_control_tra(y,y_goal);
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
