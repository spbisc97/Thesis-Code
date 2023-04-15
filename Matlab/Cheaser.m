function [dy,u] = Cheaser(t,y,y_goal)
    if ~exist('y','var')
        Return_lqr_values()
        return;

    end
    %y=y(1:6);


    u_tranlational=get_k()*(y_goal(1:6)-y(1:6));
    u_attitude=[0;0;0];
    dy = [Sat_Translational_Dyn(t,y(1:6),u_tranlational);Sat_Attitude_Dyn(t,y(7:13),u_attitude)];

end

%%Helping Functions
function K=get_k()
    %K=lqr(get_A(),get_B(),get_Q(),get_R());
    %pause
    K=[...
        3.1623   -0.0000   -0.0000    3.1633    0.0000    0.0000;...
        0.0000    3.1623    0.0000   -0.0000    3.1633    0.0000;...
        0.0000   -0.0000    3.1623    0.0000   -0.0000    3.1633;];
end

function A=get_A()
    n= 6.3135e-04;
    %Init Vars using     %F_LV_LH
    A=[ 0 0 0 1 0 0 ;...
        0 0 0 0 1 0;...
        0 0 0 0 0 1;...
        3*n^2 0 0 0 2*n 0;...
        0 0 0 -2*n 0 0;...
        0 0 -n^2 0 0 0;...
        ];
end
function B= get_B()
    B=[ 0 0 0;...
        0 0 0;...
        0 0 0;...
        1 0 0;...
        0 1 0;...
        0 0 1;];
end
function Q=get_Q()
    Q=diag([1,1,1,1,1,1]);
end
function R=get_R()
    R=diag([1,1,1])*1e-7;
end
function Return_lqr_values()
    K=lqr(get_A(),get_B(),get_Q(),get_R());
    %string([K,[";...";";...";";];"]])
    NoInput="No Input has been provided, the K lqr values are returned ";
    disp(NoInput)
    disp(K)
end


