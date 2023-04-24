function [dy,u] = Sat_Translational_Dyn(~,y,u) %#codegen

    parameters.Isp=2250;
    parameters.Tmax=1.05e-7;
    g=9.81;

    %Takes u directly as acc
    u=min(parameters.Tmax,max(-parameters.Tmax,u(:)));
    torques=u;
    y=y(:);
    %n_dot=0;
    dy=y;

    %y 1to6 position and velocity
    %y 7 is mass
    
    mass=y(7);

    %set for now
    mu=3.986004418*10^14;
    rt= 6.6*1e6 ;%m r of the targhet
    rc=norm([rt+y(1);y(2);y(3)]);
    %~GPS are 2,66e7 m
    %n=sqrt(mu/rt^3);
    n= 1.2e-3;
    eta=mu/(rc^3);

    
    %Init Vars using     %F_LV_LH
    dy(1)=y(4);
    dy(2)=y(5);
    dy(3)=y(6);
    dy(4)=n^2*y(4)+2*n*y(5)+mu/(rt^2)-(eta)*(rt+y(1))+torques(1)/mass;
    dy(5)=-2*n*y(5)+n^2*y(2)-(eta)*y(2)+torques(2)/mass;
    dy(6)=-(eta)+torques(3)/mass;

    dy(7)=-sum(abs(u(:)))/(g*parameters.Isp);

end
