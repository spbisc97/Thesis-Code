function [dy,u] = Sat_Translational_Dyn(~,y,u) %#codegen

    parameters=Sat_params();

    g=9.81;

    %Takes u directly as acc
    u=min(parameters.Tmax,max(-parameters.Tmax,u(:)))*min(1,max(0,y(7)*1e2));;
    torques=u;
    y=y(:);
    %n_dot=0;
    dy=y;

    %y 1to6 position and velocity
    %y 7 is mass

    %set for now
    mu=3.986004418*10^14;
    %syms mu
    rt= 6.6*1e6 ;%m r of the targhet
    %syms rt
    rc=norm([rt+y(1);y(2);y(3)]);
    rc=((rt+y(1))^2+y(2)^2+y(3)^2)^(1/2);
    %syms rc
    %~GPS are 2,66e7 m
    %n=sqrt(mu/rt^3);
    n= parameters.n;
    %syms n
    eta=mu/(rc^3);
    total_mass=(y(7)+parameters.mass);

    
    %Init Vars using     %F_LV_LH
    dy(1)=y(4);
    dy(2)=y(5);
    dy(3)=y(6);
    dy(4)=n^2*y(1)+2*n*y(5)+mu/(rt^2)-(eta)*(rt+y(1))+torques(1)/total_mass;
    dy(5)=-2*n*y(4)+n^2*y(2)-(eta)*y(2)+torques(2)/total_mass;
    dy(6)=-(eta)*y(3)+torques(3)/total_mass;
    dy(7)=-sum(abs(u(:)))/(g*parameters.Isp);

end



