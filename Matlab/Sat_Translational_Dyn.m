function dy = Sat_Translational_Dyn(~,y,u) %#codegen
    %Takes u directly as acc
    torques=u(:);
    y=y(:);
    %n_dot=0;
    dy=y;
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
    dy(4)=n^2*y(4)+2*n*y(5)+mu/(rt^2)-(eta)*(rt+y(1))+torques(1);
    dy(5)=-2*n*y(5)+n^2*y(2)-(eta)*y(2)+torques(2);
    dy(6)=-(eta)+torques(3);
end
