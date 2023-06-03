function [dy,u] = Sat_Translational_Dyn(~,y,u) %#codegen
    coder.inline('always');
    p=Sat_params();
    total_mass=(y(7)+p.mass);
    
    g=9.81;
    
    %Takes u directly as acc
    u=min(p.Tmax,max(-p.Tmax,u))*min(1,max(0,y(7)*1e2));
    torques=u(:);
    y=y(:);
    dy=y;
    
    %y 1to6 position and velocity
    %y 7 is mass
    
    rc=((p.rt+y(1))^2+y(2)^2+y(3)^2)^(1/2);
    n=sqrt(p.mu/p.rt^3);
    eta=p.mu/(rc^3);
    
    %Init Vars using     %F_LV_LH
    dy(1)=y(4);
    dy(2)=y(5);
    dy(3)=y(6);
    dy(4)=n^2*y(1)+2*n*y(5)+p.mu/(p.rt^2)-(eta)*(p.rt+y(1))+torques(1)/total_mass;
    dy(5)=-2*n*y(4)+n^2*y(2)-(eta)*y(2)+torques(2)/total_mass;
    dy(6)=-(eta)*y(3)+torques(3)/total_mass;
    dy(7)=-sum(abs(u))/(g*p.Isp);
    
end



