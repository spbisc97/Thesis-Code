function [dy,u]=Sat_dyn(t,y,u_tranlational,u_attitude)
    mass=y(14);
    y=y(1:13);
    y_tra=[y(1:6);mass];
    y_att=[y(7:13);mass];
    
    
    [dy_tra,u_tranlational] = Sat_Translational_Dyn(t,y_tra,u_tranlational);
    [dy_att,u_attitude]= Sat_Attitude_Dyn(t,y_att,u_attitude);
    dy_mass=dy_att(8)+dy_tra(7);
    
    dy=[dy_tra(1:6);dy_att(1:7);dy_mass];
    u=[u_tranlational;u_attitude];
end