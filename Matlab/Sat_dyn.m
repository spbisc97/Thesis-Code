function dy=Sat_dyn(t,y,u_tranlational,u_attitude)

    mass=y(14);
    y=y(1:13);


    dy_tra = Sat_Translational_Dyn(t,[y(1:6);mass],u_tranlational);
    dy_att= Sat_Attitude_Dyn(t,y(7:13),u_attitude);
    dy_mass=dy_att(8)+dy_tra(7);

    dy=[dy_tra(1:6);dy_att(1:7);dy_mass];
end