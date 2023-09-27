clear all
close all
addpath("Auxiliary/")
test=2;
if test==1
    p=Sat_params();

    y0=10*1e3;

    initialconditions=[0 y0 0 y0/2000 0 0]';
    days=2;
    timespan=1:3600*24*days;
    f=figure(1);
    hold off
    [t,w1]=ode45(@(t,w) Sat_dyn_Lin(t,w,[0,0.1*sin(t/1000)],p),timespan,initialconditions);
    plot3(w1(:,1),w1(:,2),t)
    hold on
    [t,w2]=ode45(@(t,w) Sat_dyn_nonLin(t,w,[0,0.1*sin(t/1000)],p),timespan,initialconditions);
    plot3(w2(:,1),w2(:,2),t)
    % nonLinear=1
    % [t,w2]=ode45(@(t,w) Sat_dyn(t,w,[0,0],p,nonLinear),timespan,initialconditions);


    plot3(0*t,0*t,t)

    legend
    grid on
    plot_settings(f)

    f=figure(2);
    plot(t,w1(:,1:2)-w2(:,1:2))
    grid on
    legend
    plot_settings(f)


    f=figure(3);
    plot(w2(:,1),w2(:,2))
    hold on
    plot(w1(:,1),w1(:,2))
    legend
    grid on
    plot_settings(f,"manual")


    f=figure(4);
    plot(t,[w2(:,3),w2(:,6)])
    hold on
    plot(t,sin(t/1000))
    legend
    grid on
    plot_settings(f)

end
if test==2
    p=Sat_params();

    y0=10*1e3;

    initialconditions=[0 y0 0 y0/2000 0 0]';
    days=0.1;
    timespan=1:0.1:3600*24*days;
    fh=@(t,w) Sat_dyn_Lin(t,w,[1e-3*sin(t/1000+10e-3),sin(t/1000)]',p);
    tic
    [t1,w1]=ode45(fh,timespan,initialconditions);
    toc
    tic
    [t2,w2]=RK4(fh,timespan,initialconditions);
    toc
    tic
    [t3,w3]=Euler(fh,timespan,initialconditions);
    toc

    % nonLinear=1
    % [t,w2]=ode45(@(t,w) Sat_dyn(t,w,[0,0],p,nonLinear),timespan,initialconditions);
    f=figure(2);
    plot(t1,w1(:,1:2))
    hold on
    plot(t2,w2(:,1:2))
    hold on
    plot(t3,w3(:,1:2))


    grid on
    legend
    plot_settings(f)
    [M,I]=max(abs(w1(:,1:2)-w2(:,1:2)))
    [M,I]=max(abs(w1(:,1:2)-w3(:,1:2)))
    [M,I]=max(abs(w2(:,1:2)-w3(:,1:2)))

    f=figure(3);
    tiledlayout("flow")
    nexttile
    plot(t1,abs(w1(:,1:2)-w2(:,1:2)))
    nexttile
    plot(t1,abs(w1(:,1:2)-w3(:,1:2)))


    grid on
    legend
    plot_settings(f)



end


function plot_settings(figure,aspect)
ax=figure.CurrentAxes;


if exist("aspect","var")

    % Set the remaining axes properties
    set(ax,'DataAspectRatio',[1,1,1],'XLimitMethod','padded','YLimitMethod',...
        'padded','ZLimitMethod','padded');
end

box(ax,'on');
grid(ax,'on');
hold(ax,'off');

% Create legend
legend1 = legend(ax,'show');
end