function mv = LQR_Linearize(mo)
arguments (Input)
 mo (6,1)
end
arguments (Output)
    mv (2,1)
end
global A B Q R C D
if sin(mo(3))==0
    mo(3)=mo(3)+1e-12;
end
BB=B(mo(3));
format long
disp("test")
disp(mo)
%k=lqr(A,BB,Q,R);
QXU = blkdiag(0.1*eye(6),R);
QWV = blkdiag(Q,R);
lqg(ss(A,BB,C,D),QXU,QWV)
mv=-k*mo;


