clc;
clear;
close all;
base_data;
%%变量定义
%%容量
PELY = sdpvar(1);
PFC = sdpvar(1);
PTAN = sdpvar(1);
PTRANS = sdpvar(1);
%%运行
Pgt1 = sdpvar(28, 24);Pgt2 = sdpvar(28, 24);Pgt3 = sdpvar(28, 24);Pgt4 = sdpvar(28, 24);
Pgt_a1 = sdpvar(28, 24);Pgt_a2 = sdpvar(28, 24);Pgt_a3 = sdpvar(28, 24);Pgt_a4 = sdpvar(28, 24);%%辅助变量
Igt1 = binvar(28, 24);Igt2 = binvar(28, 24);Igt3 = binvar(28, 24);Igt4 = binvar(28, 24);
Pwload = sdpvar(28, 24);Pwnet = sdpvar(28, 24);Pwc = sdpvar(28, 24);
Pfc1 = sdpvar(28, 24);Pfc2 = sdpvar(28, 24);Pely = sdpvar(28, 24);
Pnet = sdpvar(28, 24);
Vhgt = sdpvar(28, 24);Vgas = sdpvar(28, 24);
Vhe = sdpvar(28, 24);Vhf1 = sdpvar(28, 24);Vhf2 = sdpvar(28, 24);Vhsell = sdpvar(28, 24);
Vh = sdpvar(28, 24);
%%约束定义
C = [];
C = [C, PELY >= 0, PFC >= 0, PTAN >= 0, PTRANS >= 0];
C = [C, Vgas >= 0, Vhgt >= 0];
C = [C, Pwload >= 0, Pwnet >= 0, Pwc >= 0];
C = [C, Pwload <= Pwt, Pwnet <= Pwt, Pwc <= Pwt];
%%（2）功率平衡
C = [C, Pgt1 + Pgt2 + Pgt3 + Pgt4 + Pwload + Pfc1 == Pload];
C = [C, Pwt == Pwload + Pwnet + Pwc + Pely];
C = [C, Pnet == Pwnet + Pfc2, Pnet <= PTRANS];
%%（2）燃气轮机
%输出功率约束
C = [C, Pgt_a1 >= Pg_min(1), Pgt_a1 <= Pg_max(1)];
C = [C, Pgt_a2 >= Pg_min(2), Pgt_a2 <= Pg_max(2)];
C = [C, Pgt_a3 >= Pg_min(3), Pgt_a3 <= Pg_max(3)];
C = [C, Pgt_a4 >= Pg_min(4), Pgt_a4 <= Pg_max(4)];
C = [C, Pgt1 >= -M * Igt1, Pgt1 <= M * Igt1];
C = [C, Pgt1 >= -M * (1 - Igt1) + Pgt_a1, Pgt1 <= M * (1 - Igt1) + Pgt_a1];
C = [C, Pgt2 >= -M * Igt2, Pgt2 <= M * Igt2];
C = [C, Pgt2 >= -M * (1 - Igt2) + Pgt_a2, Pgt2 <= M * (1 - Igt2) + Pgt_a2];
C = [C, Pgt3 >= -M * Igt3, Pgt3 <= M * Igt3];
C = [C, Pgt3 >= -M * (1 - Igt3) + Pgt_a3, Pgt3 <= M * (1 - Igt3) + Pgt_a3];
C = [C, Pgt4 >= -M * Igt4, Pgt4 <= M * Igt4];
C = [C, Pgt4 >= -M * (1 - Igt4) + Pgt_a4, Pgt4 <= M * (1 - Igt4) + Pgt_a4];
%备用约束
C = [C, Igt1 * Pg_max(1) + Igt2 * Pg_max(2) + Igt3 * Pg_max(3) + Igt4 * Pg_max(4) - Pgt1 - Pgt2 - Pgt3 - Pgt4 >= h * Pload];
%燃料约束
C = [C, Vhgt * miu_htc + Vgas == Igt1 * c(1) + Igt2 * c(2) + Igt3 * c(3) + Igt4 * c(4) + Pgt1 * b(1) + Pgt2 * b(2) + Pgt3 * b(3) + Pgt4 * b(4)];
%%（2）燃料电池
C = [C, Pely <= PELY, Pely >= 0];
C = [C, Pely == yita_ely * Vhe];
C = [C, Pfc1 >= 0, Pfc2 >= 0, Pfc1 <= PFC, Pfc2 <= PFC, Pfc1 + Pfc2 <= PFC];
C = [C, Pfc1 == yita_fc * Vhf1, Pfc2 == yita_fc * Vhf2];
%%（3）储氢罐
C = [C, Vh >= 0, Vh <= PTAN / rou_H]; 
C = [C, Vh(:, 1) == 0.5 * PTAN / rou_H + Vhe(:, 1) - Vhf1(:, 1) - Vhf2(:, 1) - Vhgt(:, 1) - Vhsell(:, 1)];
for i = 2 : 24
 C = [C, Vh(:, i) == Vh(:, i-1) + Vhe(:, i) - Vhf1(:, i) - Vhf2(:, i) - Vhgt(:, i) - Vhsell(:, i)];   
end
C = [C, Vh(:, 24) == 0.5 * PTAN / rou_H];
C = [C, Vhsell >= 0, Vhsell <= Vhload];
%%目标函数
R = r*(1+r)^m/((1+r)^m-1);
Ccap1 = e1 * PELY;
Ccap2 = e2 * PFC;
Ccap3 = e3 * PTAN;
Ccap4 = L * e4 * PTRANS + (1.368 + 2.052 / 300 * PTRANS / 1e3 + 1.900 + 3.800 / 300 * PTRANS / 1e3) * 1e8;
Ccap5 = e5 * (PELY + PFC + PTAN);
CY = 0.9 * (Ccap1 + Ccap2 + Ccap3 + Ccap4 + Ccap5) / m;
Com = Ccap1 * Q1 + Ccap2 * Q2 + Ccap3 * Q3 + Ccap4 * Q4 + Ccap5 * Q5;
Com = Com + Qh1 * sum(sum(Vhe)) + Qh2 * sum(sum(Vhf1 + Vhf2));
I = sum(Pnet * Ke1 * (1 - miu_loss)) + Kh * sum(sum(Vhsell)) + sum((Pwload + Pfc1) * Ke2) +...
    (Kc * miu_htc + Ksub) * sum(sum(Vhgt));
%  - Kc * sum(sum(Vgas))
f = 0;
f = f + 365 / 7 / N * I;
f = f - R * (Ccap1 + Ccap2 + Ccap3 + Ccap4 + Ccap5);
f = f - Com;
f = f - CY;
%%求解器选择  
opt = sdpsettings('solver', 'gurobi+');
result = optimize(C, -f, opt);
discard = sum(sum(value(Pwc)))/sum(sum(Pwt))
fprintf('电解槽容量：%.2fMW\n',value(PELY)/1000)
fprintf('燃料电池容量：%.2fMW\n',value(PFC)/1000)
fprintf('储氢罐容量：%.2ft\n',value(PTAN)/1000)
fprintf('输电系统容量：%.2fMW\n',value(PTRANS)/1000)
fprintf('年均净收益：%.2f亿元\n',value(f)/1e8)
% [model,recoverymodel] = export(C, f, sdpsettings('solver','GUROBI'));
% iis = gurobi_iis(model);
% gurobi_write(model,'TestMpdel.lp');
fprintf('弃风率：%.2f%%\n',discard*100)