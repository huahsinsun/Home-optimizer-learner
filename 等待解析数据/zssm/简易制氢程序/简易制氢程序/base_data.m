N = 4;M = 100000;
miu_loss = 0.025;
miu_htc = 0.2999;

Kh = 4;
Kc = 1.85;
Ke1 = zeros(24,1);
Ke1(1:6) = 0.235;Ke1(24) = 0.235;
Ke1(7:8) = 0.510;Ke1(13:15) = 0.510;Ke1(23) = 0.510;
Ke1(9:12) = 0.891;Ke1(16:22) = 0.891;
Ke2 = 0.245 * Kc * ones(24,1);
Ksub = 1;
e1 = 6500; e2 = 8000; e3 = 4970; e4 = 59189 / 1e3; e5 = 515280 / 1e3;
Q1 = 0.02; Q2 = 0.02; Q3 = 0.01; Q4 = 0.005; Q5 = 0.0075;
Qh1 = 0.1; Qh2 = 0.1;

res = 0.1;
L = 80; m = 20; r = 0.05; h = 0.05; rou_H = 0.0899;

yita_ely = 5.12;
yita_fc = 1.73;

Pg_min = [3; 3; 1; 1] * 1e3; Pg_max = [15; 15; 5; 5] * 1e3;
b = 172.5 * ones(4,1) / 1e3; c = 729.2 * ones(4,1) / 1e3;

Pload0 = xlsread('C:\Users\31127\Desktop\模板\制造甲烷\简易制氢程序\文献数据.xlsx', 'LOAD', 'B2:E169') * 1e3;
Pwt0 = xlsread('C:\Users\31127\Desktop\模板\制造甲烷\简易制氢程序\文献数据.xlsx', 'WT', 'B2:E169') * 1e3;
Pload = zeros(28, 24);
Pwt = zeros(28, 24);
Vhload = zeros(28, 24);
for i = 1 : N
    for j = 1 : 7
        Pload(7 * (i - 1) + j, :) = Pload0(24 * (j - 1) + 1 : 24 * j, i);
        Pwt(7 * (i - 1) + j, :) = Pwt0(24 * (j - 1) + 1 : 24 * j, i);
    end
end
Vhload(:, 24) = 50 * 1000 / rou_H;
