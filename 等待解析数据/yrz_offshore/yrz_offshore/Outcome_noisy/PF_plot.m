clc,clear
dim = 14;col = 6094; Iteration = 150; obj_num = 3; NSGA_ref = 4;

% adress = 'D:\BaiduSyncdisk\STUDY\学术\海上风电联网\Offshore_planning\Typical_scenario\Outcome\target';
% adress = 'C:\Users\11845\Documents\BaiduSyncdisk\STUDY\学术\海上风电联网\Offshore_planning\Typical_scenario\Outcome\target';

obj_par = read_excel_files('.',[2:col],[dim+1 dim+2]);
x_par = read_excel_files('.',[2:col],[1:12]);
[par_dt, par_ndt, par_index] = selectParetoOptimalPoints(-obj_par,'min');
nor_par = normalize(obj_par,1);
x_dt = x_par(par_index,:);
min(nor_par,[],1)
[~, sorted_index] = sort(par_dt(:,2));
par_dt = par_dt(sorted_index,:); x_dt = x_dt(sorted_index,:);
% obj_ehv = read_excel_files('.',[2:col],[dim*3+3 dim*3+4]);
% [ehv_dt, ehv_ndt, ehv_index]= selectParetoOptimalPoints(-obj_ehv,'min');
% nor_ehv = normalize(ehv_dt,1);
% min(nor_ehv,[],1)
% 
% obj_nehv = read_excel_files('.',[2:col],[dim*3+5 dim*3+6]);
% [nehv_dt, nehv_ndt, nehv_index]= selectParetoOptimalPoints(-obj_nehv,'min');
% nor_nehv = normalize(obj_nehv,1);
% min(nor_nehv,[],1)

obj_NSGA = read_excel_files('.\NSGAII',[2:1800],[12+1 12+2]);
[NSGA_dt, NSGA_ndt, NSGA_index]= selectParetoOptimalPoints(-obj_NSGA,'min');
figure(3)
hold on 
grid on
% scatter(-obj_par(:,2),-obj_par(:,1) )
% twoDPFplot(ehv_dt(:,2),ehv_dt(:,1),[1 0 0])
twoDPFplot(par_dt(:,2),par_dt(:,1),[0 1 0])
% twoDPFplot(NSGA_dt(:,2),NSGA_dt(:,1),[0.5 0.5 0.5])
legend ({'AMBO','NSGA','Cont','NSGAII'},Location="best")