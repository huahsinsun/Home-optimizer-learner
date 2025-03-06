"""
Description:本文档建模为只报量建模,考虑多灵活性负荷资源
Author: Huahsin Sun
Date-Version : 2024.10.18 Version 2.0
说明:
2024.10.22
这个文档预设所有元件尺度单位都是mwh,如果需要建模ess,tcr和dl为kwh,需要修改部分代码,加入缩放因子,详见中国移动项目代码
"""
import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt
from docplex.mp.context import Context



# Set up the default context for the optimization model
context = Context.make_default_context()
context.cpus = 4

# data preparation
power_load = [3.84, 3.92, 3.76, 3.44, 3.28, 3.36, 3.6 , 4.64, 15 , 6, 5.76,
       5.84, 5.76, 5.6 , 5.44, 5.2 , 4.96, 4.8 , 4.72, 4.72, 4.8 , 4.96,
       5.2 , 4.8 ]
# price
price = [
    40.4411764700000, 38.9705882400000, 36.0294117600000, 33.3823529400000, 30.5882352900000,
    28.9705882400000, 32.9411764700000, 20.4411764700000, 43.5294117600000, 48.5294117600000,
    54.4117647100000, 55.0000000000000, 45.5882352900000, 46.3235294100000, 51.4705882400000,
    51.7647058800000, 53.8235294100000, 51.4705882400000, 44.8529411800000, 48.5294117600000,
    47.5000000000000, 48.0882352900000, 47.6470588200000, 43.2352941200000
]

# pv
pv = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06825511, 0.53766546, 0.89338147, 1.5766065, 1.60779783, 
      2.0, 1.93636582, 1.60038508, 1.31850782, 0.69622142, 0.46180505, 0.17915764, 0.01482551, 
      0.0, 0.0, 0.0, 0.0, 0.0 ]

# ambient temperature (TCR-external) 外部
temperature = [
    16, 15, 15, 14, 13, 15, 17, 18, 19, 22, 23, 24, 25, 26, 26,
    26, 26, 24, 23, 22, 21, 21, 20, 19
]

wind_power = np.roll(pv, 5)  # 利用pv的数据滚动模拟风力数据


class virtual_power_plant:
    time_line = 24

    def __init__(self, num_equipments=[1,1,1,1,1,1]):  # expected [x,x,x,x,x,x]  self.equip_nums == [dg,pv,wind,ess,tcr,dl]

        self.equip_nums = num_equipments  # dg,pv,wind,ess
        self.alpha = 0.0  # 安全裕度
        self.M = 1e9

        # 接口数据部分
        self.price_forecast = price
        self.temp_forecast = temperature
        self.pv_forecast = pv
        self.wind_forecast = wind_power
        self.load_forecast = power_load

        # 全貌

        # Define DG parameters
        self.dg_p_max = [10 ]*self.equip_nums[0] # Maximum generation capacity for each DG
        self.dg_p_min = [ 2 ]*self.equip_nums[0]  # Minimum generation capacity for each DG
        self.dg_cost = [ 2.4, 0.05, 0.0 ]
        # Define pv
        self.pv_forecast = [self.pv_forecast for _ in range(self.equip_nums[1])]
        # Define wind
        self.wind_forecast = [self.wind_forecast for _ in range(self.equip_nums[2])]
        # Define SOC (State of Charge) parameters for the energy storage system
        self.ess0 = [7.5]*self.equip_nums[3]  # Initial SOC
        self.ess_max = [15]*self.equip_nums[3]  # Maximum SOC
        self.p_in_out_max = [3]*self.equip_nums[3]  # Maximum charge/discharge power
        self.esr = [0.995]*self.equip_nums[3]  # Storage efficiency
        self.cdr = [0.95]*self.equip_nums[3]  # Charge/discharge efficiency
        self.ess_cost = 2  # Cost rate for energy storage
        # Define TCR parameters
        self.tcr_max = [30]*self.equip_nums[4]  # Maximum TCR capacity
        self.tcr_min = [20]*self.equip_nums[4]  # Minimum TCR capacity note that var define part hard code
        self.tcr_initial = [21]*self.equip_nums[4]  # Initial TCR capacity
        self.tcr_cost = 2  # Cost rate for TCR
        self.tcr_p_max = [3]*self.equip_nums[4]  # Maximum TCR power
        self.tcr_sr = [0.95]*self.equip_nums[4]  # TCR save rate
        self.tcr_tr = [0.09]*self.equip_nums[4]  # transformer ratio

        # Define DL parameters
        self.dl_max = [4]*self.equip_nums[5]  # Maximum DL capacity
        self.dl_min = [0]*self.equip_nums[5]  # Minimum DL capacity
        self.dl_required = [2.5]*self.equip_nums[5]  # Required DL capacity
        self.dl_cost = 5  # Cost rate for DL    
        self.dl_p_max = [3]*self.equip_nums[5]  # Maximum DL power
        self.dl_sr = [0.95]*self.equip_nums[5]  # DL save rate
        self.dl_tr = [0.98]*self.equip_nums[5]  # transformer ratio
        self.dl_initial = [0.25]*self.equip_nums[5]  # Initial DL
        self.dl_start = [5]*self.equip_nums[5]  # Start time for DL
        self.dl_end = [20]*self.equip_nums[5]  # End time for DL



    def create_model(self):
        mdl = Model('''只报量建模''')
        dg_l, pv_l, wind_l, ess_l = [ ], [ ], [ ], [ ]
        bin = [ ]
        p_in, p_out = [ ], [ ]
        tcr_l,p_tcr_l=[],[]
        dl_l,p_dl_l=[],[]

        # self.equip_nums == [dg,pv,wind,ess,tcr,fr]
        for i in range(self.equip_nums[ 0 ]):
            dg_l.append(mdl.continuous_var_list(keys=virtual_power_plant.time_line, name=f'发电机{i + 1}', lb=0))
        for i in range(self.equip_nums[ 1 ]):
            pv_l.append(mdl.continuous_var_list(keys=virtual_power_plant.time_line, name=f'光伏{i + 1}', lb=0))
        for i in range(self.equip_nums[ 2 ]):
            wind_l.append(mdl.continuous_var_list(keys=virtual_power_plant.time_line,name=f'风电{i+1}'))
        for i in range(self.equip_nums[ 3 ]):
            ess_l.append(mdl.continuous_var_list(keys=virtual_power_plant.time_line, name=f'储能{i + 1}', lb=0))
            p_in.append(mdl.continuous_var_list(keys=virtual_power_plant.time_line, lb=0))
            p_out.append(mdl.continuous_var_list(keys=virtual_power_plant.time_line, lb=0))
            bin.append(mdl.binary_var_list(keys=virtual_power_plant.time_line))
        for i in range(self.equip_nums[4]):
            tcr_l.append(mdl.continuous_var_list(keys=virtual_power_plant.time_line, name=f'tcr{i+1}', lb=0))
            p_tcr_l.append(mdl.continuous_var_list(keys=virtual_power_plant.time_line, name=f'p_tcr{i+1}', lb=0))
        for i in range(self.equip_nums[5]):
            dl_l.append(mdl.continuous_var_list(keys=virtual_power_plant.time_line, name=f'dl{i+1}', lb=0))
            p_dl_l.append(mdl.continuous_var_list(keys=virtual_power_plant.time_line, name=f'p_dl{i+1}', lb=0))
        vpp_output = mdl.continuous_var_list(keys=virtual_power_plant.time_line,lb=-40, name='虚拟电厂出力')

        print('*' * 10, 'Adding Constraints', '*' * 10)

        for j in range(virtual_power_plant.time_line):
            # 1. 发电机出力约束
            for i in range(self.equip_nums[0]):
                mdl.add_constraint(dg_l[ i ][ j ] <= self.dg_p_max[ i ])
                mdl.add_constraint(dg_l[ i ][ j ] >= self.dg_p_min[ i ])

            # 2. 光伏出力约束
            for i in range(self.equip_nums[1]):
                mdl.add_constraint(mdl.sum(pv_l[i][j]) <= self.pv_forecast[i][j])
            
            # 3. 风力出力约束
            for i in range(self.equip_nums[2]):
                mdl.add_constraint(mdl.sum(wind_l[i][j]) <= self.wind_forecast[i][j])


            # 4. 储能约束
            for i in range(self.equip_nums[3]):
                mdl.add_constraint(ess_l[i][j] <= self.ess_max[i])
                mdl.add_constraint(ess_l[i][j] >= 0)
                mdl.add_constraint(p_in[i][j] <= self.p_in_out_max[i])
                mdl.add_constraint(p_out[i][j] <= self.p_in_out_max[i])
                mdl.add_constraint(p_in[i][j] <= self.M * bin[i][j])
                mdl.add_constraint(p_out[i][j] <= self.M * (1 - bin[i][j]))
                if j == 0:
                    mdl.add_constraint(
                        ess_l[ i ][ j ] == self.ess0[ i ] * self.esr[ i ] + p_in[ i ][ j ] * self.cdr[ i ] - p_out[ i ][
                            j ] / self.cdr[ i ])
                else:
                    mdl.add_constraint(
                        ess_l[ i ][ j ] == ess_l[ i ][ j - 1 ] * self.esr[ i ] + p_in[ i ][ j ] * self.cdr[ i ] -
                        p_out[ i ][ j ] /
                        self.cdr[ i ])
                    # 新增：限制单次充放电变化幅度（宽松版）
                    mdl.add_constraint(p_in[ i ][ j ] - p_in[ i ][ j - 1 ] <= 0.6 * self.p_in_out_max[ i ])  # 充电功率变化限制
                    mdl.add_constraint(
                        p_out[ i ][ j ] - p_out[ i ][ j - 1 ] <= 0.6 * self.p_in_out_max[ i ])  # 放电功率变化限制

            # 5. TCR约束
            for i in range(self.equip_nums[4]):
                mdl.add_constraint(tcr_l[i][j] <= self.tcr_max[i])
                mdl.add_constraint(tcr_l[i][j]>= self.tcr_min[i])
                mdl.add_constraint(p_tcr_l[i][j] <= self.tcr_p_max[i])
                mdl.add_constraint(p_tcr_l[i][j] >= 0)
                if j==0:
                    mdl.add_constraint(tcr_l[i][j] == self.tcr_initial[i]*self.tcr_sr[i]+
                    self.tcr_tr[i]*p_tcr_l[i][j]+(1-self.tcr_sr[i])*self.temp_forecast[j])
                else:
                    mdl.add_constraint(tcr_l[i][j] == self.tcr_sr[i]*tcr_l[i][j-1]+
                    self.tcr_tr[i]*p_tcr_l[i][j]+(1-self.tcr_sr[i])*self.temp_forecast[j])

            # 6. DL约束
            for i in range(self.equip_nums[5]):

                mdl.add_constraint(dl_l[i][j] <= self.dl_max[i])
                mdl.add_constraint(dl_l[i][j] >= self.dl_min[i])
                if self.dl_start[i] <= j <= self.dl_end[i]:
                    mdl.add_constraint(p_dl_l[i][j] <= self.dl_p_max[i])
                else:
                    mdl.add_constraint(p_dl_l[i][j] == 0)
                if j > self.dl_end[i]:
                    mdl.add_constraint(dl_l[i][j] >= self.dl_required[i])
                if j == 0:
                    mdl.add_constraint(dl_l[i][j] == self.dl_initial[i] * self.dl_sr[i] +
                                    self.dl_tr[i] * p_dl_l[i][j])
                else:
                    mdl.add_constraint(dl_l[i][j] == dl_l[i][j-1] * self.dl_sr[i] +
                                    self.dl_tr[i] * p_dl_l[i][j])

            
            # 7. 虚拟电厂出力约束
       
            mdl.add_constraint(
                vpp_output[j] == 
                # 发电资源输出
                mdl.sum(dg_l[i][j] for i in range(self.equip_nums[0])) +  # 发电机
                mdl.sum(pv_l[i][j] for i in range(self.equip_nums[1])) +  # 光伏发电
                mdl.sum(wind_l[i][j] for i in range(self.equip_nums[2])) + # 风力发电
                # 储能系统净输出 (充电为负,放电为正)
                mdl.sum(p_out[i][j] - p_in[i][j] for i in range(self.equip_nums[3])) -
                # 可控负荷
                mdl.sum(p_tcr_l[i][j] for i in range(self.equip_nums[4])) - # TCR负荷
                mdl.sum(p_dl_l[i][j] for i in range(self.equip_nums[5])) -  # DL负荷
                # 预测负荷
                self.load_forecast[j]
            )

            # 8. 虚拟电厂出力上下限约束
            # mdl.add_constraint(vpp_output[j] <= self.vpp_max)
            # mdl.add_constraint(vpp_output[j] >= self.vpp_min)

        #9. 添加储能系统充放电的鼓励约束
        for j in range(virtual_power_plant.time_line - 1):
            for i in range(self.equip_nums[3]):  # 遍历所有储能设备
                price_diff = self.price_forecast[j + 1] - self.price_forecast[j]
                if price_diff > 0:
                    # 当下一时段电价上升时,鼓励放电
                    mdl.add_constraint(ess_l[i][j + 1] <= ess_l[i][j] )
                elif price_diff < 0:
                    # 当下一时段电价下降时,鼓励充电
                    mdl.add_constraint(ess_l[i][j + 1] >= ess_l[i][j] )

        # 成本与收益计算
        pure_profit = mdl.sum(
            # 电网收益/成本
            self.price_forecast[k] * vpp_output[k] -
            
            # 发电机成本 (假设使用二次函数表示)
            mdl.sum(
                self.dg_cost[0] * dg_l[i][k] * dg_l[i][k] + 
                self.dg_cost[1] * dg_l[i][k] + 
                self.dg_cost[2]
                for i in range(self.equip_nums[0])
            ) -
            
            # TCR和ESS的运行成本
            mdl.sum(self.tcr_cost * p_tcr_l[i][k] for i in range(self.equip_nums[4])) -
            mdl.sum(self.ess_cost * (p_in[i][k] + p_out[i][k]) for i in range(self.equip_nums[3]))-
            mdl.sum(self.dl_cost*p_dl_l[i][k] for i in range(self.equip_nums[-1]))
            
            for k in range(virtual_power_plant.time_line)
        )

        mdl.maximize(pure_profit)
        solution = mdl.solve()
        mdl.print_solution()
        print(f'final profit:{solution.objective_value} dollor')



        # 获取解决方案中的变量值
        dg_output = [ [ solution.get_value(dg_l[ i ][ j ]) for j in range(virtual_power_plant.time_line) ] for i in
                      range(self.equip_nums[ 0 ]) ]
        pv_output = [ [ solution.get_value(pv_l[ i ][ j ]) for j in range(virtual_power_plant.time_line) ] for i in
                      range(self.equip_nums[ 1 ]) ]
        wind_output = [ [ solution.get_value(wind_l[ i ][ j ]) for j in range(virtual_power_plant.time_line) ] for i in
                        range(self.equip_nums[ 2 ]) ]
        ess_output = [ [ solution.get_value(ess_l[ i ][ j ]) for j in range(virtual_power_plant.time_line) ] for i in
                       range(self.equip_nums[ 3 ]) ]
        tcr_output = [ [ solution.get_value(tcr_l[ i ][ j ]) for j in range(virtual_power_plant.time_line) ] for i in
                       range(self.equip_nums[ 4 ]) ]
        dl_output = [ [ solution.get_value(dl_l[ i ][ j ]) for j in range(virtual_power_plant.time_line) ] for i in
                      range(self.equip_nums[ 5 ]) ]
        vpp_output_values = [ solution.get_value(vpp_output[ j ]) for j in range(virtual_power_plant.time_line) ]

        # 绘制各种输出曲线
        plt.figure(figsize=(12, 8))
        for i in range(self.equip_nums[ 0 ]):
            plt.plot(range(virtual_power_plant.time_line), dg_output[ i ], label=f'DG {i + 1}')
        for i in range(self.equip_nums[ 1 ]):
            plt.plot(range(virtual_power_plant.time_line), pv_output[ i ], label=f'PV {i + 1}')
        for i in range(self.equip_nums[ 2 ]):
            plt.plot(range(virtual_power_plant.time_line), wind_output[ i ], label=f'Wind {i + 1}')
        for i in range(self.equip_nums[ 3 ]):
            plt.plot(range(virtual_power_plant.time_line), ess_output[ i ], label=f'ESS {i + 1}')
        for i in range(self.equip_nums[ 4 ]):
            plt.plot(range(virtual_power_plant.time_line), tcr_output[ i ], label=f'TCR {i + 1}')
        for i in range(self.equip_nums[ 5 ]):
            plt.plot(range(virtual_power_plant.time_line), dl_output[ i ], label=f'DL {i + 1}')
        plt.plot(range(virtual_power_plant.time_line), vpp_output_values, label='VPP Output', linewidth=2,
                 color='black')
        plt.plot(range(virtual_power_plant.time_line), self.load_forecast, label='Load Forecast', linewidth=2,
                 color='red', linestyle='--')

        plt.xlabel('Time')
        plt.ylabel('Power Output')
        plt.title('Virtual Power Plant Output')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('vis/vpp_output.png')
        plt.close()

        # 绘制价格曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(virtual_power_plant.time_line), self.price_forecast, label='Price Forecast')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Price Forecast')
        plt.legend()
        plt.savefig('vis/price_forecast.png')
        plt.close()

        print("Visualization results have been saved in the 'vis' folder.")

       
            
object = virtual_power_plant()
object.create_model()