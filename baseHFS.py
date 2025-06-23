import os
import sys
import numpy as np
from mip import Model, xsum,  minimize, CBC, OptimizationStatus, BINARY
from time import time
import matplotlib.pyplot as plt


class RandomInstance:
    def __init__(self, n, seed=None):
        np.random.seed(seed)
        self.n = n
        self.m = 3  # 1: alto forno, 2: LD, 3: descarte
        self.p = np.random.randint(1, 20, size=(n, self.m))  # tempos de processamento
        self.t = np.random.randint(10, 50, size=n)  # limites de tempo para cada job
        self.e = 2

    @staticmethod
    def create_random(n, seed=None):
        return RandomInstance(n, seed)

class CModel():
    def __init__(self,inst):
        self.inst = inst
        self.create_model()

    def create_model(self):
        inst = self.inst
        N = range(inst.n)
        M = range(inst.m)
        Q = range(inst.e) # Vetor com quantidade de máquinas em cada estágio (e.g., 1 for alto forno, 1 for LD, 1 for descarte)
        
        model = Model('Problema de Carros Torpedo',solver_name=CBC)
        # variavel: se o job é feito apos o job i no estágio k
        x = [[[model.add_var(var_type=BINARY, name=f"x_{i}_{j}_{k}") if i != j else None for k in M] for j in N] for i in N]
        # variavel: Makespan maximo
        Cmax = model.add_var(name='Cmax')
        # variavel: se o job j na é alocado na muaquina m no estágio k
        y = [[[model.add_var(var_type=BINARY, name=f"y_{j}_{k}_{m}") for m in Q] for k in M] for j in N]
        # variavel: data de inicio do job j na maquina m
        C = [[model.add_var(name=f'C_{j}_{k}') for k in M] for j in N]
        
        # funcao objetivo: minimizar o tempo de conclusao do ultimo job mais o custo dos jobs descartados
        model.objective = minimize(Cmax)

        # restricao: all operations are assigned strictly to one machine at each stage
        for j in N:
            for k in M:
                model += xsum(y[j][k][m] for m in Q) == 1
                
        # restricao:the starting time of operation ojk to be greater or equal to its release time from the previous stage
        for j in N:
            for k in M:
                if k > 0:
                    model += C[j][k] >= C[j][k-1] + xsum(inst.p[j][k-1] * y[j][k-1][m] for m in Q)
                else:
                    for jp in N:
                        if j != jp:
                            for m in Q:
                                model += C[j][k] >= C[jp][k] + inst.p[jp][k] - 1e6 * (1 - y[j][k][m] - y[jp][k][m] + x[jp][j][k])
                            for m in Q:
                                model += 1e6 * (2 - y[j][k][m] - y[jp][k][m] + x[j][jp][k]) + C[j][k] - C[jp][k] >= inst.p[j][k]
        for j in N:
            for k in M:
                model += Cmax >= C[j][k] + xsum(inst.p[j][k] * y[j][k][m] for m in Q)
            for k in M:
                model += Cmax >= C[j][k] + inst.p[j][k]
        
        # desliga a impressao do solver
        model.verbose = 0
        self.x = x
        self.y = y
        self.model = model

    def run(self):
        inst = self.inst
        N = range(inst.n)
        model,x = self.model,self.x
        # otimiza o modelo chamando o resolvedor
        start = time()
        status = model.optimize()
        end = time()
        # impressao do resultado
        if status == OptimizationStatus.OPTIMAL:
           print("Optimal solution: {:10.2f}".format(model.objective_value))
           print(f'running time    : {end-start:10.2f} s')
           newln = 0
           self.plot_gantt_chart(inst, N)
           return model.objective_value
        return float('-inf')          

    def plot_gantt_chart(self, inst, N):
        # Gantt chart generation (fix: ensure correct job selection and plotting)
        jobs = []
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        machine_names = ['Alto Forno', 'LD', 'Descarte']

        for m in range(inst.m):
            for j in N:
                # Get start time and processing time for job j on machine m
                var = self.model.var_by_name(f'C_{j}_{m}')
                if var is not None and var.x is not None:
                    start = var.x
                    duration = inst.p[j][m]
                    # Only plot if job is not discarded for m=0,1; plot if discarded for m=2
                    if m < 2 and self.y[j].x < 0.5:
                        jobs.append((m, start, duration, j))
                    elif m == 2 and self.y[j].x > 0.5:
                        jobs.append((m, start, duration, j))

        if jobs:
            _, ax = plt.subplots(figsize=(10, 4))
            bar_height = 8
            job_gap = 1
            for m in range(inst.m):
                machine_jobs = [(start, duration, j) for mm, start, duration, j in jobs if mm == m]
                for idx, (start, duration, j) in enumerate(machine_jobs):
                    y_base = m * 15 + idx * (bar_height + job_gap)
                    ax.broken_barh([(start, duration)], (y_base, bar_height), facecolors=colors[m], label=machine_names[m] if idx == 0 else "")
                    ax.text(start + duration/2, y_base + bar_height/2, f'Job {j}', va='center', ha='center', color='white', fontsize=8)

            # Remove y-axis ticks and labels
            ax.set_yticks([])
            ax.set_yticklabels([])
            # Save the plot to a file instead of showing it interactively
            plt.tight_layout()
            plt.savefig("gantt_chart.png")
            print("Gantt chart saved as gantt_chart.png")
            ax.set_xlabel('Tempo')
            ax.set_title('Diagrama de Gantt')
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.tight_layout()
            plt.show()
            print('\n\n')
        else:
            print("Nenhum job para exibir no gráfico de Gantt.")

    def swap_positions(self, result_list, pos1, pos2):
            """
        Troca os elementos nas posições pos1 e pos2 de result_list.
        """
            if not (0 <= pos1 < len(result_list)) or not (0 <= pos2 < len(result_list)):
                raise IndexError("Posições fora do intervalo da lista.")
            result_list[pos1], result_list[pos2] = result_list[pos2], result_list[pos1]
            return result_list

if __name__ == "__main__":
    # Exemplo de uso
    n = 3# número de jobs
    seed = 4  # para reprodutibilidade
    instancia = RandomInstance.create_random(n, seed)
    modelo = CModel(instancia)
    resultado = modelo.run()
    print(f"Resultado final: {resultado}")