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
        self.m = 5  # 1: alto forno, 2: LD, 3: descarte
        self.p = np.random.randint(1, 20, size=(n, self.m))  # tempos de processamento
        self.t = np.random.randint(2, 6, size=n)  # limites de tempo para cada job

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
        
        model = Model('Problema de Carros Torpedo',solver_name=CBC)
        # variavel: se o job é feito apos o job i na maquina m
        x = [[[model.add_var(var_type=BINARY) if i != j else None for m in M] for j in N] for i in N]
        # variavel: Makespan maximo
        Cmax = model.add_var(name='Cmax')
        # variavel: se o job j é descartado
        y = [model.add_var(var_type=BINARY) for j in N]
        # variavel: data de inicio do job j na maquina m
        C = [[model.add_var(name=f'C_{j}_{m}') for m in M] for j in N]
        
        # funcao objetivo: minimizar o tempo de conclusao do ultimo job mais o custo dos jobs descartados
        model.objective = minimize(Cmax + xsum(1e6 *y[j] for j in N))

        # restricao: Makespan deve ser maior ou igual ao tempo de conclusao do job j em LD
        for j in N:
            model += Cmax >= C[j][1] + inst.p[j][1] - 1e6 * y[j]
        # restricao: o tempo de conclusao do job j no alto forno é maior ou igual ao tempo de inicio mais o tempo de processamento
        for j in N:
            for i in N:
                if i != j:
                    # m=1 for alto forno
                    model += C[j][0] >= C[i][0] + inst.p[i][0] - 1e6 * x[i][j][0]
                else:
                    model += C[j][0] >= 0

        # restricao: o tempo de conclusao do job j em LD é maior ou igual ao tempo de inicio mais o tempo de processamento
        for j in N:
            for i in N:
                if i != j:
                    # m=2 for LD
                    model += C[j][1] >= C[i][1] + inst.p[i][1] - 1e6 * x[i][j][1] - 1e6 * y[j]
                else:
                    model += C[j][1] >= 0

        # restricao: em um dado job j, o tempo de inicio em LD deve ser maior ou igual ao tempo de conclusao no alto forno
        for j in N:
            model += C[j][1] >= C[j][0] + inst.p[j][0] - 1e6 * y[j]

        # restricao: cada job só pode ser feito uma vez
        for m in M:
            for j in N:
                for i in N:
                    if i != j:
                        model += x[i][j][m] + x[j][i][m] <= 1  # cada job só pode ser feito uma vez na mesma máquina


        # restricao: restricao de descarte dos jobs
        for j in N:
            for i in N:
                # Only add the constraint if x[i][j][2] is not None (i != j)
                if x[i][j][2] is not None:
                    model += C[j][2] >= C[i][2] + inst.p[i][2] - 1e6 * x[i][j][2] - 1e6 * y[j] - 1e6 * y[i]

        #for j in N:
        #    for i in N:
        #        if i != j:
        #            C[j][0] >= C[i][1] + inst.p[i][1] * (1-y[i]) + inst.p[i][2]*y[i]

        # restricao: limite de tempo de processamento
        for j in N:
            model += C[j][1]+ inst.p[j][1] <= C[j][2] + inst.t[j] + 1e6 * y[j]

        # restricao: Makespan deve ser maior ou igual ao tempo de conclusao do job j em LD
        for j in N:
            model += Cmax >= C[j][1] + inst.p[j][1] - 1e6 * y[j]

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
            print("\nVariáveis de decisão:")
            # Variáveis x[i][j][m]
            print("\nx[i][j][m]:")
            for i in N:
                for j in N:
                    for m in range(self.inst.m):
                        if i != j and self.x[i][j][m] is not None and self.x[i][j][m].x > 0.5:
                            print(f"x[{i}][{j}][{m}] = 1")
            # Variáveis y[j]
            print("\ny[j]:")
            for j in N:
                print(f"y[{j}] = {int(round(self.y[j].x))}")
            # Variáveis C[j][m]
            print("\nC[j][m]:")
            for j in N:
                for m in range(self.inst.m):
                    var = self.model.var_by_name(f"C_{j}_{m}")
                    if var is not None and var.x is not None:
                        print(f"C[{j}][{m}] = {var.x:.2f}")
            # Makespan
            print(f"\nCmax = {self.model.var_by_name('Cmax').x:.2f}")
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
if __name__ == "__main__":
    # Exemplo de uso
    n = 5# número de jobs
    seed = 4  # para reprodutibilidade
    instancia = RandomInstance.create_random(n, seed)
    modelo = CModel(instancia)
    resultado = modelo.run()
    print(f"Resultado final: {resultado}")