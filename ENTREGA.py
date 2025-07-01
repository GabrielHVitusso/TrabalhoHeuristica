# Aluno: Gabriel Hesse Vitusso 2022056242

# Bibliotecas necessárias
import os
import sys
import numpy as np
from mip import Model, xsum,  minimize, CBC, OptimizationStatus, BINARY
from time import time
import random
from itertools import permutations
import pandas as pd
import matplotlib.pyplot as plt

 # Modelo de Programação Linear
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

# Métodos de Heuristica e Metaheuristica

def johnson_hybrid_flowshop(p, t_limite=50, n_passes=3):
                """
                Algoritmo de Johnson adaptado para Hybrid FlowShop com 3 centros de máquinas:
                - Centro 1: 1 máquina (sequencial)
                - Centro 2: 2 máquinas paralelas
                - Centro 3: 1 máquina (sequencial), cada job deve passar n_passes vezes
                p: matriz numpy de tempos de processamento (n_jobs x 3), colunas: [centro1, centro2, centro3]
                t_limite: vetor/lista/array de limites de tempo para cada job no centro 2 (opcional)
                n_passes: número de vezes que cada job deve passar pelo centro 3
                Retorna:
                    ordem_final: ordem dos jobs para o centro 3 (sequencial)
                    maquinas_c2: lista de listas, jobs em cada máquina paralela do centro 2
                    start_times: matriz (n_jobs x 3) com tempos de início da última passagem em cada centro
                    descartados: lista de índices dos jobs descartados por violar o limite de tempo no centro 2
                """
                n = p.shape[0]
                # Se t_limite for um inteiro, converte para array do mesmo valor para todos os jobs
                if isinstance(t_limite, int):
                    t_limite = np.full(n, t_limite)
                # 1. Ordena os jobs pelo algoritmo de Johnson clássico para centros 1 e 3
                p_johnson = np.column_stack((p[:, 0], p[:, 2]))
                ordem = johnson_flowshop(p_johnson)
                # 2. Simula o fluxo:
                # Centro 1: sequencial, ordem = ordem de Johnson
                start_times = np.zeros((n, 3))
                for idx, j in enumerate(ordem):
                    if idx == 0:
                        start_times[j, 0] = 0
                    else:
                        prev_j = ordem[idx - 1]
                        start_times[j, 0] = start_times[prev_j, 0] + p[prev_j, 0]
                # 3. Centro 2: 2 máquinas paralelas, alocação greedy conforme chegada do centro 1
                maquinas_c2 = [[], []]
                fim_maquinas_c2 = [0, 0]
                descartados = []
                centro2_entrada = np.zeros(n)
                centro2_saida = np.zeros(n)
                tempo_total_centro2 = np.zeros(n)
                for j in ordem:
                    ready_time = start_times[j, 0] + p[j, 0]
                    idx_m = np.argmin([max(fim_maquinas_c2[m], ready_time) for m in range(2)])
                    start_times[j, 1] = max(fim_maquinas_c2[idx_m], ready_time)
                    centro2_entrada[j] = start_times[j, 1]
                    fim_maquinas_c2[idx_m] = start_times[j, 1] + p[j, 1]
                    centro2_saida[j] = fim_maquinas_c2[idx_m]
                    maquinas_c2[idx_m].append(j)
                    # Acumula o tempo total que o job ficou no centro 2 (tempo de processamento)
                    tempo_total_centro2[j] += p[j, 1]
                # 4. Verifica restrição de tempo no centro 2 (tempo total de processamento no centro 2)
                for j in ordem:
                    if t_limite is not None and tempo_total_centro2[j] > t_limite[j]:
                        descartados.append(j)
                # 5. Centro 3: sequencial, ordem = ordem de Johnson, mas pula descartados, cada job passa n_passes vezes
                # Nova versão: simula n_passes alternando entre centro 2 (paralelo) e centro 3 (sequencial)
                ordem_final = [j for j in ordem if j not in descartados]
                # Inicializa tempos de início para cada job, cada passagem
                # start_times[job][pass][centro]
                n_jobs = p.shape[0]
                start_times_full = np.zeros((n_jobs, n_passes + 1, 3))
                # Centro 1: sequencial, ordem = ordem de Johnson
                for idx, j in enumerate(ordem_final):
                    if idx == 0:
                        start_times_full[j, 0, 0] = 0
                    else:
                        prev_j = ordem_final[idx - 1]
                        start_times_full[j, 0, 0] = start_times_full[prev_j, 0, 0] + p[prev_j, 0]
                # Para cada passagem (do centro 2 e 3)
                fim_maquinas_c2 = [0, 0]
                for pass_num in range(n_passes):
                    # Centro 2: 2 máquinas paralelas, alocação greedy conforme chegada do centro 1 ou centro 3
                    for j in ordem_final:
                        if pass_num == 0:
                            ready_time = start_times_full[j, 0, 0] + p[j, 0]
                        else:
                            ready_time = start_times_full[j, pass_num, 2] + p[j, 2]
                        idx_m = np.argmin([max(fim_maquinas_c2[m], ready_time) for m in range(2)])
                        start_times_full[j, pass_num, 1] = max(fim_maquinas_c2[idx_m], ready_time)
                        fim_maquinas_c2[idx_m] = start_times_full[j, pass_num, 1] + p[j, 1]
                    # Centro 3: sequencial, ordem = ordem_final
                    prev_end = 0
                    for idx, j in enumerate(ordem_final):
                        ready_time = start_times_full[j, pass_num, 1] + p[j, 1]
                        if idx == 0:
                            start = ready_time
                        else:
                            prev_j = ordem_final[idx - 1]
                            start = max(ready_time, start_times_full[prev_j, pass_num + 1, 2] + p[prev_j, 2])
                        start_times_full[j, pass_num + 1, 2] = start
                # Atualiza start_times para refletir o início da última passagem no centro 3
                for j in ordem_final:
                    start_times[j, 2] = start_times_full[j, n_passes, 2]
                return ordem_final, maquinas_c2, start_times, descartados

def plot_gantt_hybrid(ordem, maquinas, p, n_passes=3):
                """
                Plota o diagrama de Gantt para o Hybrid FlowShop com 3 centros de máquinas,
                sendo o segundo centro com 2 máquinas paralelas.
                Cada job é mostrado em uma única linha, com barras coloridas para cada estágio/passagem.
                ordem: ordem dos jobs para o terceiro centro (sequencial)
                maquinas: lista de listas, jobs em cada máquina paralela do segundo centro
                p: matriz de tempos de processamento (n_jobs x 3)
                n_passes: número de vezes que cada job passa entre centro 2 e 3
                """
                import matplotlib.pyplot as plt
                n = len(ordem)
                m = p.shape[1]
                total_jobs = p.shape[0]
                cores = ['tab:orange', 'tab:cyan', 'tab:blue', 'tab:green']
                nomes_maquinas = ['Alto Forno', 'Paralela 1 (Centro 2)', 'Paralela 2 (Centro 2)', 'LD']
                # Inicializa tempos de início para cada job, cada passagem
                # start_times[job][pass][centro]
                start_times = np.zeros((total_jobs, n_passes + 1, m))
                # Centro 1: sequencial, ordem = ordem de Johnson
                for idx, j in enumerate(ordem):
                    if idx == 0:
                        start_times[j, 0, 0] = 0
                    else:
                        prev_j = ordem[idx - 1]
                        start_times[j, 0, 0] = start_times[prev_j, 0, 0] + p[prev_j, 0]
                # Para cada passagem (do centro 2 e 3)
                fim_maquinas_c2 = [0, 0]
                centro2_aloc = np.full((total_jobs, n_passes), -1)
                for pass_num in range(n_passes):
                    # Centro 2: 2 máquinas paralelas, alocação conforme 'maquinas'
                    for m_id, jobs_m in enumerate(maquinas):
                        for idx, j in enumerate(jobs_m):
                            if pass_num == 0:
                                ready_time = start_times[j, 0, 0] + p[j, 0]
                            else:
                                ready_time = start_times[j, pass_num - 1, 2] + p[j, 2]
                            if idx == 0 and pass_num == 0:
                                start = ready_time
                            else:
                                start = max(fim_maquinas_c2[m_id], ready_time)
                            start_times[j, pass_num, 1] = start
                            fim_maquinas_c2[m_id] = start + p[j, 1]
                            centro2_aloc[j, pass_num] = m_id
                    # Centro 3: sequencial, ordem = ordem de Johnson
                    for idx, j in enumerate(ordem):
                        ready_time = start_times[j, pass_num, 1] + p[j, 1]
                        if idx == 0 and pass_num == 0:
                            start = ready_time
                        elif pass_num == 0:
                            prev_j = ordem[idx - 1]
                            start = max(ready_time, start_times[prev_j, pass_num, 2] + p[prev_j, 2])
                        else:
                            prev_j = ordem[idx - 1]
                            start = max(ready_time, start_times[prev_j, pass_num, 2] + p[prev_j, 2])
                        start_times[j, pass_num, 2] = start
                # Centro 1 (sequencial, só 1 vez)
                for idx, j in enumerate(ordem):
                    pass  # já computado

                # Plot: cada job em uma linha
                _, ax = plt.subplots(figsize=(14, max(4, 0.7 * total_jobs)))
                bar_height = 0.6
                yticks = []
                ylabels = []
                for idx, j in enumerate(ordem):
                    y = idx
                    yticks.append(y + bar_height / 2)
                    ylabels.append(f'Job {j}')
                    # Centro 1
                    ax.broken_barh(
                        [(start_times[j, 0, 0], p[j, 0])],
                        (y, bar_height),
                        facecolors=cores[0],
                        label=nomes_maquinas[0] if idx == 0 else ""
                    )
                    # Centro 2 e 3, para cada passagem
                    for pass_num in range(n_passes):
                        # Centro 2
                        m_id = centro2_aloc[j, pass_num]
                        cor_c2 = cores[1 + m_id] if m_id >= 0 else 'gray'
                        ax.broken_barh(
                            [(start_times[j, pass_num, 1], p[j, 1])],
                            (y, bar_height),
                            facecolors=cor_c2,
                            label=nomes_maquinas[1 + m_id] if idx == 0 and pass_num == 0 and m_id >= 0 else ""
                        )
                        # Centro 3
                        ax.broken_barh(
                            [(start_times[j, pass_num, 2], p[j, 2])],
                            (y, bar_height),
                            facecolors=cores[3],
                            label=nomes_maquinas[3] if idx == 0 and pass_num == 0 else ""
                        )
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabels)
                ax.set_xlabel('Tempo')
                ax.set_title(f'Diagrama de Gantt (Hybrid FlowShop Johnson, {n_passes} passagens)')
                ax.grid(True, axis='x')
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right')
                plt.tight_layout()
                plt.savefig("gantt_hybrid_johnson.png")
                print("Gantt chart do Hybrid FlowShop salvo como gantt_hybrid_johnson.png")

def random_swap_in_tail_hybrid(ordem, maquinas, num_swaps=1, tail_size=2, n_passes=3, seed=None):
                    """
                    Realiza trocas aleatórias entre jobs da cauda da ordem do terceiro centro (LD)
                    na solução do johnson_hybrid_flowshop, considerando a repetitividade (n_passes) do centro 3.
                    ordem: ordem dos jobs para o terceiro centro (LD)
                    maquinas: lista de listas, jobs em cada máquina paralela do segundo centro
                    num_swaps: número de trocas a serem realizadas
                    tail_size: tamanho da cauda considerada para trocas
                    n_passes: número de vezes que cada job passa pelo centro 3
                    seed: semente para reprodutibilidade
                    Retorna: nova ordem dos jobs para o terceiro centro (LD)
                    """
                    if seed is not None:
                        random.seed(seed)
                    n = len(ordem)
                    if tail_size > n:
                        tail_size = n
                    tail_indices = list(range(n - tail_size, n))
                    ordem_swapped = ordem.copy()
                    for _ in range(num_swaps):
                        if len(tail_indices) >= 2:
                            idx1, idx2 = random.sample(tail_indices, 2)
                            ordem_swapped[idx1], ordem_swapped[idx2] = ordem_swapped[idx2], ordem_swapped[idx1]
                            # Garante que não há jobs repetidos e nenhum job descartado é reinserido sem controle
                            ordem_swapped_no_overlap = [j for j in ordem_swapped if ordem_swapped.count(j) == 1]
                            if len(ordem_swapped_no_overlap) == len(ordem_swapped):
                                return ordem_swapped_no_overlap
                    return ordem_swapped

                # Exemplo de uso:

def johnson_flowshop(p):
        """
        Implementação do algoritmo de Johnson para flowshop de 2 máquinas.
        p: matriz numpy de tempos de processamento (n_jobs x 2)
        Retorna: ordem ótima dos jobs
        """
        n, m = p.shape
        if m != 2:
            raise ValueError("O algoritmo de Johnson clássico só é aplicável para 2 máquinas.")
        jobs = list(range(n))
        left = []
        right = []
        while jobs:
            times = [(j, p[j,0], p[j,1]) for j in jobs]
            j_min, p1, p2 = min(times, key=lambda x: min(x[1], x[2]))
            if p1 <= p2:
                left.append(j_min)
            else:
                right.insert(0, j_min)
            jobs.remove(j_min)
        return left + right

def calc_makespan(ordem, maquinas, p, descartados, n_passes=3):
                        """
                        Calcula o makespan considerando os jobs descartados e repetitividade do centro 3.
                        Para cada job descartado, adiciona 1e6 ao makespan.
                        n_passes: número de vezes que cada job passa pelo centro 3
                        """
                        n = len(ordem)
                        m = p.shape[1]
                        total_jobs = p.shape[0]
                        # start_times[job][pass][centro]
                        start_times = np.zeros((total_jobs, n_passes + 1, m))
                        # Centro 1: sequencial, ordem = ordem de Johnson
                        for idx, j in enumerate(ordem):
                            if idx == 0:
                                start_times[j, 0, 0] = 0
                            else:
                                prev_j = ordem[idx - 1]
                                start_times[j, 0, 0] = start_times[prev_j, 0, 0] + p[prev_j, 0]
                        # Para cada passagem (do centro 2 e 3)
                        fim_maquinas_c2 = [0, 0]
                        fim_centro3 = 0
                        for pass_num in range(n_passes):
                            # Centro 2: 2 máquinas paralelas, alocação conforme 'maquinas'
                            for m_id, jobs_m in enumerate(maquinas):
                                for idx, j in enumerate(jobs_m):
                                    # O job só entra no centro 2 após terminar o centro 1 (na primeira passagem)
                                    # ou após terminar o centro 3 (nas passagens seguintes)
                                    if pass_num == 0:
                                        ready_time = start_times[j, 0, 0] + p[j, 0]
                                    else:
                                        ready_time = start_times[j, pass_num - 1, 2] + p[j, 2]
                                    if idx == 0 and pass_num == 0:
                                        start = ready_time
                                    else:
                                        start = max(fim_maquinas_c2[m_id], ready_time)
                                    start_times[j, pass_num, 1] = start
                                    fim_maquinas_c2[m_id] = start + p[j, 1]
                            # Centro 3: sequencial, ordem = ordem de Johnson
                            for idx, j in enumerate(ordem):
                                ready_time = start_times[j, pass_num, 1] + p[j, 1]
                                if idx == 0 and pass_num == 0:
                                    start = ready_time
                                elif pass_num == 0:
                                    prev_j = ordem[idx - 1]
                                    start = max(ready_time, start_times[prev_j, pass_num, 2] + p[prev_j, 2])
                                else:
                                    prev_j = ordem[idx - 1]
                                    start = max(ready_time, start_times[prev_j, pass_num, 2] + p[prev_j, 2])
                                start_times[j, pass_num, 2] = start
                        # O makespan é o término da última passagem do último job
                        makespan = 0
                        if ordem:
                            last_job = ordem[-1]
                            makespan = start_times[last_job, n_passes - 1, 2] + p[last_job, 2]
                        makespan += 1e6 * len(descartados)
                        return makespan

def vns_hybrid_johnson(p, max_iter=100, tail_sizes=[2, 3, 4], n_passes=3, seed=None):
                        """
                        Variable Neighborhood Search (VNS) para o Hybrid Johnson considerando repetitividade do centro 3.
                        Usa random_swap_in_tail_hybrid como movimento de vizinhança.
                        p: matriz numpy de tempos de processamento (n_jobs x 3)
                        max_iter: número máximo de iterações sem melhoria
                        tail_sizes: lista de tamanhos de cauda para os vizinhos
                        n_passes: número de vezes que cada job passa pelo centro 3
                        seed: semente para reprodutibilidade
                        Retorna: melhor ordem encontrada, alocação nas máquinas paralelas e makespan
                        """
                        if seed is not None:
                            random.seed(seed)
                            np.random.seed(seed)
                        # Solução inicial pelo Hybrid Johnson
                        ordem, maquinas, start_times, descartados = johnson_hybrid_flowshop(p, n_passes=n_passes)
                        best_ordem = ordem.copy()
                        best_maquinas = [m.copy() for m in maquinas]
                        best_descartados = descartados.copy()

                        def calc_makespan(ordem, maquinas, p, descartados, n_passes=3):
                            n = len(ordem)
                            m = p.shape[1]
                            total_jobs = p.shape[0]
                            # Ajuste: permite n_passes arbitrário
                            start_times = np.zeros((total_jobs, n_passes + 1, m))
                            # Centro 1: sequencial
                            for idx, j in enumerate(ordem):
                                if idx == 0:
                                    start_times[j, 0, 0] = 0
                                else:
                                    prev_j = ordem[idx - 1]
                                    start_times[j, 0, 0] = start_times[prev_j, 0, 0] + p[prev_j, 0]
                            # Para cada passagem (do centro 2 e 3)
                            for pass_num in range(n_passes):
                                fim_maquinas_c2 = [0, 0]
                                # Centro 2: 2 máquinas paralelas, alocação conforme 'maquinas'
                                for m_id, jobs_m in enumerate(maquinas):
                                    for idx, j in enumerate(jobs_m):
                                        if pass_num == 0:
                                            ready_time = start_times[j, 0, 0] + p[j, 0]
                                        else:
                                            ready_time = start_times[j, pass_num - 1, 2] + p[j, 2]
                                        if idx == 0 and pass_num == 0:
                                            start = ready_time
                                        else:
                                            start = max(fim_maquinas_c2[m_id], ready_time)
                                        start_times[j, pass_num, 1] = start
                                        fim_maquinas_c2[m_id] = start + p[j, 1]
                                # Centro 3: sequencial, ordem = ordem de Johnson
                                for idx, j in enumerate(ordem):
                                    ready_time = start_times[j, pass_num, 1] + p[j, 1]
                                    if idx == 0 and pass_num == 0:
                                        start = ready_time
                                    elif pass_num == 0:
                                        prev_j = ordem[idx - 1]
                                        start = max(ready_time, start_times[prev_j, pass_num, 2] + p[prev_j, 2])
                                    else:
                                        prev_j = ordem[idx - 1]
                                        start = max(ready_time, start_times[prev_j, pass_num, 2] + p[prev_j, 2])
                                    start_times[j, pass_num, 2] = start
                            makespan = 0
                            if ordem:
                                last_job = ordem[-1]
                                makespan = start_times[last_job, n_passes - 1, 2] + p[last_job, 2]
                            makespan += 1e6 * len(descartados)
                            return makespan

                        best_makespan = calc_makespan(best_ordem, best_maquinas, p, best_descartados, n_passes=n_passes)
                        iter_sem_melhoria = 0

                        while iter_sem_melhoria < max_iter:
                            improved = False
                            for tail_size in tail_sizes:
                                # Gera vizinho usando random_swap_in_tail_hybrid
                                ordem_viz = random_swap_in_tail_hybrid(
                                    best_ordem, best_maquinas, num_swaps=100, tail_size=tail_size, n_passes=n_passes, seed=None
                                )
                                # Recalcula solução híbrida a partir da nova ordem
                                # Para garantir consistência, roda johnson_hybrid_flowshop com a ordem vizinha
                                # mas força a ordem do centro 3 (LD) para ordem_viz
                                # Para isso, simula o fluxo com a ordem_viz
                                # (reaproveita lógica do johnson_hybrid_flowshop, mas força ordem do centro 3)
                                # Aqui, para simplificação, apenas recalcula os descartados com a nova ordem
                                _, maquinas_viz, _, descartados_viz = johnson_hybrid_flowshop(p, n_passes=n_passes)
                                
                                # Nova variável tridimensional para tempos de início
                                total_jobs = p.shape[0]
                                m = p.shape[1]
                                start_times_viz_3d = np.zeros((total_jobs, n_passes + 1, m))
                                # Centro 1: sequencial, ordem = ordem_viz
                                for idx, j in enumerate(ordem_viz):
                                    if idx == 0:
                                        start_times_viz_3d[j, 0, 0] = 0
                                    else:
                                        prev_j = ordem_viz[idx - 1]
                                        start_times_viz_3d[j, 0, 0] = start_times_viz_3d[prev_j, 0, 0] + p[prev_j, 0]
                                # Para cada passagem (do centro 2 e 3)
                                fim_maquinas_c2 = [0, 0]
                                for pass_num in range(n_passes):
                                    # Centro 2: 2 máquinas paralelas, alocação conforme 'maquinas_viz'
                                    for m_id, jobs_m in enumerate(maquinas_viz):
                                        for idx, j in enumerate(jobs_m):
                                            if pass_num == 0:
                                                ready_time = start_times_viz_3d[j, 0, 0] + p[j, 0]
                                            else:
                                                ready_time = start_times_viz_3d[j, pass_num - 1, 2] + p[j, 2]
                                            if idx == 0 and pass_num == 0:
                                                start = ready_time
                                            else:
                                                start = max(fim_maquinas_c2[m_id], ready_time)
                                            start_times_viz_3d[j, pass_num, 1] = start
                                            fim_maquinas_c2[m_id] = start + p[j, 1]
                                    # Centro 3: sequencial, ordem = ordem_viz
                                    for idx, j in enumerate(ordem_viz):
                                        ready_time = start_times_viz_3d[j, pass_num, 1] + p[j, 1]
                                        if idx == 0 and pass_num == 0:
                                            start = ready_time
                                        elif pass_num == 0:
                                            prev_j = ordem_viz[idx - 1]
                                            start = max(ready_time, start_times_viz_3d[prev_j, pass_num, 2] + p[prev_j, 2])
                                        else:
                                            prev_j = ordem_viz[idx - 1]
                                            start = max(ready_time, start_times_viz_3d[prev_j, pass_num, 2] + p[prev_j, 2])
                                        start_times_viz_3d[j, pass_num, 2] = start

                                # Remove jobs descartados da ordem vizinha
                                ordem_viz_valid = [j for j in ordem_viz if j not in descartados_viz]
                                makespan_viz = calc_makespan(ordem_viz_valid, maquinas_viz, p, descartados_viz, n_passes=n_passes)
                                # Checa overlap nas máquinas paralelas do centro 2
                                overlap = False
                                for m_id, jobs_m in enumerate(maquinas_viz):
                                    job_intervals = []
                                    for idx, j in enumerate(jobs_m):
                                        # Para múltiplas passagens, pega todos os intervalos de centro 2
                                        for pass_num in range(n_passes):
                                            if pass_num == 0:
                                                ready_time = start_times_viz_3d[j, 0, 0] + p[j, 0]
                                            else:
                                                ready_time = start_times_viz_3d[j, pass_num - 1, 2] + p[j, 2]
                                            start = start_times_viz_3d[j, pass_num, 1]
                                            end = start + p[j, 1]
                                            job_intervals.append((start, end))
                                    # Ordena intervalos por tempo de início
                                    job_intervals.sort()
                                    for k in range(1, len(job_intervals)):
                                        if job_intervals[k][0] < job_intervals[k-1][1] - 1e-8:
                                            overlap = True
                                            break
                                    if overlap:
                                        break
                                if not overlap and makespan_viz < best_makespan:
                                    best_ordem = ordem_viz_valid
                                    best_maquinas = [m.copy() for m in maquinas_viz]
                                    best_descartados = descartados_viz.copy()
                                    best_makespan = makespan_viz
                                    improved = True
                                    iter_sem_melhoria = 0
                                    break  # volta para o menor vizinho
                            if not improved:
                                iter_sem_melhoria += 1
                        return best_ordem, best_maquinas, best_makespan, best_descartados

                    # Exemplo de uso do VNS para Hybrid Johnson:

class RandomInstance:
    def __init__(self, filename='instancia_testes.csv'):
        # Lê a tabela CSV e registra os dados
        df = pd.read_csv(filename, index_col='Job')
        # Ajusta para pegar apenas as colunas relevantes (ajuste conforme necessário)
        self.p = df[['Alto Forno', 'carro', 'LD']].to_numpy()
        self.t = df['Limite_Tempo'].to_numpy()
        self.n = self.p.shape[0]
        self.m = self.p.shape[1]

    @staticmethod
    def create_from_csv(filename='instancia_testes.csv'):
        return RandomInstance(filename)

if __name__ == "__main__":
    # Exemplo de uso
    instancia = RandomInstance.create_from_csv('instancia_jobs.csv')
    modelo = CModel(instancia)
    resultado = modelo.run()
    print(f"Resultado final: {resultado}")
    
    
    # Supondo que queremos aplicar Hybrid Johnson para o caso em que somente a segunda máquina é paralela
    p3 = instancia.p  # matriz completa de 3 colunas
    ordem_hybrid, maquinas_hybrid, start_times_hybrid, descartados = johnson_hybrid_flowshop(p3)
    print("Ordem dos jobs para o segundo estágio (LD):", ordem_hybrid)
    print("Alocação dos jobs nas máquinas paralelas do primeiro estágio:", maquinas_hybrid)
    print("Descartados:", descartados)
    #plot_gantt_hybrid(ordem_hybrid, maquinas_hybrid, p3)
    
    
    # Troca aleatória na cauda da solução híbrida de Johnson
    ordem_hybrid_swapped = random_swap_in_tail_hybrid(ordem_hybrid, maquinas_hybrid, num_swaps=100, tail_size=3, seed=42)
    print("Ordem após troca aleatória na cauda (Hybrid):", ordem_hybrid_swapped)
    #plot_gantt_hybrid(ordem_hybrid_swapped, maquinas_hybrid, p3)

    makespan_swapped = calc_makespan(ordem_hybrid_swapped, maquinas_hybrid, p3, descartados, n_passes=3)
    print("Makespan após troca aleatória na cauda (Hybrid):", makespan_swapped)
    
    
    p3 = instancia.p  # matriz completa de 3 colunas
    ordem_vns, maquinas_vns, makespan_vns, descartados_vns = vns_hybrid_johnson(p3, max_iter=50, tail_sizes=[2, 3, 4], n_passes=5, seed=42)
    print("Ordem final pelo VNS (Hybrid Johnson):", ordem_vns)
    print("Alocação nas máquinas paralelas:", maquinas_vns)
    print("Makespan final:", makespan_vns)
    print("Descartados:", descartados_vns)
                        
    plot_gantt_hybrid(ordem_vns, maquinas_vns, p3, n_passes=5)