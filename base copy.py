import os
import sys
import numpy as np
from mip import Model, xsum,  minimize, CBC, OptimizationStatus, BINARY
from time import time
import random
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

    def plot_gantt_johnson(ordem, p):
        import matplotlib.pyplot as plt
        n = len(ordem)
        m = p.shape[1]
        cores = ['tab:blue', 'tab:orange']
        nomes_maquinas = ['Alto Forno', 'LD']
        start_times = np.zeros((n, m))
        for idx, j in enumerate(ordem):
            if idx == 0:
                start_times[idx, 0] = 0
            else:
                start_times[idx, 0] = start_times[idx-1, 0] + p[ordem[idx-1], 0]
            if idx == 0:
                start_times[idx, 1] = start_times[idx, 0] + p[j, 0]
            else:
                start_times[idx, 1] = max(start_times[idx-1, 1] + p[ordem[idx-1], 1], start_times[idx, 0] + p[j, 0])
        jobs = []
        for idx, j in enumerate(ordem):
            for m_id in range(m):
                jobs.append((m_id, start_times[idx, m_id], p[j, m_id], j))
        _, ax = plt.subplots(figsize=(10, 3))
        bar_height = 8
        job_gap = 8  # Espaçamento maior entre os jobs
        for m_id in range(m):
            machine_jobs = [(start, duration, j) for mm, start, duration, j in jobs if mm == m_id]
            for idx2, (start, duration, j) in enumerate(machine_jobs):
                y_base = m_id * (n * (bar_height + job_gap)) + idx2 * (bar_height + job_gap)
                ax.broken_barh([(start, duration)], (y_base, bar_height), facecolors=cores[m_id], label=nomes_maquinas[m_id] if idx2 == 0 else "")
                ax.text(start + duration/2, y_base + bar_height/2, f'Job {j}', va='center', ha='center', color='white', fontsize=8)
        ax.set_yticks([])
        ax.set_yticklabels([])
        plt.tight_layout()
        plt.savefig("gantt_johnson.png")
        print("Gantt chart de Johnson salvo como gantt_johnson.png")
        ax.set_xlabel('Tempo')
        ax.set_title('Diagrama de Gantt (Johnson 2 máquinas)')
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        plt.show()

    
    # Exemplo de uso do algoritmo de Johnson para 2 máquinas:
    if __name__ == "__main__":
        # Supondo que queremos aplicar Johnson para as duas primeiras máquinas
        p2 = instancia.p[:, :2]
        ordem = johnson_flowshop(p2)
        print("Ordem ótima pelo algoritmo de Johnson (2 máquinas):", ordem)
        plot_gantt_johnson(ordem, p2)
        
        def swap_jobs_in_tail(order, idx1, idx2):
            """
            Troca dois jobs de lugar na cauda (lista de jobs).
            order: lista de índices dos jobs (ordem atual)
            idx1, idx2: posições na lista a serem trocadas
            Retorna: nova ordem com os jobs trocados
            """
            new_order = order.copy()
            new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
            return new_order

        # Exemplo de uso:
        # Suponha que queremos trocar os dois últimos jobs da ordem de Johnson
        if len(ordem) >= 2:
            swapped_order = swap_jobs_in_tail(ordem, -2, -1)
            print("Ordem após troca dos dois últimos jobs:", swapped_order)
            plot_gantt_johnson(swapped_order, p2)
            
            def johnson_hybrid_flowshop(p):
                """
                Algoritmo de Johnson adaptado para Hybrid FlowShop com 2 máquinas paralelas no primeiro estágio
                e 1 máquina no segundo estágio.
                p: matriz numpy de tempos de processamento (n_jobs x 2), onde a primeira coluna é o tempo no primeiro estágio
                (máquinas paralelas) e a segunda coluna é o tempo no segundo estágio (única máquina).
                Retorna: ordem dos jobs e alocação nas máquinas paralelas.
                """
                n = p.shape[0]
                # Ordena os jobs pelo algoritmo de Johnson clássico para 2 máquinas
                ordem = johnson_flowshop(p)
                # Aloca os jobs do primeiro estágio nas duas máquinas paralelas de forma balanceada (greedy)
                maquinas = [[], []]  # listas de jobs para cada máquina paralela
                tempos_maquinas = [0, 0]  # tempo acumulado em cada máquina
                for j in ordem:
                    idx = np.argmin(tempos_maquinas)
                    maquinas[idx].append(j)
                    tempos_maquinas[idx] += p[j, 0]
                # Gera a ordem final dos jobs (mantendo a ordem de chegada em cada máquina paralela)
                ordem_final = []
                for m in maquinas:
                    ordem_final.extend(m)
                return ordem_final, maquinas

            def plot_gantt_hybrid(ordem, maquinas, p):
                """
                Plota o diagrama de Gantt para o Hybrid FlowShop com 2 máquinas paralelas no primeiro estágio
                e 1 máquina no segundo estágio.
                ordem: ordem dos jobs para o segundo estágio
                maquinas: lista de listas, jobs em cada máquina paralela do primeiro estágio
                p: matriz de tempos de processamento (n_jobs x 2)
                """
                import matplotlib.pyplot as plt
                n = len(ordem)
                m = p.shape[1]
                cores = ['tab:blue', 'tab:cyan', 'tab:orange']
                nomes_maquinas = ['Paralela 1', 'Paralela 2', 'LD']
                start_times = np.zeros((n, m))
                # Primeiro estágio: máquinas paralelas
                fim_maquinas = [0, 0]
                job_to_machine = {}
                for m_id, jobs in enumerate(maquinas):
                    for idx, j in enumerate(jobs):
                        if idx == 0:
                            start = 0
                        else:
                            start = start_times[jobs[idx-1], 0] + p[jobs[idx-1], 0]
                        start_times[j, 0] = start
                        fim_maquinas[m_id] = start + p[j, 0]
                        job_to_machine[j] = m_id
                # Segundo estágio: LD (sequencial, ordem dada por 'ordem')
                for idx, j in enumerate(ordem):
                    if idx == 0:
                        start = start_times[j, 0] + p[j, 0]
                    else:
                        prev_j = ordem[idx-1]
                        start = max(start_times[j, 0] + p[j, 0], start_times[prev_j, 1] + p[prev_j, 1])
                    start_times[j, 1] = start
                # Monta lista de jobs para plotar
                jobs_plot = []
                for m_id in range(2):  # máquinas paralelas
                    for j in maquinas[m_id]:
                        jobs_plot.append((m_id, start_times[j, 0], p[j, 0], j))
                for idx, j in enumerate(ordem):  # LD
                    jobs_plot.append((2, start_times[j, 1], p[j, 1], j))
                # Plot
                _, ax = plt.subplots(figsize=(12, 4))
                bar_height = 8
                job_gap = 8
                for m_id in range(3):
                    machine_jobs = [(start, duration, j) for mm, start, duration, j in jobs_plot if mm == m_id]
                    for idx2, (start, duration, j) in enumerate(machine_jobs):
                        y_base = m_id * (n * (bar_height + job_gap)) + idx2 * (bar_height + job_gap)
                        ax.broken_barh([(start, duration)], (y_base, bar_height), facecolors=cores[m_id], label=nomes_maquinas[m_id] if idx2 == 0 else "")
                        ax.text(start + duration/2, y_base + bar_height/2, f'Job {j}', va='center', ha='center', color='white', fontsize=8)
                ax.set_yticks([])
                ax.set_yticklabels([])
                plt.tight_layout()
                plt.savefig("gantt_hybrid_johnson.png")
                print("Gantt chart do Hybrid FlowShop salvo como gantt_hybrid_johnson.png")
                ax.set_xlabel('Tempo')
                ax.set_title('Diagrama de Gantt (Hybrid FlowShop Johnson)')
                ax.grid(True)
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
                plt.tight_layout()
                plt.show()

            # Exemplo de uso do algoritmo Hybrid FlowShop Johnson:
            if __name__ == "__main__":
                # Supondo que queremos aplicar Hybrid Johnson para as duas primeiras máquinas (primeira é paralela)
                p2 = instancia.p[:, :2]
                ordem_hybrid, maquinas_hybrid = johnson_hybrid_flowshop(p2)
                print("Ordem dos jobs para o segundo estágio (LD):", ordem_hybrid)
                print("Alocação dos jobs nas máquinas paralelas do primeiro estágio:", maquinas_hybrid)
                plot_gantt_hybrid(ordem_hybrid, maquinas_hybrid, p2)
                
                def random_swap_in_tail_hybrid(ordem, maquinas, num_swaps=1, tail_size=2, seed=None):
                    """
                    Realiza trocas aleatórias entre jobs da cauda da ordem do segundo estágio (LD)
                    na solução do johnson_hybrid_flowshop.
                    ordem: ordem dos jobs para o segundo estágio (LD)
                    maquinas: lista de listas, jobs em cada máquina paralela do primeiro estágio
                    num_swaps: número de trocas a serem realizadas
                    tail_size: tamanho da cauda considerada para trocas
                    seed: semente para reprodutibilidade
                    Retorna: nova ordem dos jobs para o segundo estágio (LD)
                    """
                    if seed is not None:
                        random.seed(seed)
                    n = len(ordem)
                    if tail_size > n:
                        tail_size = n
                    tail_indices = list(range(n - tail_size, n))
                    ordem_swapped = ordem.copy()
                    for _ in range(num_swaps):
                        idx1, idx2 = random.sample(tail_indices, 2)
                        ordem_swapped[idx1], ordem_swapped[idx2] = ordem_swapped[idx2], ordem_swapped[idx1]
                    return ordem_swapped

                # Exemplo de uso:
                if __name__ == "__main__":
                    # Troca aleatória na cauda da solução híbrida de Johnson
                    ordem_hybrid_swapped = random_swap_in_tail_hybrid(ordem_hybrid, maquinas_hybrid, num_swaps=1, tail_size=3, seed=42)
                    print("Ordem após troca aleatória na cauda (Hybrid):", ordem_hybrid_swapped)
                    plot_gantt_hybrid(ordem_hybrid_swapped, maquinas_hybrid, p2)
                    
                    
                    def vns_hybrid_johnson(p, max_iter=100, tail_sizes=[2, 3, 4], seed=None):
                        """
                        Variable Neighborhood Search (VNS) para o Hybrid Johnson.
                        p: matriz numpy de tempos de processamento (n_jobs x 2)
                        max_iter: número máximo de iterações sem melhoria
                        tail_sizes: lista de tamanhos de cauda para os vizinhos
                        seed: semente para reprodutibilidade
                        Retorna: melhor ordem encontrada, alocação nas máquinas paralelas e makespan
                        """
                        if seed is not None:
                            random.seed(seed)
                            np.random.seed(seed)
                        # Solução inicial pelo Hybrid Johnson
                        ordem, maquinas = johnson_hybrid_flowshop(p)
                        best_ordem = ordem.copy()
                        best_maquinas = [m.copy() for m in maquinas]
                        best_makespan = None

                        def calc_makespan(ordem, maquinas, p):
                            n = len(ordem)
                            m = p.shape[1]
                            start_times = np.zeros((n, m))
                            fim_maquinas = [0, 0]
                            for m_id, jobs in enumerate(maquinas):
                                for idx, j in enumerate(jobs):
                                    if idx == 0:
                                        start = 0
                                    else:
                                        start = start_times[jobs[idx-1], 0] + p[jobs[idx-1], 0]
                                    start_times[j, 0] = start
                                    fim_maquinas[m_id] = start + p[j, 0]
                            for idx, j in enumerate(ordem):
                                if idx == 0:
                                    start = start_times[j, 0] + p[j, 0]
                                else:
                                    prev_j = ordem[idx-1]
                                    start = max(start_times[j, 0] + p[j, 0], start_times[prev_j, 1] + p[prev_j, 1])
                                start_times[j, 1] = start
                            makespan = max(start_times[j, 1] + p[j, 1] for j in ordem)
                            return makespan

                        best_makespan = calc_makespan(best_ordem, best_maquinas, p)
                        iter_sem_melhoria = 0

                        while iter_sem_melhoria < max_iter:
                            improved = False
                            for tail_size in tail_sizes:
                                # Gera vizinho por troca aleatória na cauda
                                ordem_viz = random_swap_in_tail_hybrid(best_ordem, best_maquinas, num_swaps=1, tail_size=tail_size)
                                # Mantém a mesma alocação das máquinas paralelas
                                makespan_viz = calc_makespan(ordem_viz, best_maquinas, p)
                                if makespan_viz < best_makespan:
                                    best_ordem = ordem_viz
                                    best_makespan = makespan_viz
                                    improved = True
                                    iter_sem_melhoria = 0
                                    break  # volta para o menor vizinho
                            if not improved:
                                iter_sem_melhoria += 1
                        return best_ordem, best_maquinas, best_makespan

                    # Exemplo de uso do VNS para Hybrid Johnson:
                    if __name__ == "__main__":
                        p2 = instancia.p[:, :2]
                        ordem_vns, maquinas_vns, makespan_vns = vns_hybrid_johnson(p2, max_iter=50, tail_sizes=[2, 3, 4], seed=123)
                        print("Ordem final pelo VNS (Hybrid Johnson):", ordem_vns)
                        print("Alocação nas máquinas paralelas:", maquinas_vns)
                        print("Makespan final:", makespan_vns)
                        plot_gantt_hybrid(ordem_vns, maquinas_vns, p2)