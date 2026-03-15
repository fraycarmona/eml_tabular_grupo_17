from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
import gymnasium as gym


class FrozenAgentBasic:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, env, state: tuple[int]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[state]))

    def update(
        self,
        state: tuple[int],
        action: int,
        reward: float,
        terminated: bool,
        next_state: tuple[int],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_state])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[state][action]
        )

        self.q_values[state][action] = (
            self.q_values[state][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class FrozenAgentGreedy:
    def __init__(
        self,
        env,
		epsilon: float,
        discount_factor: float = 0.95,
    ):
        """
        Args:
            epsilon: The initial epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        #almacenamos los valores iniciales
        self.descripcion=f"FrozenAgentGreedy. epsilon={epsilon} discount_factor={discount_factor}"
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self.env = env

        #incializo el agente con estos parametros
        self.initAgent()
        
    def __str__(self):
        return self.descripcion

    def initAgent(self):
        #inicializa el agente con la configuración inicial
        #parametros del epsilon greedy y su decaimiento
        self.epsilon = self._epsilon
        self.discount_factor = self._discount_factor
        
        # Matriz de valores  Q
        self.nA = self.env.action_space.n
        self.Q = np.zeros([self.env.observation_space.n, self.nA])

        # Número de visitas. Vamos a realizar la versión incremental.
        self.n_visits = np.zeros([self.env.observation_space.n, self.nA])
        
        #para hacer trackiong del episodio
        self.episode=[]
        self.result_sum=0.0
        self.factor=1.0

        # Para mostrar la evolución en el terminal y algún dato que mostrar
        self.stats = 0.0
        self.episodes = 0.0
        self.list_stats = [self.stats]
        self.list_episodes = [self.episodes]
        
        #para hacer tracking del total de episodios
        self.numEpisodes=0

    def initEpisode(self):
        #Inicializa el agente con unnuevo episodio
        self.episode=[]
        self.result_sum=0.0
        self.factor=1.0

    def get_action(self, env, state: tuple[int]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        return self.epsilon_greedy_policy(state)        

    def updateStep(self, state: tuple[int], action: int, reward: float, terminated: bool, next_state: tuple[int]):
        """Actualiza a nivel de step"""
        self.episode.append((state, action))
        self.result_sum += self.factor * reward
        self.factor *= self.discount_factor

    def updateEpisode(self):
        """Actualiza a nivel de episodio"""
        for (state, action) in self.episode:
            self.n_visits[state, action] += 1.0
            alpha = 1.0 / self.n_visits[state, action]
            self.Q[state, action] += alpha * (self.result_sum - self.Q[state, action])
        # Guardamos datos sobre la evolución
        self.stats += self.result_sum
        self.episodes += len(self.episode)
        self.list_stats.append(self.stats/(self.numEpisodes+1))
        self.list_episodes.append(self.episodes/(self.numEpisodes+1))
        self.numEpisodes += 1

    def decay_epsilon(self):
        #self.epsilon = min(self.final_epsilon, self.initial_epsilon/(self.numEpisodes+1))
        self.epsilon = min(1.0, 1000.0/(self.numEpisodes+1))


    # Política epsilon-soft. Se usa para el entrenamiento
    def random_epsilon_greedy_policy(self, state):
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    # Política epsilon-greedy a partir de una epsilon-soft
    def epsilon_greedy_policy(self, state):
        pi_A = self.random_epsilon_greedy_policy(state)
        return np.random.choice(np.arange(self.nA), p=pi_A)

    # Política Greedy a partir de los valones Q. Se usa para mostrar la solución.
    def pi_star_from_Q(self, env, Q):
        done = False
        pi_star = np.zeros([env.observation_space.n, env.action_space.n])
        state, info = env.reset() # start in top-left, = 0
        actions = ""
        while not done:
            action = np.argmax(Q[state, :])
            actions += f"{action}, "
            pi_star[state,action] = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return pi_star, actions


class FrozenAgentMC_On_First:
    def __init__(self, env, epsilon: float, discount_factor: float = 0.95):
        self.descripcion=f"FrozenAgentMC_On_First. epsilon={epsilon} discount_factor={discount_factor}"
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self.env = env
        self.initAgent()

    def __str__(self):
        return self.descripcion

    def initAgent(self):
        self.epsilon = self._epsilon
        self.discount_factor = self._discount_factor
        self.nA = self.env.action_space.n
        #inicializamos Q con valores aleatorios
        self.Q = np.random.randn(self.env.observation_space.n, self.nA)
        self.returns = np.zeros((self.env.observation_space.n, self.nA))  # Almacena la suma de todas las recompensas por estado-acción
        self.nreturns = np.zeros((self.env.observation_space.n, self.nA))  # Almacena la cantidad de elementos de todas las recompensas por estado-acción
        #definimos una política soft
        self.policy = np.ones((self.env.observation_space.n, self.nA)) / self.nA
        self.updatePolicy()
        #para optimizar el cálculo de primera visita
        self.visited=set()
        self.episode = []
        self.stats = 0.0
        self.episodes = 0.0
        self.list_stats = [self.stats]
        self.list_episodes = [self.episodes]
        self.numEpisodes = 0

    def initEpisode(self):
        #para optimizar el cálculo de primera visita
        self.visited=set()
        self.episode = []

    def get_action(self, env, state: tuple[int]) -> int:
        return np.random.choice(np.arange(self.nA), p=self.policy[state])
        #return np.argmax(self.Q[state])
        #return self.epsilon_greedy_policy(state)

    def updateStep(self, state: tuple[int], action: int, reward: float, terminated: bool, next_state: tuple[int]):
        #antes de marcar este par (s,a) como visitado lo veo antes y así lo indico en la lista de episodios
        existe=(state,action) in self.visited
        self.visited.add((state,action))
        self.episode.append((state, action, reward, existe))

    def updateEpisode(self):
        G = 0  # Retorno acumulado
        for (state, action, reward, existe) in self.episode[::-1]:
            G = self.discount_factor * G + reward
            if existe==False:
                self.returns[state, action]+=G
                self.nreturns[state, action]+=1
                self.Q[state, action] = self.returns[state, action]/self.nreturns[state, action]
                #como hemos modificado Q[s,a] actualizamos la política de s
                best_action = np.argmax(self.Q[state])
                self.policy[state] = self.epsilon / self.nA
                self.policy[state, best_action] += (1 - self.epsilon)
        
        self.stats += G
        self.episodes += len(self.episode)
        self.list_stats.append(self.stats/(self.numEpisodes+1))
        self.list_episodes.append(self.episodes/(self.numEpisodes+1))
        self.numEpisodes += 1
        #self.updatePolicy()
        

    def decay_epsilon(self):
        self.epsilon = min(1.0, 1000.0/(self.numEpisodes+1))

    def updatePolicy(self):
        for state in range(self.env.observation_space.n):
            best_action = np.argmax(self.Q[state])
            self.policy[state] = self.epsilon / self.nA
            self.policy[state, best_action] += (1 - self.epsilon)

    def random_epsilon_greedy_policy(self, state):
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    def epsilon_greedy_policy(self, state):
        pi_A = self.random_epsilon_greedy_policy(state)
        return np.random.choice(np.arange(self.nA), p=pi_A)

    # Política Greedy a partir de los valones Q. Se usa para mostrar la solución.
    def pi_star_from_Q(self, env, Q):
        done = False
        pi_star = np.zeros([env.observation_space.n, env.action_space.n])
        state, info = env.reset() # start in top-left, = 0
        actions = ""
        while not done:
            action = np.argmax(Q[state, :])
            actions += f"{action}, "
            pi_star[state,action] = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return pi_star, actions

class FrozenAgentMC_On_All:
    def __init__(self, env, epsilon: float, discount_factor: float = 0.95):
        self.descripcion=f"FrozenAgentMC_On_All. epsilon={epsilon} discount_factor={discount_factor}"
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self.env = env
        self.initAgent()

    def __str__(self):
        return self.descripcion

    def initAgent(self):
        self.epsilon = self._epsilon
        self.discount_factor = self._discount_factor
        self.nA = self.env.action_space.n
        #inicializamos Q con valores aleatorios
        self.Q = np.random.randn(self.env.observation_space.n, self.nA)
        self.returns = np.zeros((self.env.observation_space.n, self.nA))  # Almacena la suma de todas las recompensas por estado-acción
        self.nreturns = np.zeros((self.env.observation_space.n, self.nA))  # Almacena la cantidad de elementos de todas las recompensas por estado-acción
        #definimos una política soft
        self.policy = np.ones((self.env.observation_space.n, self.nA)) / self.nA
        self.updatePolicy()
        #para optimizar el cálculo de primera visita
        self.visited=set()
        
        # Número de visitas.
        self.n_visits = np.zeros([self.env.observation_space.n, self.nA])
        
        self.episode = []
        self.stats = 0.0
        self.episodes = 0.0
        self.list_stats = [self.stats]
        self.list_episodes = [self.episodes]
        self.numEpisodes = 0

    def initEpisode(self):
        self.episode = []

    def get_action(self, env, state: tuple[int]) -> int:
        return np.random.choice(np.arange(self.nA), p=self.policy[state])
        #return np.argmax(self.Q[state])
        #return self.epsilon_greedy_policy(state)

    def updateStep(self, state: tuple[int], action: int, reward: float, terminated: bool, next_state: tuple[int]):
        self.episode.append((state, action, reward))

    def updateEpisode(self):
        G = 0  # Retorno acumulado
        for (state, action, reward) in self.episode[::-1]:
            G = self.discount_factor * G + reward
            self.returns[state, action]+=G
            self.nreturns[state, action]+=1
            self.Q[state, action] = self.returns[state, action]/self.nreturns[state, action]
            #como hemos modificado Q[s,a] actualizamos la política de s
            best_action = np.argmax(self.Q[state])
            self.policy[state] = self.epsilon / self.nA
            self.policy[state, best_action] += (1.0 - self.epsilon)
        
        self.stats += G
        self.episodes += len(self.episode)
        self.list_stats.append(self.stats/(self.numEpisodes+1))
        self.list_episodes.append(self.episodes/(self.numEpisodes+1))
        self.numEpisodes += 1

    def decay_epsilon(self):
        self.epsilon = min(1.0, 1000.0/(self.numEpisodes+1))

    def updatePolicy(self):
        for state in range(self.env.observation_space.n):
            best_action = np.argmax(self.Q[state])
            self.policy[state] = self.epsilon / self.nA
            self.policy[state, best_action] += (1 - self.epsilon)

    def random_epsilon_greedy_policy(self, state):
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    def epsilon_greedy_policy(self, state):
        pi_A = self.random_epsilon_greedy_policy(state)
        return np.random.choice(np.arange(self.nA), p=pi_A)

    # Política Greedy a partir de los valones Q. Se usa para mostrar la solución.
    def pi_star_from_Q(self, env, Q):
        done = False
        pi_star = np.zeros([env.observation_space.n, env.action_space.n])
        state, info = env.reset() # start in top-left, = 0
        actions = ""
        while not done:
            action = np.argmax(Q[state, :])
            actions += f"{action}, "
            pi_star[state,action] = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return pi_star, actions

class FrozenAgentMC_Off_Pi:
    def __init__(self, env, epsilon: float, discount_factor: float = 0.95):
        self.descripcion=f"FrozenAgentMC_Off_Pi. epsilon={epsilon} discount_factor={discount_factor}"
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self.env = env
        self.initAgent()

    def __str__(self):
        return self.descripcion

    def initAgent(self):
        self.epsilon = self._epsilon
        self.discount_factor = self._discount_factor
        self.nA = self.env.action_space.n
        #inicializamos Q con valores aleatorios
        self.Q = np.random.randn(self.env.observation_space.n, self.nA)
        self.C = np.zeros((self.env.observation_space.n, self.nA))     #acumulado del muestreo de importancia
        #definimos la política en base a argmax_a(Q(S,a))
        self.policy = np.ones(self.env.observation_space.n)
        for state in range(self.env.observation_space.n):
            self.policy[state] = np.argmax(self.Q[state])
        self.bpolicy = np.ones((self.env.observation_space.n, self.nA)) / self.nA   #política b
        
        self.episode = []
        self.stats = 0.0
        self.episodes = 0.0
        self.list_stats = [self.stats]
        self.list_episodes = [self.episodes]
        self.numEpisodes = 0

    def initEpisode(self):
        self.episode = []
        #redefino una política b a partir de pi 
        for state in range(self.env.observation_space.n):
            best_action = np.argmax(self.Q[state])
            self.bpolicy[state] = self.epsilon / self.nA
            self.bpolicy[state, best_action] += (1 - self.epsilon)
        

    def get_action(self, env, state: tuple[int]) -> int:
        return np.random.choice(np.arange(self.nA), p=self.bpolicy[state])      #tomo la acción en base a la bpolitica
        #return np.argmax(self.Q[state])
        #return self.epsilon_greedy_policy(state)

    def updateStep(self, state: tuple[int], action: int, reward: float, terminated: bool, next_state: tuple[int]):
        self.episode.append((state, action, reward))

    def updateEpisode(self):
        G = 0  # Retorno acumulado
        W = 1.0 #proporción 
        for (state, action, reward) in self.episode[::-1]:
            G = self.discount_factor * G + reward
            self.C[state, action] = self.C[state, action] + W
            self.Q[state, action] = self.Q[state, action] + (W/self.C[state, action]) * (G - self.Q[state, action])
            
            best_action = np.argmax(self.Q[state])
            self.policy[state] = best_action
            
            if action!=best_action:
                break
            
            W = W / self.bpolicy[state,action]
        
        self.stats += G
        self.episodes += len(self.episode)
        self.list_stats.append(self.stats/(self.numEpisodes+1))
        self.list_episodes.append(self.episodes/(self.numEpisodes+1))
        self.numEpisodes += 1

    def decay_epsilon(self):
        self.epsilon = min(1.0, 1000.0/(self.numEpisodes+1))

    def random_epsilon_greedy_policy(self, state):
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    def epsilon_greedy_policy(self, state):
        pi_A = self.random_epsilon_greedy_policy(state)
        return np.random.choice(np.arange(self.nA), p=pi_A)

    # Política Greedy a partir de los valones Q. Se usa para mostrar la solución.
    def pi_star_from_Q(self, env, Q):
        done = False
        pi_star = np.zeros([env.observation_space.n, env.action_space.n])
        state, info = env.reset() # start in top-left, = 0
        actions = ""
        while not done:
            action = np.argmax(Q[state, :])
            actions += f"{action}, "
            pi_star[state,action] = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return pi_star, actions

class FrozenAgentSARSA:
    def __init__(self, env, epsilon: float, alpha: float, discount_factor: float = 0.95):
        self.descripcion=f"FrozenAgentSARSA. epsilon={epsilon} alpha={alpha} discount_factor={discount_factor}"
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self._alpha=alpha
        self.env = env
        self.initAgent()

    def __str__(self):
        return self.descripcion

    def initAgent(self):
        self.alpha = self._alpha
        self.epsilon = self._epsilon
        self.discount_factor = self._discount_factor
        self.nA = self.env.action_space.n
        #inicializamos Q con valores aleatorios
        self.Q = np.ones((self.env.observation_space.n, self.nA))
        #self.Q = np.random.randn(self.env.observation_space.n, self.nA)
        #ponemos a 0 la Q de los estados terminales
        mapa_lineal = self.env.unwrapped.desc.flatten().astype(str).tolist()
        for index, valor in enumerate(mapa_lineal):
            if valor=='G':
                self.Q[index]=np.zeros(self.nA)
        
        self.episode = []
        self.stats = 0.0
        self.episodes = 0.0
        self.list_stats = [self.stats]
        self.list_episodes = [self.episodes]
        self.numEpisodes = 0

    def initEpisode(self):
        self.G=0
        self.length=0

    def get_action(self, env, state: tuple[int]) -> int:
        best_action = np.argmax(self.Q[state])
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        pi_A[best_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(self.nA), p=pi_A)

    def updateStep(self, state: tuple[int], action: int, reward: float, terminated: bool, next_state: tuple[int]):
        self.G = self.discount_factor * self.G + reward
        self.length+=1
        if not terminated:
            next_action = self.get_action(self.env,next_state)
            self.Q[state, action] = self.Q[state, action] + self.alpha * ( reward + self.discount_factor * self.Q[next_state,next_action] - self.Q[state, action])
        else:
            self.Q[state, action] = self.Q[state, action] + self.alpha * ( reward - self.Q[state, action])

    def updateEpisode(self):
        self.stats += self.G
        self.episodes += self.length
        self.list_stats.append(self.stats/(self.numEpisodes+1))
        self.list_episodes.append(self.episodes/(self.numEpisodes+1))
        self.numEpisodes += 1

    def decay_epsilon(self):
        self.epsilon = min(1.0, 1000.0/(self.numEpisodes+1))

    # Política Greedy a partir de los valones Q. Se usa para mostrar la solución.
    def pi_star_from_Q(self, env, Q):
        done = False
        pi_star = np.zeros([env.observation_space.n, env.action_space.n])
        state, info = env.reset() # start in top-left, = 0
        actions = ""
        while not done:
            action = np.argmax(Q[state, :])
            actions += f"{action}, "
            pi_star[state,action] = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return pi_star, actions

class FrozenAgentQ_Learning:
    def __init__(self, env, epsilon: float, alpha: float, discount_factor: float = 0.95):
        self.descripcion=f"FrozenAgentQ_Learning. epsilon={epsilon} alpha={alpha} discount_factor={discount_factor}"
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self._alpha=alpha
        self.env = env
        self.initAgent()

    def __str__(self):
        return self.descripcion

    def initAgent(self):
        self.alpha = self._alpha
        self.epsilon = self._epsilon
        self.discount_factor = self._discount_factor
        self.nA = self.env.action_space.n
        #inicializamos Q con valores aleatorios
        #self.Q = np.zeros((self.env.observation_space.n, self.nA))
        self.Q = np.ones((self.env.observation_space.n, self.nA))
        #ponemos a 0 la Q de los estados terminales
        mapa_lineal = self.env.unwrapped.desc.flatten().astype(str).tolist()
        for index, valor in enumerate(mapa_lineal):
            if valor=='G':
                self.Q[index]=np.zeros(self.nA)
        
        self.episode = []
        self.stats = 0.0
        self.episodes = 0.0
        self.list_stats = [self.stats]
        self.list_episodes = [self.episodes]
        self.numEpisodes = 0

    def initEpisode(self):
        self.G=0
        self.length=0

    def get_action(self, env, state: tuple[int]) -> int:
        best_action = np.argmax(self.Q[state])
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        pi_A[best_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(self.nA), p=pi_A)
        '''
        if np.random.randn()<self.epsilon:
            return np.random.choice(np.arange(self.nA))
        else:
            return np.argmax(self.Q[state])
        '''
        
    def updateStep(self, state: tuple[int], action: int, reward: float, terminated: bool, next_state: tuple[int]):
        self.G = self.discount_factor * self.G + reward
        self.length+=1
        if not terminated:
            next_action = self.get_action(self.env,next_state)
            maxQ = np.max(self.Q[next_state])
            self.Q[state, action] = self.Q[state, action] + self.alpha * ( reward + self.discount_factor * maxQ - self.Q[state, action])
        else:
            self.Q[state, action] = self.Q[state, action] + self.alpha * ( reward - self.Q[state, action])

    def updateEpisode(self):
        self.stats += self.G
        self.episodes += self.length
        self.list_stats.append(self.stats/(self.numEpisodes+1))
        self.list_episodes.append(self.episodes/(self.numEpisodes+1))
        self.numEpisodes += 1

    def decay_epsilon(self):
        self.epsilon = min(1.0, 1000.0/(self.numEpisodes+1))

    # Política Greedy a partir de los valones Q. Se usa para mostrar la solución.
    def pi_star_from_Q(self, env, Q):
        done = False
        pi_star = np.zeros([env.observation_space.n, env.action_space.n])
        state, info = env.reset() # start in top-left, = 0
        actions = ""
        while not done:
            action = np.argmax(Q[state, :])
            actions += f"{action}, "
            pi_star[state,action] = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return pi_star, actions

