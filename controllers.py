import numpy as np
import json

class MLPController:
    def __init__(self, model_path=None, num_agents=1, agent_idx=0):
        self.in_size = 11 if num_agents > 1 else 10
        self.out_size = 5 if num_agents > 1 else 4
        self.h_size = 16
        self.agent_idx = agent_idx
        self.w1 = np.zeros((self.in_size, self.h_size))
        self.b1 = np.zeros(self.h_size)
        self.w2 = np.zeros((self.h_size, self.out_size))
        self.b2 = np.zeros(self.out_size)
        if model_path: self.load(model_path)

    def act(self, obs, info=None):
        x = np.atleast_2d(obs)
        h = np.maximum(0, x @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        return int(np.argmax(out))

    def reset(self): pass

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            if 'brain_a' in data:
                data = data['brain_a'] if self.agent_idx == 0 else data['brain_b']
            self.w1 = np.array(data['w1'])
            self.b1 = np.array(data['b1'])
            self.w2 = np.array(data['w2'])
            self.b2 = np.array(data['b2'])

class GRUController:
    def __init__(self, model_path=None, num_agents=1, agent_idx=0, use_social_filter=True):
        self.in_size = 11 if num_agents > 1 else 10
        self.out_size = 5 if num_agents > 1 else 4
        self.h_size = 16
        self.agent_idx = agent_idx
        self.use_bg = True
        self.use_social_filter = use_social_filter
        self.w_gru = np.zeros((self.in_size + self.h_size, 3 * self.h_size))
        self.b_gru = np.zeros(3 * self.h_size)
        self.w_out = np.zeros((self.h_size, self.out_size))
        self.b_out = np.zeros(self.out_size)
        # MATRIZ RESIDUAL: Conexión directa (Res-GRU)
        self.w_res = np.zeros((self.in_size, self.out_size))
        self.h = np.zeros((1, self.h_size))
        if model_path: self.load(model_path)

    def act(self, obs, info=None):
        x = np.atleast_2d(obs)
        concat = np.column_stack([x, self.h])
        gates = concat @ self.w_gru + self.b_gru
    
        z = 1.0 / (1.0 + np.exp(-gates[:, :self.h_size]))
        r = 1.0 / (1.0 + np.exp(-gates[:, self.h_size:2*self.h_size]))
        h_tilde = np.tanh(np.column_stack([x, r * self.h]) @ self.w_gru[:, 2*self.h_size:] + self.b_gru[2*self.h_size:])
        self.h = (1 - z) * self.h + z * h_tilde
    
        # 1. Cálculo de salida limpia (GRU + Residual)
        out_gru = self.h @ self.w_out + self.b_out
        near_wall = np.max(x[0, :8]) > 0.82
        multi = 0.3 if near_wall else 5.0
        out_total = out_gru + (x @ self.w_res) * multi
    
        # --- LÓGICA V44: EL FILTRO DE ATENCIÓN SOCIAL ---
        # Solo aplicamos una regla: Si eres el Seguidor (1), hay un muro enfrente
        # y escuchas al Faro (x[0,10]), entonces DEJA DE ACELERAR contra el muro.
        #if self.use_social_filter and self.agent_idx == 1: COMM
        if self.use_social_filter:
            social_active = abs(x[0, 10]) > 0.05
            if near_wall and social_active:
                # Quitamos el avance (Acción 0) para que el agente "se detenga a escuchar"
                out_total[0, 0] -= 30.0
                # NO FORZAMOS GIRO. Dejamos que la red GRU use el sensor social obs[10]
                # para decidir el giro por sí sola, sin vibraciones externas.

        # --- GANGLIO BASAL: EL FARO (Acción 4) ---
        # Umbral 0.32 para que nunca se apague en la meta
        if self.use_bg and self.out_size > 4 and x[0, 9] > 0.32: 
            out_total[0, 4] += 20.0 
        
        return int(np.argmax(out_total))

    def reset(self):
        self.h = np.zeros((1, self.h_size))

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            if 'brain_a' in data:
                data = data['brain_a'] if self.agent_idx == 0 else data['brain_b']
            
            # Carga de parámetros con verificación de existencia
            if 'w_gru' in data: self.w_gru = np.array(data['w_gru'])
            if 'b_gru' in data: self.b_gru = np.array(data['b_gru'])
            if 'w_out' in data: self.w_out = np.array(data['w_out'])
            if 'b_out' in data: self.b_out = np.array(data['b_out'])
            if 'w_res' in data: self.w_res = np.array(data['w_res'])

class ResController:
    def __init__(self, model_path=None, num_agents=1, agent_idx=0):
        self.in_size = 11 if num_agents > 1 else 10
        self.out_size = 5 if num_agents > 1 else 4
        self.h_size = 16
        self.agent_idx = agent_idx
        self.w1 = np.zeros((self.in_size, self.h_size))
        self.b1 = np.zeros(self.h_size)
        self.w2 = np.zeros((self.h_size, self.out_size))
        self.b2 = np.zeros(self.out_size)
        self.w_skip = np.eye(self.in_size, self.out_size)
        if model_path: self.load(model_path)

    def act(self, obs, info=None):
        x = np.atleast_2d(obs)
        h = np.maximum(0, x @ self.w1 + self.b1)
        res = h @ self.w2 + self.b2
        skip = x @ self.w_skip
        out = res + skip
        return int(np.argmax(out))

    def reset(self): pass

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            if 'brain_a' in data:
                data = data['brain_a'] if self.agent_idx == 0 else data['brain_b']
            self.w1 = np.array(data['w1'])
            self.b1 = np.array(data['b1'])
            self.w2 = np.array(data['w2'])
            self.b2 = np.array(data['b2'])
            if 'w_skip' in data:
                self.w_skip = np.array(data['w_skip'])

class RandomController:
    def __init__(self, num_agents=1): self.out_size = 5 if num_agents > 1 else 4
    def act(self, obs, info=None): return np.random.randint(0, self.out_size)
    def reset(self): pass

class BraitenbergController:
    def __init__(self, num_agents=1): self.num_agents = num_agents
    def act(self, obs, info=None):
        left_sensors, right_sensors = obs[0:3], obs[5:8]
        if np.max(left_sensors) > 0.5: return 2
        if np.max(right_sensors) > 0.5: return 1
        return 0
    def reset(self): pass

class HeuristicCalibrationController:
    """
    Heurístico para la U con reacople activo.

    B:
    - sigue un lado preferido (izq/der)
    - si choca: backoff + turn + advance
    - si pierde la pared: reacquire = turn + advance
    """

    def __init__(self, agent_idx):
        self.agent_idx = agent_idx
        self.phase = "normal"
        self.timer = 0
        self.turn_dir = 0
        self.free_counter = 0
        self.subphase = None

    def reset(self):
        self.phase = "normal"
        self.timer = 0
        self.turn_dir = 0
        self.free_counter = 0
        self.subphase = None

    def act(self, obs):
        prox = obs[0:8]
        odor = obs[9]
        soc = obs[10] if len(obs) > 10 else 0.0

        front = max(prox[0], prox[1], prox[7])
        left = max(prox[1], prox[2], prox[3])
        right = max(prox[5], prox[6], prox[7])

        if self.agent_idx == 1:
            print(
                f"B -> phase={self.phase:>12} sub={str(self.subphase):>7} "
                f"timer={self.timer} free={self.free_counter} "
                f"front={front:.2f} L={left:.2f} R={right:.2f} "
                f"soc={soc:.2f} odor={odor:.2f}"
            )

        # Líder
        if self.agent_idx == 0 and odor > 0.30:
            return 4

        # lado preferido
        preferred = +1 if soc > 0 else -1

        # --------------------------------------------------
        # BACKOFF
        # --------------------------------------------------
        if self.phase == "backoff":
            self.timer -= 1
            if self.timer <= 0:
                self.phase = "turn"
                self.timer = 3
            return 3

        # --------------------------------------------------
        # TURN
        # --------------------------------------------------
        if self.phase == "turn":
            self.timer -= 1
            if self.timer <= 0:
                self.phase = "advance"
                self.timer = 8
            return 2 if self.turn_dir > 0 else 1

        # --------------------------------------------------
        # ADVANCE
        # --------------------------------------------------
        if self.phase == "advance":
            self.timer -= 1

            if front > 0.50:
                self.phase = "backoff"
                self.timer = 4
                return 3

            if self.timer <= 0:
                self.phase = "normal"
                self.free_counter = 0

            if left > 0.82:
                return 1
            if right > 0.82:
                return 2

            return 0

        # --------------------------------------------------
        # REACQUIRE_WALL = turn a bit + advance a bit
        # --------------------------------------------------
        if self.phase == "reacquire_wall":
            if self.subphase == "turn":
                self.timer -= 1
                if self.timer <= 0:
                    self.subphase = "advance"
                    self.timer = 5

                return 2 if preferred > 0 else 1

            if self.subphase == "advance":
                self.timer -= 1

                # si reacopló la pared, salir
                if preferred > 0 and left > 0.15:
                    self.phase = "normal"
                    self.subphase = None
                    self.free_counter = 0
                    return 0

                if preferred < 0 and right > 0.15:
                    self.phase = "normal"
                    self.subphase = None
                    self.free_counter = 0
                    return 0

                # si aparece pared frontal, maniobra completa
                if front > 0.50:
                    self.phase = "backoff"
                    self.subphase = None
                    self.timer = 4
                    self.turn_dir = preferred
                    return 3

                if self.timer <= 0:
                    self.phase = "normal"
                    self.subphase = None
                    self.free_counter = 0
                    return 0

                return 0

        # --------------------------------------------------
        # NORMAL
        # --------------------------------------------------
        if front > 0.50:
            self.phase = "backoff"
            self.timer = 4
            self.turn_dir = preferred
            return 3

        if left > 0.85:
            return 1
        if right > 0.85:
            return 2

        # Si perdió la pared del lado preferido, intentar reacoplarla
        if preferred > 0:
            if left < 0.04 and front < 0.25:
                self.phase = "reacquire_wall"
                self.subphase = "turn"
                self.timer = 2
                return 2
        else:
            if right < 0.04 and front < 0.25:
                self.phase = "reacquire_wall"
                self.subphase = "turn"
                self.timer = 2
                return 1

        # En libre: avanzar con sesgo suave, no órbitas grandes
        self.free_counter += 1

        if preferred > 0:
            if self.free_counter % 8 == 0:
                return 2
            return 0
        else:
            if self.free_counter % 8 == 0:
                return 1
            return 0
            
def make_controller(type, num_agents=1, agent_idx=0, model_path=None, use_bg=True, use_social_filter=True):
    t = type.lower()
    if t == "mlp": return MLPController(model_path, num_agents, agent_idx)
    if t == "gru":
        ctrl = GRUController(
            model_path=model_path,
            num_agents=num_agents,
            agent_idx=agent_idx,
            use_social_filter=use_social_filter,
        )
        ctrl.use_bg = use_bg
        return ctrl
    if t == "res" or t == "residual": return ResController(model_path, num_agents, agent_idx)
    if t == "random": return RandomController(num_agents)
    if t == "braitenberg": return BraitenbergController(num_agents)
    if t == "heuristic": return HeuristicCalibrationController(agent_idx)
    raise ValueError(f"Tipo de controlador desconocido: {type}")
    