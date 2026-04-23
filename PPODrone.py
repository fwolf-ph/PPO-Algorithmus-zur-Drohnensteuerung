import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from drone_env.drone_2d_env import Drone2dEnv

env = Drone2dEnv(render_sim=False)


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

import numpy as np
import tensorflow as tf
from tensorflow import keras


#Batch: anzahl der schritte
#Mini-batch: Größe der Häppchen
#epoch: wie häufig wird über einen Datensatz trainiert
#Iterationen: wie häufig wird eine Trajektoti gesammelt und und dann einmal ausgeführt
class ActorCritic:
    def __init__(self, na=64, nc=128):
        # Initialisierer (Standardabweichung hilft bei der Stabilität)
        init = tf.keras.initializers.RandomNormal(stddev=0.01)

        # --- Actor-Netzwerk (Wählt die Aktion) ---
        # Input: 8 -> Hidden: na -> Output: 4
        self.a_dense1 = tf.keras.layers.Dense(na, activation='relu', kernel_initializer=init)
        self.a_dense_output = tf.keras.layers.Dense(4, activation=None, kernel_initializer=init)

        # --- Critic-Netzwerk (Bewertet den Zustand) ---
        # Input: 8 -> Hidden: nc -> Output: 1
        self.c_dense1 = tf.keras.layers.Dense(nc, activation='relu', kernel_initializer=init)
        self.c_dense_output = tf.keras.layers.Dense(1, activation=None, kernel_initializer=init)


    def policy_function(self, state):
      x = self.a_dense1(state)
      out = self.a_dense_output(x)  # (batch,4)
      mu, log_std = tf.split(out, num_or_size_splits=2, axis=-1)  # (batch,2), (batch,2)
      log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
      return mu, log_std


    def value_function(self, state):
        # Hier nutzen wir NUR die Critic-Layer (c_...)
        x1 = self.c_dense1(state)
        v_value = self.c_dense_output(x1)
        return v_value # Gibt eine einzelne Zahl (Zustandswert) zurück

def sample_action(mu, log_std):
  '''
  mu: Mittelwert
  log_std: Standardabweichung

  Shapes:
  mu:      Tensor shape (batch, 2) oder (1, 2)
  log_std: Tensor shape (batch, 2) oder (1, 2)
  '''
    """
    returns:
      action: Tensor shape (batch, 2) in [-1,1]
    """
    std = tf.exp(log_std) #da log_std stabiler zum Lernen ist, wird log_std gespeichert. 
                          #Anwenden von exp gibt std zurück
    #Schritt über eps um Gradientenberechung beim Backprob zu stabiliseren
    eps = tf.random.normal(shape=tf.shape(mu)) #tf.random.normal erzeugt Zufallszahllen im Shape wie mu, also [batch, 2]
           '''
           tf.random.normal(
                shape,
                mean=0.0, <- Mittelwert = 0
                stddev=1.0, <- Standardabweichung = 1
                dtype=tf.dtypes.float32,
                seed=None,
                name=None
            )
           '''                          
    pre_tanh = mu + std * eps #Reskalsieren mu erzeugt offset, std*eps streut
    action = tf.tanh(pre_tanh) # Aktivierungfunktion tanh quetscht Werte in [-1,1]
    return action
  
def logp_from_mu_logstd_action(mu, log_std, action):
    """
    mu, log_std: (1,2) oder (batch,2)
    action:      (1,2) oder (batch,2) in [-1,1]
    returns: logp (batch,)
    """
    std = tf.exp(log_std)
    a = tf.clip_by_value(action, -1.0 + 1e-6, 1.0 - 1e-6)
    pre_tanh = 0.5 * tf.math.log((1.0 + a) / (1.0 - a))

    # logprob
    logp_gauss = -0.5 * tf.reduce_sum(
        ((pre_tanh - mu) / (std + 1e-8))**2 + 2.0 * log_std + np.log(2*np.pi),
        axis=-1
    )

    logp = logp_gauss
    return logp



def calculate_advantages(returns, values):
    # Sicherstellen, dass beide Tensors die gleiche Form haben (N,)
    returns = tf.reshape(returns, [-1])
    values = tf.reshape(values, [-1])
    
    advantages = returns - values
    
    # Normalisierung
    mean = tf.reduce_mean(advantages)
    std = tf.math.reduce_std(advantages)
    advantages = (advantages - mean) / (std + 1e-8)
    return advantages
  

def ppo_policy_loss(old_log_prob, new_log_prob, advantages, eps=0.2):
    advantages = tf.stop_gradient(advantages)
    log_ratio = new_log_prob - old_log_prob
    log_ratio = tf.clip_by_value(log_ratio, -20.0, 20.0)  # <- wichtig,  clipt den logarithmus des verhältnisses
    policy_ratio = tf.exp(log_ratio) # verwandelt den logarithmus wieder in eine zahl(ist so semi gecappt bei 485 +10^6 und 2,06 + 10^-9)
    
    surrogate_1 = policy_ratio * advantages # 
    surrogate_2 = tf.clip_by_value(policy_ratio, 1.0 - eps, 1.0  + eps) * advantages # dieses clip verhalten kennt man, so wie in dem video
    surrogate = -tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2)) # durchschnitt summe, und minus für gradient decent = gradient acent
    return surrogate


def value_loss(reward_to_go, value):
  '''
  s = 0
  for i in range(batch_size):
    s += (reward_to_go - value)**2
  s = s / batch_size'''
  return tf.reduce_mean(tf.square(reward_to_go - value))



class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma):
        self.obs_buf   = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf   = np.zeros((size, act_dim), dtype=np.float32)
        self.logp_buf  = np.zeros((size,),         dtype=np.float32)
        self.rew_buf   = np.zeros((size,),         dtype=np.float32)
        self.val_buf   = np.zeros((size,),         dtype=np.float32)
        self.done_buf  = np.zeros((size,),         dtype=np.float32)  # 1.0 wenn terminal, sonst 0.0

        self.rtg_buf   = np.zeros((size,),         dtype=np.float32)  # returns-to-go

        self.gamma = gamma
        self.ptr = 0
        self.max_size = size

    def store(self, obs, act, logp, rew, val, done):
        assert self.ptr < self.max_size, "Buffer voll"
        self.obs_buf[self.ptr]  = obs
        self.act_buf[self.ptr]  = act
        self.logp_buf[self.ptr] = logp
        self.rew_buf[self.ptr]  = rew
        self.val_buf[self.ptr]  = val
        self.done_buf[self.ptr] = float(done)
        self.ptr += 1

    def compute_rtg(self, last_val=0.0):
        #Bootstraping
        rtg = last_val
        for t in reversed(range(self.ptr)):
            if self.done_buf[t] == 1.0:
                rtg = 0.0
            rtg = self.rew_buf[t] + self.gamma * rtg
            self.rtg_buf[t] = rtg

    def get(self):
        data = dict(
            obs=self.obs_buf[:self.ptr],
            act=self.act_buf[:self.ptr],
            logp=self.logp_buf[:self.ptr],
            rew=self.rew_buf[:self.ptr],
            val=self.val_buf[:self.ptr],
            done=self.done_buf[:self.ptr],
            rtg=self.rtg_buf[:self.ptr],
        )
        return data

    def clear(self):
        self.ptr = 0

def get_trajectories(env, ac, max_size, gamma):
    obs_dim = env.observation_space.shape[0]   # 8
    act_dim = env.action_space.shape[0]        # 2

    buf = RolloutBuffer(obs_dim=obs_dim, act_dim=act_dim, size=max_size, gamma=gamma)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    done = False

    for t in range(max_size):
        # state -> Tensor (1,8)
        state = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)

        # Actor: mu, log_std
        mu, log_std = ac.policy_function(state)         # (1,2), (1,2)

        # Sample action (1,2) und logprob berechnen
        action_t = sample_action(mu, log_std)           # (1,2) in [-1,1]
        logp_t = logp_from_mu_logstd_action(mu, log_std, action_t)  # (1,)

        # Critic value
        v_t = ac.value_function(state)[:, 0]            # (1,)

        action = action_t.numpy()[0].astype(np.float32)
        logp = float(logp_t.numpy()[0])
        val = float(v_t.numpy()[0])

      
        step_out = env.step(action)
        if len(step_out) == 4:
            next_obs, rew, done, info = step_out
        else:
            next_obs, rew, terminated, truncated, info = step_out
            done = bool(terminated or truncated)

        if isinstance(next_obs, tuple): #reduzieren auf next_obs[0]
            next_obs = next_obs[0]

        #Clipping
        if (not np.all(np.isfinite(next_obs))) or (not np.isfinite(rew)) or (not np.all(np.isfinite(action))):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            continue

        # speichern
        buf.store(obs, action, logp, rew, val, done)

        
        obs = next_obs

        # Reset
        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False

    # RTG berechnen
    buf.compute_rtg(last_val=0.0)

    data = buf.get()
    return data

def actor_vars(ac): #sammelt trainierbare Parameter, dadurch wird verhindert das der Actor-Optimizer den Critic updated
    return ac.a_dense1.trainable_variables + ac.a_dense_output.trainable_variables

def critic_vars(ac):
    return ac.c_dense1.trainable_variables + ac.c_dense_output.trainable_variables

#ppo_update updated die Parameter
def ppo_update(ac, dataset, actor_opt, critic_opt,
               clip_eps=0.2, vf_coef=0.5,
               epochs=4 '''wie häufig wird über einen Datensatz trainiert''', batch_size=128):

    # Daten aus einer Trajektorie werden abgerufen
    obs = tf.convert_to_tensor(dataset["obs"], dtype=tf.float32)       # (N,8)
    act = tf.convert_to_tensor(dataset["act"], dtype=tf.float32)       # (N,2)
    logp_old = tf.convert_to_tensor(dataset["logp"], dtype=tf.float32) # (N,)
    rtg = tf.convert_to_tensor(dataset["rtg"], dtype=tf.float32)       # (N,)
    val_old = tf.convert_to_tensor(dataset["val"], dtype=tf.float32)   # (N,)

    # Advantages
    adv = calculate_advantages(rtg, val_old) #Shape: Advatages für jeden Schritt werden bereechnet aus Schätzung: Formel echte RTG - gescha4tzter RTG
    N = obs.shape[0]
    if N == 0:
        return 0.0, 0.0
    

    pi_vars = actor_vars(ac)
    v_vars  = critic_vars(ac)

    last_pi_loss = 0.0
    last_critic_loss = 0.0

    for _ in range(epochs):
        idx = tf.random.shuffle(tf.range(N)) #Mini-Batches werden anders gemischt, stabiler

        for start in range(0, N, batch_size): 
            mb = idx[start:start+batch_size] 
            #beno4tigen Daten werden gezogen:
            obs_b = tf.gather(obs, mb)
            act_b = tf.gather(act, mb)
            logp_old_b = tf.gather(logp_old, mb)
            rtg_b = tf.gather(rtg, mb)
            adv_b = tf.gather(adv, mb)

            # Actor update
            with tf.GradientTape() as tape_pi: #ableitungen werden berechnet
                mu, log_std = ac.policy_function(obs_b)
                logp_new_b = logp_from_mu_logstd_action(mu, log_std, act_b)
                pi_loss = ppo_policy_loss(logp_old_b, logp_new_b, adv_b, eps=clip_eps) '''r_t = pi_neu(a_s|a_T)/pi_alt(a_s|a_T), hier ist die zu minimierende Funktion;clip begrentz größe der Updates'''


            pi_grads = tape_pi.gradient(pi_loss, pi_vars) #Gradienten berechnen
            pi_grads, _ = tf.clip_by_global_norm(pi_grads, 0.5) #Gradienten clipen
            actor_opt.apply_gradients([(g, v) for g, v in zip(pi_grads, pi_vars) if g is not None]) #jetzt werden parameter verändert, Adam: quasi adaptive learning rate

            # Critic update
            with tf.GradientTape() as tape_v:
                v_pred = ac.value_function(obs_b)[:, 0]
                v_loss = value_loss(rtg_b, v_pred)
                critic_loss = vf_coef * v_loss

            v_grads = tape_v.gradient(critic_loss, v_vars)
            v_grads, _ = tf.clip_by_global_norm(v_grads, 0.5)
            critic_opt.apply_gradients([(g, v) for g, v in zip(v_grads, v_vars) if g is not None])

            last_pi_loss = float(pi_loss.numpy())
            last_critic_loss = float(critic_loss.numpy())

    return last_pi_loss, last_critic_loss

#
import os
import imageio.v2 as imageio
import gymnasium as gym
from gym.envs.registration import register

#CHAT-GPT Video-Ausgabe
def deterministic_action_from_ac(ac, obs):
    """mean-action (ohne Sampling) -> tanh -> [-1,1]"""
    state = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
    mu, _ = ac.policy_function(state)
    return tf.tanh(mu)[0].numpy().astype(np.float32)


def record_video_snapshot(ac, iteration, out_dir="./video", max_frames=500, fps=30):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"iter_{iteration:04d}.mp4")

    # direkt Klasse nutzen (wie im Training) -> KEINE Registrierung nötig
    env_rec = Drone2dEnv(
        render_sim=True, render_path=True, render_shade=True,
        shade_distance=70, n_steps=500, n_fall_steps=10,
        change_target=True, initial_throw=True,
        render_mode="rgb_array",
    )

    obs = env_rec.reset()
    if isinstance(obs, tuple):  # falls (obs, info)
        obs = obs[0]

    frames = []
    for _ in range(max_frames):
        frame = env_rec.render()
        if frame is not None:
            frames.append(frame)

        action = deterministic_action_from_ac(ac, obs)

        step_out = env_rec.step(action)
        if len(step_out) == 4:
            obs, reward, done, info = step_out
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = step_out

        if isinstance(obs, tuple):
            obs = obs[0]

        if terminated or truncated:
            break

    env_rec.close()

    if len(frames) == 0:
        print("Keine Frames erhalten (render_mode/render() prüfen).")
        return

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"🎥 Saved video: {out_path}")
#CHAT-GPT ENDE



#Ablauf ab hier
ac = ActorCritic()
actor_opt = tf.keras.optimizers.Adam(3e-4) 
critic_opt = tf.keras.optimizers.Adam(9e-4)

history = {"avg_rew": [], "pi_loss": [], "critic_loss": []}

NUM_ITERS = 2000
batch_size = 128

avg_rew_10 = []         # sammelt die letzten 10 avg_rew
avg_rew_10_list = []    # speichert den 10er-Mittelwert (moving by blocks)

for it in range(1, NUM_ITERS + 1):
    dataset = get_trajectories(env, ac, max_size=2048, gamma=0.99)
    N = dataset["obs"].shape[0]

    if N < batch_size:
        print(f"Only {N} valid samples -> skipping iter {it}")
        continue

    pi_l, c_l = ppo_update(
        ac, dataset, actor_opt, critic_opt,
        clip_eps=0.2, vf_coef=0.5,
        epochs=10, batch_size=128
    )

    avg_rew = float(np.mean(dataset["rew"]))   # avg reward pro step im rollout

    if it % 100 == 0:
        try:
            record_video_snapshot(ac, it)
  
