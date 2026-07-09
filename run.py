# -*- coding: utf-8 -*-
import argparse
import time
import csv
import numpy as np
import imageio
import random
import pygame
from foraging_env import ForagingEnv, ForagingEnvConfig
from controllers import make_controller
from pathlib import Path

class DummyController:
    def reset(self):
        pass

WF_ESCAPE_STEPS = 0
WF_ESCAPE_TURN = 2

def get_version():
    try:
        return Path(__file__).resolve().with_name("VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        return "0.0.0"

def wall_follow_policy(obs):
    global WF_ESCAPE_STEPS, WF_ESCAPE_TURN

    num_sensors = len(obs) - 2
    prox = np.array(obs[:num_sensors], dtype=np.float32)
    # print("prox:", np.round(prox, 2)) # Comentado para no ensuciar

    # Mapeo confirmado (sentido horario):
    # 0 = frente
    # 1 = frente-izquierda
    # 2 = izquierda
    # 3 = atrás-izquierda
    # 4 = atrás
    # 5 = atrás-derecha
    # 6 = derecha
    # 7 = frente-derecha

    front       = float(prox[0])
    front_left  = float(prox[1])
    left        = float(prox[2])
    back_left   = float(prox[3])
    right       = float(prox[6])
    front_right = float(prox[7])

    # --- modo escape ---
    if WF_ESCAPE_STEPS > 0:
        WF_ESCAPE_STEPS -= 1
        if WF_ESCAPE_STEPS >= 1:   # menos pasos de retroceso
            return 3  # backward
        return WF_ESCAPE_TURN

    # Umbrales ajustados
    hard_block   = 0.70
    front_block  = 0.40
    wall_present = 0.18
    too_close    = 0.65
    corner_block = 0.65

    # 1) Choque fuerte en frente o frente-izquierda → escape
    if front > hard_block or front_left > hard_block:
        WF_ESCAPE_STEPS = 2   # escape más corto
        WF_ESCAPE_TURN = 1
        return 3

    # 2) Choque fuerte en izquierda → escape
    if left > hard_block or back_left > corner_block:
        WF_ESCAPE_STEPS = 2
        WF_ESCAPE_TURN = 1
        return 3

    # 3) Bloqueo frontal moderado → girar derecha
    if front > front_block or front_left > front_block:
        return 1

    # 4) Si pared izquierda presente:
    if left > wall_present:
        if left > too_close:
            if left < 0.75:   # corrección más gradual
                return 0      # avanza un poco
            else:
                return 1      # corrige derecha
        return 0              # avanza

    # 5) Si no hay pared izquierda, pero detecta algo a la derecha → zig-zag suave
    if right > wall_present or front_right > wall_present:
        return 2 if front_right > 0.5 else 0

    # 6) Si no detecta nada relevante → avanza
    return 0




def main():
    parser = argparse.ArgumentParser(description="EFSIM Master Runner - High Performance")
    parser.add_argument("--type", type=str, default="gru", choices=["mlp", "gru", "res", "random", "braitenberg", "wall_follow"])
    parser.add_argument("--model", type=str, required=True, help="Ruta al archivo .json del modelo")
    parser.add_argument("--map", type=str, default="default", choices=["default", "n_shape"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--agents", type=int, default=2, help="1 para modelos viejos, 2 para sociales")
    parser.add_argument("--render", action="store_true", help="Mostrar ventana de simulación")
    parser.add_argument("--save_video", action="store_true", help="Grabar el primer episodio")
    parser.add_argument("--save_snapshot", action="store_true", help="Guarda PNG en alta resolución al final") 
    parser.add_argument("--out", type=str, default=None, help="Nombre del archivo de video")
    parser.add_argument("--sleep", type=float, default=0.02, help="Velocidad de renderizado")
    parser.add_argument("--u_env", action="store_true", help="Activa el laberinto en U del benchmark")
    parser.add_argument("--no_bg", action="store_true", help="Desactiva el Ganglio Basal (Modo Mudo)")
    parser.add_argument("--no_social_filter", action="store_true", help="Desactiva el filtro manual social del follower")
    parser.add_argument("--random_spawns", action="store_true", help="Randomiza posiciones iniciales de agentes")
    parser.add_argument("--random_food", action="store_true", help="Randomiza posición de la comida")
    parser.add_argument("--fixed_orientations", action="store_true", help="Fija orientación inicial en 0.0")
    parser.add_argument(
        "--social_mode",
        type=str,
        default="normal",
        choices=["normal", "off", "shuffled", "mute_sender"],
        help="Modo de canal social para ablaciones"
    )
    parser.add_argument(
        "--social_variant",
        type=str,
        default="angle_strength",
        choices=["angle_strength", "angle_only", "strength_only"],
        help="Codificación del canal social"
    )
    parser.add_argument("--save_csv", action="store_true", help="Guarda métricas por episodio en CSV")
    parser.add_argument("--csv_out", type=str, default=None, help="Nombre del archivo CSV de salida")
    parser.add_argument("--version", action="version", version=f"EFSIM {get_version()}")
    args = parser.parse_args()

    # 1. Configuración del Entorno
    env_config = ForagingEnvConfig(
        map_name=args.map,
        num_agents=args.agents,
        u_env=args.u_env,
        max_steps=1000,  # <--- AÑADE ESTA LÍNEA (ej. 1000 pasos en lugar de 600) COMM
        randomize_spawns=args.random_spawns,
        randomize_food=args.random_food,
        randomize_orientations=not args.fixed_orientations,
        social_mode=args.social_mode,
        social_variant=args.social_variant,
    )

    render_mode = "rgb_array" if args.save_video else ("human" if args.render else None)
    env = ForagingEnv(config=env_config, render_mode=render_mode)

    # 2. Cargar Controladores
    if args.type == "wall_follow":
        controllers = [DummyController() for _ in range(args.agents)]
    else:
        ctrl_params = {
            "model_path": args.model,
            "num_agents": args.agents,
            "use_bg": not args.no_bg,
            "use_social_filter": not args.no_social_filter,
        }
        controllers = [make_controller(args.type, agent_idx=i, **ctrl_params) for i in range(args.agents)]

    print(f"\n--- SISTEMA EFSIM: Corriendo {args.type.upper()} con {args.agents} agentes ---")

    results = []
    for ep in range(args.episodes):
        obs_list, info = env.reset()
        
        # --- PARCHE DE COMIDA INTELIGENTE PARA EL RUN ---
        if args.random_food:
            if args.u_env:
                # Si es la U, forzamos la comida ADENTRO del laberinto
                env.food_pos = np.array([
                    np.random.uniform(280, 500), 
                    np.random.uniform(280, 430)
                ], dtype=np.float32)
            else:
                # Si es el mapa default, cualquier lugar es bueno
                env.food_pos = env._sample_food().astype(np.float32)
            
        for c in controllers:
            c.reset()
        frames, done = [], False

        stuck_counters = [0] * args.agents
        last_positions = [p.copy() for p in env.agent_pos]
        ep_stuck_steps = 0

        while not done:
            if args.type == "wall_follow":
                actions = [wall_follow_policy(obs_list[i]) for i in range(args.agents)]
            else:
                actions = [controllers[i].act(obs_list[i], info) for i in range(args.agents)]
                
             # GANGLIO BASAL (Mismo que en entrenamiento) ---
            for i in range(args.agents):
                # Calculamos si se movió respecto al frame anterior
                dist_moved = np.linalg.norm(env.agent_pos[i] - last_positions[i])
                
                # 1. Instinto de Reclutamiento (Si ya llegó)
                if info['individual_success'][i] and random.random() < 0.7:
                    actions[i] = 4
                # 2. Instinto de Auxilio (Si gira pero no se mueve)
                elif actions[i] in [1, 2] and dist_moved < 0.5 and random.random() < 0.2:
                    actions[i] = 4
            # ------------------------------------------------------------

            obs_list, _, terminated, truncated, info = env.step(actions)          

            at = info['individual_success']
            for i in range(args.agents):
                if not at[i]:
                    dist_moved = np.linalg.norm(env.agent_pos[i] - last_positions[i])
                    if dist_moved < 1.0:
                        stuck_counters[i] += 1
                        if stuck_counters[i] > 30:
                            ep_stuck_steps += 1
                    else:
                        stuck_counters[i] = 0
                last_positions[i] = env.agent_pos[i].copy()

            if args.save_video or args.render:
                frame = env.render()
                if args.save_video and ep == 0:
                    frames.append(frame)

            done = terminated or truncated
            if args.render and not args.save_video:
                time.sleep(args.sleep)

        if args.save_video and ep == 0 and len(frames) > 0:
            if info['success']:
                for _ in range(20):
                    frames.append(frames[-1])

            video_name = args.out if args.out else f"video_{args.type}_{args.map}.mp4"
            imageio.mimsave(video_name, frames, fps=30, macro_block_size=1)
            print(f"[LOG] Video guardado como: {video_name}")
            
        if args.save_snapshot:
            # 1. Forzamos un renderizado para asegurar que la superficie exista
            env.render() 
            # 2. Obtenemos la superficie activa de Pygame
            final_surface = pygame.display.get_surface()
            
            if final_surface is not None:
                # 3. Escalamos al 300% (2400x1800 px) con suavizado
                hi_res_size = (env.config.world_width * 3, env.config.world_height * 3)
                hi_res_surface = pygame.transform.smoothscale(final_surface, hi_res_size)
                # 4. Guardamos como PNG
                snap_name = f"snapshot_ep{ep+1}_highres.png"
                pygame.image.save(hi_res_surface, snap_name)
                print(f"[LOG] High-Res Snapshot saved: {snap_name}")
            else:
                print("[ERROR] Could not capture surface for snapshot.")

        info['stuck_steps'] = ep_stuck_steps
        results.append(info)
        status = "EXITO" if info['success'] else "FALLO"
        print(
            f"Episodio {ep+1}: {status} en {info['step_count']} pasos | "
            f"SuccessSteps={info['success_steps']} | "
            f"Gap={info['goal_gap']} | "
            f"Screams={info['signal_counts']} | "
            f"Stuck={ep_stuck_steps}"
        )

    success_rate = np.mean([1.0 if r["success"] else 0.0 for r in results])
    avg_steps = np.mean([r["step_count"] for r in results])
    avg_stuck = np.mean([r["stuck_steps"] for r in results])

    valid_gaps = [r["goal_gap"] for r in results if r["goal_gap"] >= 0]
    avg_gap = np.mean(valid_gaps) if len(valid_gaps) > 0 else -1.0

    total_signal_counts = [sum(r["signal_counts"]) for r in results]
    avg_total_signals = np.mean(total_signal_counts)

    print(f"\n=== RESULTADOS FINALES ===")
    print(f"Éxito Grupal: {success_rate*100:.1f}%")
    print(f"Promedio de pasos: {avg_steps:.1f}")
    print(f"Promedio de Pasos Atorados (STUCK): {avg_stuck:.1f}")
    print(f"Promedio Goal Gap: {avg_gap:.1f}" if avg_gap >= 0 else "Promedio Goal Gap: N/A")
    print(f"Promedio de señales totales: {avg_total_signals:.1f}")

    if args.save_csv:
        csv_name = args.csv_out if args.csv_out else f"results_{args.type}_{args.map}_{args.social_mode}.csv"
        with open(csv_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "success",
                "step_count",
                "success_steps",
                "first_goal_step",
                "second_goal_step",
                "goal_gap",
                "signal_counts",
                "stuck_steps",
                "social_mode",
                "social_variant",
                "random_spawns",
                "random_food",
                "u_env",
            ])

            for ep_idx, r in enumerate(results, start=1):
                writer.writerow([
                    ep_idx,
                    int(r["success"]),
                    r["step_count"],
                    "|".join(map(str, r["success_steps"])),
                    r["first_goal_step"],
                    r["second_goal_step"],
                    r["goal_gap"],
                    "|".join(map(str, r["signal_counts"])),
                    r["stuck_steps"],
                    args.social_mode,
                    args.social_variant,
                    int(args.random_spawns),
                    int(args.random_food),
                    int(args.u_env),
                ])

        print(f"[LOG] CSV guardado como: {csv_name}")

    env.close()


if __name__ == "__main__":
    main()
