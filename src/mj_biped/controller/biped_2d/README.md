# Инференс biped_2d

Два независимых процесса, каждый в своём Docker-контейнере, общаются по ROS2:

```
 simulation                                 inference (ONNX)
 ───────────────────────────                ──────────────────
 собирает наблюдения (37) ── /biped_2d/observation ──▶ политика
 применяет PD по действию ◀── /biped_2d/action ──────  выдаёт действие (10)
 шагает физику + рисует
```

- **simulation/** — крутит физику робота на чистом MuJoCo (без mjlab), собирает
  наблюдения, ждёт действие, считает PD и шагает. Образ: ROS2 + `mujoco`.
- **inference/** — грузит политику `policy/model_350.onnx` и на каждый вектор
  наблюдений отвечает действием. Образ: ROS2 + `onnxruntime` (CPU).

Связь — стандартными `std_msgs/Float64MultiArray`.

## Запуск

Нужен Docker, для окна просмотрщика  X11:

```bash
xhost +local:root # один раз: доступ к дисплею

cd controller/biped_2d
docker compose up --build  # поднимет обе ноды, после откроется вьювер MuJoCo
```

Только инференс или только симуляция:

```bash
docker compose up --build inference
docker compose up --build simulation
```

Headless режим:

```bash
docker compose run --rm simulation \
  bash -lc "source /opt/ros/jazzy/setup.bash && \
            python3 /ws/controller/biped_2d/simulation/simulation_node.py --headless"
```