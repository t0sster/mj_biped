FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip \
      libgl1 \
      libglfw3 \
      libegl1 \
      libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages \
      "mujoco==3.6.0" \
      "numpy"

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

WORKDIR /ws
SHELL ["/bin/bash", "-c"]
CMD ["bash"]