#!/bin/bash
# Prepara o ambiente para rodar o Isaac Sim no Docker

# Diretórios de cache e dados (necessário antes do primeiro docker compose up)
sudo mkdir -p ~/docker/isaac-sim/cache/main/ov
sudo mkdir -p ~/docker/isaac-sim/cache/main/warp
sudo mkdir -p ~/docker/isaac-sim/cache/computecache
sudo mkdir -p ~/docker/isaac-sim/config
sudo mkdir -p ~/docker/isaac-sim/data/documents
sudo mkdir -p ~/docker/isaac-sim/data/Kit
sudo mkdir -p ~/docker/isaac-sim/logs
sudo mkdir -p ~/docker/isaac-sim/pkg
sudo chown -R 1234:1234 ~/docker/isaac-sim

# Acesso ao display X11
xhost +local:

echo "Display: $DISPLAY"
echo "Xauthority: $HOME/.Xauthority"
echo "Pronto. Execute: docker compose up"
