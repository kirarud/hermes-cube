#!/bin/zsh
# Hermes Face for macOS — uses the project venv (numpy + moderngl), no cloud dependency.
cd -- "${0:A:h}" || exit 1
if [[ ! -x ./venv/bin/python ]]; then
  echo 'Не найден Python проекта: venv/bin/python'
  exit 1
fi
export HERMES_MAC_OPAQUE="${HERMES_MAC_OPAQUE:-1}"
exec ./venv/bin/python -u main.py
