from dataSave import *
from fileTreat import *
import math
import sys
import os

if len(sys.argv) < 3:
    print("Uso: python3 exampleVTK.py <ID> <velocity_set>")
    sys.exit(1)

simulation_id = sys.argv[1]
velocity_set = sys.argv[2]

path = f"./../bin/{velocity_set}/{simulation_id}/"

if not os.path.exists(path):
    print(f"Erro: O caminho {path} não existe.")
    sys.exit(1)

macrSteps = getMacrSteps(path)
info = getSimInfo(path)

for step in macrSteps:
    macr = getMacrsFromStep(step, path)
    print("Processando passo", step)
    saveVTK3D(macr, path, info['ID'] + "macr" + str(step).zfill(6), points=True)
