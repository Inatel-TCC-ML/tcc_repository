from pathlib import Path
from detection import CNNDetector

detector = CNNDetector()

keyframes_folder = Path("select_frames/MSSIM/output")
keyframes = sorted(keyframes_folder.glob("*.jpg"))

# Trava o script aqui se a pasta estiver errada, ao invés de só imprimir e seguir
assert keyframes_folder.exists(), f"Pasta não encontrada: {keyframes_folder.resolve()}"
assert len(keyframes) > 0, "Nenhum arquivo .jpg encontrado na pasta"

print(f'{len(keyframes)} keyframes encontrados em {keyframes_folder.resolve()}')

scores = []
for keyframe in keyframes:
    prob = detector.predict(str(keyframe))
    scores.append(prob)
    print(f'{keyframe.name}: {prob:.4f}')

score_agregado = sum(scores) / len(scores)
print(f'\nScore agregado: {score_agregado:.4f}')