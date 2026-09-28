"""Analisa o único vídeo em ``datasets/videos/input``."""

from pathlib import Path

from src.aggregation import MeanAggregator
from src.config import INPUT_VIDEO_DIR
from src.detection import CNNDetectionDetector
from src.extraction import MSSIMKeyframeExtractor
from src.pipeline import VideoDetectionPipeline


SUPPORTED_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def find_input_video(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        file
        for file in input_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(
            f"Nenhum vídeo encontrado em: {input_dir}\n"
            "Coloque um vídeo nessa pasta e execute novamente."
        )
    if len(videos) > 1:
        names = "\n".join(f"- {video.name}" for video in videos)
        raise RuntimeError(
            "Há mais de um vídeo na pasta de entrada. Deixe apenas um:\n" + names
        )
    return videos[0]


def main() -> None:
    video_path = find_input_video(Path(INPUT_VIDEO_DIR))
    pipeline = VideoDetectionPipeline(
        extractor=MSSIMKeyframeExtractor(threshold=0.50),
        detectors=[CNNDetectionDetector()],
        aggregator=MeanAggregator(),
    )
    result = pipeline.run(str(video_path))
    final_score = result["final_score"]

    print(f"Vídeo analisado: {video_path.name}")
    print(f"Keyframes extraídos: {len(result['keyframes'])}")
    print(f"Score de imagem sintética: {final_score:.6f} ({final_score:.2%})")
    print("Classificação: possivelmente sintético" if final_score >= 0.5 else "Classificação: possivelmente real")


if __name__ == "__main__":
    main()
