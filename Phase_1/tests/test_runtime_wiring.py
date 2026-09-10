"""Structural checks for the standalone Phase 1 container image."""

from pathlib import Path
import shlex
import unittest


DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"


def dockerfile_instructions(path: Path) -> list[tuple[str, str]]:
    """Return logical Dockerfile instructions, joining backslash continuations."""
    instructions = []
    current = ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        current = f"{current} {line}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue

        opcode, argument = current.split(maxsplit=1)
        instructions.append((opcode.upper(), argument))
        current = ""

    if current:
        raise ValueError("Dockerfile has an unterminated continuation")
    return instructions


class DockerRuntimeWiringTests(unittest.TestCase):
    def test_ultralytics_install_preserves_the_pinned_base_image_release(self):
        """Upgrading Ultralytics beyond the pinned base image is a compatibility bug."""
        instructions = dockerfile_instructions(DOCKERFILE)
        base_images = [argument for opcode, argument in instructions if opcode == "FROM"]
        ultralytics_install_commands = [
            shlex.split(argument)
            for opcode, argument in instructions
            if opcode == "RUN" and "pip install" in argument
            if any(token.startswith("ultralytics") for token in shlex.split(argument))
        ]
        installed_packages = [
            token
            for command in ultralytics_install_commands
            for token in command
            if token.startswith("ultralytics")
        ]

        self.assertEqual(base_images, ["ultralytics/ultralytics:8.4.146"])
        self.assertEqual(installed_packages, ["ultralytics==8.4.146"])
        self.assertTrue(all("-U" not in command for command in ultralytics_install_commands))

    def test_image_copies_every_runtime_file_needed_by_run_script(self):
        """A direct image run missing configuration modules or YAML files is a bug."""
        instructions = dockerfile_instructions(DOCKERFILE)
        workdirs = [argument for opcode, argument in instructions if opcode == "WORKDIR"]
        copied_sources = set()

        for opcode, argument in instructions:
            if opcode != "COPY":
                continue
            parts = shlex.split(argument)
            sources, destination = parts[:-1], parts[-1]
            if destination in {".", "./", "/app"}:
                copied_sources.update(sources)

        self.assertEqual(workdirs, ["/app"])
        self.assertTrue(
            {
                "train.py",
                "run.sh",
                "tuning_config.py",
                "wandb_config.py",
                "config.yaml",
                "tuner.yaml",
            }.issubset(copied_sources)
        )


if __name__ == "__main__":
    unittest.main()
