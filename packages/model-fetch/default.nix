{
  python3Packages,
  writers,
}:
writers.writePython3Bin "model-fetch" {
  libraries = [python3Packages.huggingface-hub];
  flakeIgnore = ["E501"];
} (builtins.readFile ./model-fetch.py)
