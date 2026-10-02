import sys

from .config import load_config
from .model.transformer import Transformer


def main() -> None:
    print("agentmodel CLI", file=sys.stderr)
    cfg = load_config("configs/pretrain_nano.yaml")
    model = Transformer(cfg.model)
    print(f"params: {model.num_params():,}")


if __name__ == "__main__":
    main()
