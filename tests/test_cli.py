from agentmodel.cli import main


def test_info_command_reports_model_metadata(capsys):
    assert main(["info", "configs/pretrain_nano.yaml"]) == 0
    output = capsys.readouterr().out
    assert "params:" in output
    assert "config_hash:" in output


def test_parser_accepts_pretrain_command():
    from agentmodel.cli import build_parser

    args = build_parser().parse_args(["pretrain", "configs/pretrain_nano.yaml", "--steps", "1"])
    assert args.command == "pretrain"
    assert args.steps == 1
