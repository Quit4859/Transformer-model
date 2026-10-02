from agentmodel.cli import build_parser, main


def test_check_parser_supports_skip_tests():
    args = build_parser().parse_args(["check", "--skip-tests"])
    assert args.command == "check"
    assert args.skip_tests is True


def test_check_validates_config_without_running_tests(capsys):
    assert main(["check", "--skip-tests"]) == 0
    output = capsys.readouterr().out
    assert "config_hash:" in output
    assert "params:" in output
