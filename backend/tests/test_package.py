import docs_rag


def test_package_exposes_version() -> None:
    assert docs_rag.__version__ == "0.1.0"
