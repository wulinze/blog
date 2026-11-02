import json
from pathlib import Path


def read_result(path):
    data = json.loads(Path(path).read_text())
    assert data["device"] and data["versions"] and data["scope"]
    assert len(data["cases"]) == 4
    for case in data["cases"]:
        assert case["status"] == "PASS"
        assert case["max_abs"] >= 0
        if "composition_ms_samples" in case:
            samples = case["composition_ms_samples"]
            assert len(samples) >= 10 and all(t > 0 for t in samples)
    return data


if __name__ == "__main__":
    import sys

    result = read_result(sys.argv[1])
    print(result["device"], result["scope"], "metadata checks passed")
