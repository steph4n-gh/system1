"""Exercise the real imported n8n workflow, including single-branch batches."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

from develop import OUTPUT


def main(service_down=False):
    url = "http://127.0.0.1:5681/webhook/system1-development"
    token = os.environ["SYSTEM1_CLASSIFIER_TOKEN"]
    accepted = [dict(id="timer", text="Please set a timer for seven minutes."),
                dict(id="translate", text="Translate good morning into French.")]
    review = [dict(id="unfamiliar", text="Compose a symphony about translucent mountains."),
              dict(id="malformed", text=17)]
    scenarios = {"service_unavailable": accepted} if service_down else {
        "mixed": accepted + review, "all_local": accepted, "all_review": review}
    report = dict(scope="real n8n development integration; no quality qualification or teacher-disconnection claim",
                  n8n_version="2.39.10", service_down=service_down, scenarios=[])
    for name, rows in scenarios.items():
        request = urllib.request.Request(url, data=json.dumps(dict(requests=rows)).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=30) as response:
            outputs = json.load(response)
        elapsed = (time.perf_counter() - start) * 1000
        assert isinstance(outputs, list) and len(outputs) == len(rows)
        assert {r["id"] for r in outputs} == {r["id"] for r in rows}
        for output in outputs:
            expected_review = service_down or output["id"] in ("unfamiliar", "malformed")
            assert output["route"] == ("review" if expected_review else "local")
            assert output["needsReview"] is expected_review
            assert output["teacherCalls"] == 0
            assert output["qualification"] == "development_only"
            assert "error" not in output  # Internal HTTP errors are not the public review contract.
            if expected_review:
                assert output["category"] is None
            else:
                assert output["category"] == output["id"]
        report["scenarios"].append(dict(name=name, inputs=rows, outputs=outputs, total_round_trip_ms=elapsed))
    request = urllib.request.Request(url, data=json.dumps(dict(requests=accepted)).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer invalid"})
    try:
        urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as exc:
        assert exc.code in (401, 403)
        report["unauthorized_status"] = exc.code
    else:
        raise AssertionError("Invalid webhook credential accepted")
    path = OUTPUT / ("n8n-service-down.json" if service_down else "n8n-integration.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(scenarios=len(scenarios), decisions=sum(len(r["inputs"]) for r in report["scenarios"]),
                         service_down=service_down, unauthorized_status=report["unauthorized_status"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-down", action="store_true")
    main(parser.parse_args().service_down)
