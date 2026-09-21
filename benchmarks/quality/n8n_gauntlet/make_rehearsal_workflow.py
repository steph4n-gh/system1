"""Export the development-only n8n teacher-disconnection workflow (no secrets)."""
import hashlib
import json
import uuid

from serve_rehearsal import DEFAULT_CANDIDATE, HERE


def workflow():
    manifest_path = DEFAULT_CANDIDATE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    credential = {"httpHeaderAuth": {"id": "SYSTEM1_LOCAL_CREDENTIAL", "name": "System1 local classifier"}}
    nodes, connections = [], {}

    def node(name, kind, version, x, y, parameters, **extra):
        nodes.append(dict(id=str(uuid.uuid5(uuid.NAMESPACE_URL, "system1/rehearsal/" + name)), name=name,
            type="n8n-nodes-base." + kind, typeVersion=version, position=[x, y], parameters=parameters, **extra))

    def connect(source, target, output=0, input_index=0):
        outputs = connections.setdefault(source, {"main": []})["main"]
        while len(outputs) <= output:
            outputs.append([])
        outputs[output].append(dict(node=target, type="main", index=input_index))

    def condition(name, expression, x, y):
        node(name, "if", 2.2, x, y, dict(conditions=dict(options=dict(caseSensitive=True,
            typeValidation="strict", version=2), combinator="and", conditions=[dict(id=name,
            leftValue="={{ " + expression + " }}", rightValue=True,
            operator=dict(type="boolean", operation="true", singleValue=True))]), options={}))

    def result(name, expression, x, y):
        node(name, "set", 3.4, x, y, dict(mode="raw", jsonOutput="={{ " + expression + " }}",
                                        includeOtherFields=False, options={}))

    def service(name, port, text, x, y, timeout):
        body = {key: manifest[key] for key in ("categories", "instructions")}
        node(name, "httpRequest", 4.2, x, y, dict(method="POST",
            url=f"http://127.0.0.1:{port}/v1/classify", authentication="genericCredentialType",
            genericAuthType="httpHeaderAuth", sendBody=True, specifyBody="json",
            jsonBody="={{ { ..." + json.dumps(body) + ", text: " + text + " } }}",
            options=dict(timeout=timeout)), credentials=credential, onError="continueRegularOutput")

    node("Requests webhook", "webhook", 2, 0, 200,
         dict(httpMethod="POST", path="system1-disconnection-rehearsal", authentication="headerAuth",
              responseMode="responseNode", options={}), webhookId="system1-disconnection-rehearsal", credentials=credential)
    node("Requests", "splitOut", 1, 220, 200,
         dict(fieldToSplitOut="body.requests", options=dict(destinationFieldName="request")))
    service("Saved System1 skill", 8793, "$json.request.text", 440, 200, 5000)
    bound = "$json.candidate === " + json.dumps(identity)
    allowed = json.dumps(list(manifest["categories"])) + ".includes($json.category)"
    condition("Accepted locally", bound + " && $json.needsReview === false && " + allowed, 660, 200)
    common = "id:$('Requests').item.json.request.id, text:$('Requests').item.json.request.text, qualification:'development_only'"
    result("Local decision", "{ ...$json, " + common + ", route:'local', teacherRequests:0, teacherCalls:0 }", 910, 60)
    condition("Valid request needs teacher", bound + " && $json.needsReview === true && !$json.reason", 910, 320)
    result("Contract or service review", "{ " + common + ", route:'review', needsReview:true, category:null, "
           "teacherRequests:0, teacherCalls:0, candidate:$json.candidate ?? null, "
           "reason:$json.reason ?? ($json.error ? 'local_service_failure' : 'candidate_mismatch') }", 1160, 500)
    service("Live Gemini teacher", 8794, "$('Requests').item.json.request.text", 1160, 260, 25000)
    accepted = "(" + bound + " && $json.needsReview === false && " + allowed + ")"
    result("Teacher answer or review", "{ " + common + ", route:" + accepted + " ? 'teacher' : 'review', "
           "needsReview:!" + accepted + ", category:" + accepted + " ? $json.category : null, "
           "candidate:$json.candidate ?? null, teacherRequests:1, "
           "teacherCalls:typeof $json.teacherCalls === 'number' ? $json.teacherCalls : null, "
           "teacherStatus:$json.error ? 'unavailable' : 'responded', "
           "reason:$json.error ? 'teacher_transport_failure' : ($json.reason ?? 'invalid_teacher_response'), "
           "usage:$json.usage ?? null, model:$json.model ?? null }", 1400, 260)
    node("Collect decisions", "merge", 3.2, 1650, 200, dict(mode="append", numberInputs=3))
    node("Return decisions", "respondToWebhook", 1.4, 1870, 200, dict(respondWith="allIncomingItems", options={}))
    node("Unqualified rehearsal", "stickyNote", 1, 200, -250, dict(width=1300, height=240, color=5,
         content="## DEVELOPMENT ONLY — teacher-disconnection rehearsal\n"
         "The first official quality gauntlet FAILED. These saved review candidates need fresh independent confirmation.\n\n"
         "Local decisions use the exact saved banking skill. Only uncertain valid requests call Gemini. Stop the separate teacher process: local decisions continue; uncertain requests go to review.\n\n"
         "New teacher answers do not retrain or promote this frozen skill. API attempts and usage are journaled. A missing teacher response means an unknown API call count, not a claimed zero."))
    for source, target in (("Requests webhook", "Requests"), ("Requests", "Saved System1 skill"),
        ("Saved System1 skill", "Accepted locally"), ("Live Gemini teacher", "Teacher answer or review"),
        ("Collect decisions", "Return decisions")):
        connect(source, target)
    connect("Accepted locally", "Local decision")
    connect("Accepted locally", "Valid request needs teacher", 1)
    connect("Valid request needs teacher", "Live Gemini teacher")
    connect("Valid request needs teacher", "Contract or service review", 1)
    for i, name in enumerate(("Local decision", "Teacher answer or review", "Contract or service review")):
        connect(name, "Collect decisions", input_index=i)
    return dict(name="System1 — UNQUALIFIED teacher-disconnection rehearsal", nodes=nodes,
                connections=connections, settings=dict(executionOrder="v1"), active=False)


if __name__ == "__main__":
    path = HERE / "n8n-disconnection-rehearsal.json"
    path.write_text(json.dumps(workflow(), indent=2, ensure_ascii=False) + "\n")
    print("Wrote credential-free development rehearsal")
