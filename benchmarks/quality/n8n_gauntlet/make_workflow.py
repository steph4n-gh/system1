"""Export a credential-free n8n workflow bound to the development artifact."""
import json
import uuid

from develop import OUTPUT
from runtime_probe import digest


def main():
    manifest_path = OUTPUT / "clinc-runtime-candidate/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = digest(manifest_path)
    credential = {"httpHeaderAuth": {"id": "SYSTEM1_LOCAL_CREDENTIAL", "name": "System1 local classifier"}}
    nodes, connections = [], {}

    def node(name, kind, version, x, y, parameters, **extra):
        nodes.append(dict(id=str(uuid.uuid5(uuid.NAMESPACE_URL, "system1/n8n/" + name)), name=name,
            type="n8n-nodes-base." + kind, typeVersion=version, position=[x, y], parameters=parameters, **extra))

    def connect(source, target, output=0, input_index=0):
        outputs = connections.setdefault(source, {"main": []})["main"]
        while len(outputs) <= output:
            outputs.append([])
        outputs[output].append(dict(node=target, type="main", index=input_index))

    node("Requests webhook", "webhook", 2, 0, 200,
         dict(httpMethod="POST", path="system1-development", authentication="headerAuth", responseMode="responseNode", options={}),
         webhookId="system1-development", credentials=credential)
    node("Requests", "splitOut", 1, 220, 200,
         dict(fieldToSplitOut="body.requests", options=dict(destinationFieldName="request")))
    body = dict(categories=manifest["categories"], instructions=manifest["instructions"])
    node("Local System1", "httpRequest", 4.2, 440, 200,
         dict(method="POST", url="http://127.0.0.1:8792/v1/classify", authentication="genericCredentialType",
              genericAuthType="httpHeaderAuth", sendBody=True, specifyBody="json",
              jsonBody="={{ { ..." + json.dumps(body) + ", text: $json.request.text } }}",
              options=dict(timeout=5000)), credentials=credential, onError="continueRegularOutput")
    node("Keep item identity", "set", 3.4, 660, 200,
         dict(assignments=dict(assignments=[
             dict(id="id", name="id", type="string", value="={{ $('Requests').item.json.request.id }}"),
             dict(id="text", name="text", type="string", value="={{ $('Requests').item.json.request.text }}")]), includeOtherFields=True, options={}))
    node("Accepted by this artifact", "if", 2.2, 900, 200,
         dict(conditions=dict(options=dict(caseSensitive=True, typeValidation="strict", version=2), combinator="and", conditions=[
             dict(id="accepted", leftValue="={{ $json.needsReview === false }}", rightValue=True,
                  operator=dict(type="boolean", operation="true", singleValue=True)),
             dict(id="artifact", leftValue="={{ $json.candidate ?? '' }}", rightValue=identity,
                  operator=dict(type="string", operation="equals"))]), options={}))
    for name, route, y in (("Local decision", "local", 100), ("Explicit review", "review", 320)):
        if route == "review":
            node(name, "set", 3.4, 1160, y, dict(mode="raw", jsonOutput="={{ {id:$json.id, text:$json.text, "
                "route:'review', qualification:'development_only', needsReview:true, category:null, teacherCalls:0, "
                "candidate:$json.candidate ?? null, reason:$json.reason ?? ($json.error ? 'local_service_failure' : 'uncertain')} }}",
                includeOtherFields=False, options={}))
            continue
        node(name, "set", 3.4, 1160, y, dict(assignments=dict(assignments=[
            dict(id="route", name="route", type="string", value=route),
            dict(id="qualification", name="qualification", type="string", value="development_only")]),
            includeOtherFields=True, options={}))
    node("Collect decisions", "merge", 3.2, 1410, 200, dict(mode="append", numberInputs=2))
    node("Return decisions", "respondToWebhook", 1.4, 1640, 200, dict(respondWith="allIncomingItems", options={}))
    node("Development notice", "stickyNote", 1, 300, -140, dict(width=1150, height=200, color=5,
         content="## System1 — development integration\nThis exact saved candidate runs locally with no teacher calls. Uncertain inputs and service failures go to review.\n\n**Not qualified:** the full CLINC + BANKING77 gauntlet is still incomplete. This workflow is an integration check, not the promised takeover recording."))
    for source, target in (("Requests webhook", "Requests"), ("Requests", "Local System1"),
                           ("Local System1", "Keep item identity"), ("Keep item identity", "Accepted by this artifact")):
        connect(source, target)
    connect("Accepted by this artifact", "Local decision", 0)
    connect("Accepted by this artifact", "Explicit review", 1)
    connect("Local decision", "Collect decisions", input_index=0)
    connect("Explicit review", "Collect decisions", input_index=1)
    connect("Collect decisions", "Return decisions")
    workflow = dict(name="System1 — development classification", nodes=nodes, connections=connections,
                    settings=dict(executionOrder="v1"), active=False)
    from pathlib import Path
    path = Path(__file__).with_name("n8n-development-workflow.json")
    path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote workflow bound to {identity}")


if __name__ == "__main__":
    main()
