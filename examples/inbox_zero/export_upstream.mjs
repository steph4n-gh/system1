// Read the pinned Inbox Zero contract and authored evaluation cases, without
// executing its application or making teacher calls. Requires its installed
// TypeScript dependency. Upstream text stays in the user's ignored output dir.
import { createRequire } from "node:module";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";

const [checkout, output] = process.argv.slice(2);
if (!checkout || !output) throw new Error("Usage: node export_upstream.mjs INBOX_ZERO_CHECKOUT OUTPUT_DIRECTORY");
const root = resolve(checkout);
const require = createRequire(join(root, "apps/web/package.json"));
const ts = require("typescript");
const pinned = "2571ce4970aa5d024bb664a6739bd91b1172c367";
const commit = pinned;
const digests = {};
function file(path) {
  const text = execFileSync("git", ["show", `${pinned}:${path}`], { cwd: root, encoding: "utf8" });
  digests[path] = createHash("sha256").update(text).digest("hex");
  return ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true);
}
function initializer(source, name) {
  for (const statement of source.statements) {
    if (!ts.isVariableStatement(statement)) continue;
    for (const declaration of statement.declarationList.declarations) {
      if (declaration.name.getText(source) === name) return declaration.initializer;
    }
  }
  throw new Error(`Missing ${name}`);
}
function literal(node) {
  if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) return node.text;
  if (ts.isArrayLiteralExpression(node)) return node.elements.map(literal);
  if (ts.isObjectLiteralExpression(node)) return Object.fromEntries(node.properties.map(property => {
    if (!ts.isPropertyAssignment(property)) throw new Error("Unsupported object property");
    return [property.name.text, literal(property.initializer)];
  }));
  if (ts.isCallExpression(node) && node.expression.getText() === "getEmail" && node.arguments.length === 1) {
    return literal(node.arguments[0]);
  }
  throw new Error(`Unsupported fixture expression: ${node.kind}`);
}
const configs = file("apps/web/utils/rule/consts.ts");
const configObject = initializer(configs, "ruleConfig");
const categories = ["Newsletter", "Marketing", "Calendar", "Receipt", "Notification", "OTP", "Conversations"];
const criteria = {};
for (const property of configObject.properties) {
  if (!ts.isPropertyAssignment(property) || !ts.isObjectLiteralExpression(property.initializer)) continue;
  const fields = property.initializer.properties.filter(ts.isPropertyAssignment);
  const name = fields.find(field => field.name.getText(configs) === "name");
  const instructions = fields.find(field => field.name.getText(configs) === "instructions");
  if (name && instructions && categories.includes(literal(name.initializer))) {
    criteria[literal(name.initializer)] = literal(instructions.initializer);
  }
}
criteria.Conversations = literal(initializer(file("apps/web/utils/ai/choose-rule/run-rules.ts"), "CONVERSATION_TRACKING_INSTRUCTIONS"));
const chooser = file("apps/web/utils/ai/choose-rule/classifier-choose-rule.ts");
criteria.None = literal(initializer(chooser, "NONE_DEFINITION"));
const question = literal(initializer(chooser, "QUESTION"));
const evaluation = file("apps/web/__tests__/eval/choose-rule.test.ts");
const cases = ["testCases", "promotionalBoundaryCases"].flatMap(cohort => literal(initializer(evaluation, cohort)).map((row, index) => ({
  id: `${cohort}:${index}`, cohort,
  email: {
    from: row.email.from ?? "", subject: row.email.subject ?? "", content: row.email.content ?? "",
    hasListUnsubscribeHeader: Boolean(row.email.listUnsubscribe),
  },
  acceptable: Array.isArray(row.expectedRule) ? row.expectedRule : [row.expectedRule],
})));
mkdirSync(output, { recursive: true });
writeFileSync(join(output, "contract.json"), JSON.stringify({ upstream: { repository: "elie222/inbox-zero", commit }, categories, question, criteria }, null, 2) + "\n");
writeFileSync(join(output, "upstream-evaluation.json"), JSON.stringify({ upstream: { repository: "elie222/inbox-zero", commit, file_sha256: digests }, provenance: "Public maintainer-authored regression examples; not private mailbox data or production accuracy evidence. Never used for fitting or calibration.", cases }, null, 2) + "\n");
writeFileSync(join(output, "UPSTREAM_LICENSE"), execFileSync("git", ["show", `${pinned}:LICENSE`], { cwd: root }));
console.log(`Exported ${cases.length} upstream cases and ${categories.length} category definitions to ${output}`);
