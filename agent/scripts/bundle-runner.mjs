// Bundles the headless runner (src/runner/entry.ts) into one self-contained ESM file,
// dist/runner.mjs, with the document skills embedded, for copying into a sandbox.
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { isBuiltin } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const skillsDir = join(root, "skills");
const skills = {};
for (const name of readdirSync(skillsDir)) {
  const file = join(skillsDir, name, "SKILL.md");
  if (statSync(join(skillsDir, name)).isDirectory() && existsSync(file)) skills[name] = readFileSync(file, "utf8");
}

const outfile = join(root, "dist", "runner.mjs");
const result = await build({
  entryPoints: [join(root, "src", "runner", "entry.ts")],
  outfile,
  bundle: true,
  platform: "node",
  target: "node22.19",
  format: "esm",
  minify: true,
  legalComments: "eof",
  metafile: true,
  logLevel: "warning",
  define: { __EXP_CODE_SKILLS__: JSON.stringify(skills), PI_BUNDLED_NODE: "true" },
  banner: {
    js: "// exp-code headless runner. Contains Pi (MIT, Copyright (c) Mario Zechner and contributors).\nimport { createRequire as __expCodeRequire } from 'node:module'; const require = __expCodeRequire(import.meta.url);"
  }
});

// Fail when the bundle still imports a package: the sandbox has no node_modules. These optional
// accelerators and helpers are loaded behind try/catch by their callers, as in Pi's own bundle.
const optional = new Set(["bufferutil", "utf-8-validate", "supports-color", "kerberos"]);
const external = new Set();
for (const input of Object.values(result.metafile.inputs)) {
  for (const imported of input.imports) {
    if (!imported.external || isBuiltin(imported.path) || optional.has(imported.path) || imported.path.startsWith("<define:")) continue;
    external.add(imported.path);
  }
}
if (external.size > 0) throw new Error(`runner bundle left external imports: ${[...external].sort().join(", ")}`);

const source = readFileSync(outfile);
const sha256 = createHash("sha256").update(source).digest("hex");
console.log(`dist/runner.mjs: ${source.length} bytes, sha256 ${sha256}`);
