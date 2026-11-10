/**
 * The bundled document skills (`agent/skills/<name>/SKILL.md`, Agent Skills format).
 *
 * An installed package reads them from its own `skills/` directory. The single-file runner bundle
 * carries their text instead (injected at bundle time as `__EXP_CODE_SKILLS__`) and writes them to
 * a directory on first use, so Pi discovers both the same way.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

declare const __EXP_CODE_SKILLS__: Record<string, string> | undefined;

/** Skill text embedded by the runner bundle, or undefined in an installed package. */
function embeddedSkills(): Record<string, string> | undefined {
  return typeof __EXP_CODE_SKILLS__ === "undefined" ? undefined : __EXP_CODE_SKILLS__;
}

/** The package's `skills/` directory (two levels above both `src/extension` and `dist/extension`). */
export function packageSkillsDir(): string {
  return fileURLToPath(new URL("../../skills", import.meta.url));
}

/** Read every `<dir>/<name>/SKILL.md` into a name-to-text map. */
export function readSkills(dir: string): Record<string, string> {
  const skills: Record<string, string> = {};
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const file = join(dir, entry.name, "SKILL.md");
    if (entry.isDirectory() && existsSync(file)) skills[entry.name] = readFileSync(file, "utf8");
  }
  return skills;
}

/** Write a skill map under `root`, in a directory named by its content so updates never mix. */
export function materializeSkills(skills: Record<string, string>, root: string): string {
  const digest = createHash("sha256").update(JSON.stringify(skills)).digest("hex").slice(0, 16);
  const dir = join(root, `exp-code-skills-${digest}`);
  if (existsSync(dir)) return dir;
  const staging = `${dir}.${process.pid}.tmp`;
  for (const [name, text] of Object.entries(skills)) {
    mkdirSync(join(staging, name), { recursive: true });
    writeFileSync(join(staging, name, "SKILL.md"), text);
  }
  try {
    renameSync(staging, dir);
  } catch (error) {
    // Another process won the race; its directory has identical content.
    if (!existsSync(dir)) throw error;
  }
  return dir;
}

/** The directory Pi should discover the bundled skills in. */
export function resolveSkillsDir(materializeRoot: string = tmpdir()): string {
  const embedded = embeddedSkills();
  return embedded === undefined ? packageSkillsDir() : materializeSkills(embedded, materializeRoot);
}
