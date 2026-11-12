import { chmodSync, existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { materializeSkills, packageSkillsDir, readSkills, resolveSkillsDir } from "./skills.js";

let dir: string;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "exp-code-skills-"));
});
afterEach(() => rm(dir, { recursive: true, force: true }));

describe("document skills", () => {
  it("ships the six document skills in Agent Skills format", () => {
    const skills = readSkills(packageSkillsDir());
    expect(Object.keys(skills).sort()).toEqual(["charts", "data-analysis", "docx", "pdf", "pptx", "xlsx"]);
    for (const [name, text] of Object.entries(skills)) {
      const frontmatter = /^---\n([\s\S]*?)\n---\n/.exec(text)?.[1] ?? "";
      expect(frontmatter).toMatch(new RegExp(`^name: ${name}$`, "m"));
      const description = /^description: (.+)$/m.exec(frontmatter)?.[1] ?? "";
      expect(description.length).toBeGreaterThan(40);
      expect(description.length).toBeLessThanOrEqual(1024);
      expect(name).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
    }
  });

  it("uses the package directory when skills are not embedded", () => {
    expect(resolveSkillsDir(dir)).toBe(packageSkillsDir());
  });

  it("materializes embedded skills into a content-addressed directory", () => {
    const first = materializeSkills({ docx: "---\nname: docx\n---\n" }, dir);
    expect(readFileSync(join(first, "docx", "SKILL.md"), "utf8")).toContain("name: docx");
    expect(materializeSkills({ docx: "---\nname: docx\n---\n" }, dir)).toBe(first);
    const second = materializeSkills({ docx: "changed" }, dir);
    expect(second).not.toBe(first);
    expect(existsSync(second)).toBe(true);
  });

  it("makes skills readable by every account and writable by none, for a runner whose tools run as another user", () => {
    const skillsDir = materializeSkills({ pptx: "---\nname: pptx\n---\n" }, join(dir, "share"));
    expect(statSync(skillsDir).mode & 0o777).toBe(0o755);
    expect(statSync(join(skillsDir, "pptx")).mode & 0o777).toBe(0o755);
    expect(statSync(join(skillsDir, "pptx", "SKILL.md")).mode & 0o777).toBe(0o644);
  });

  it("refuses a skills directory others can change (a planted skill would steer the agent)", () => {
    const skills = { pptx: "---\nname: pptx\n---\n" };
    const planted = materializeSkills(skills, join(dir, "a"));
    chmodSync(planted, 0o777);
    expect(() => materializeSkills(skills, join(dir, "a"))).toThrow(/refusing skills directory/);
    mkdirSync(join(dir, "open"), { mode: 0o777 });
    chmodSync(join(dir, "open"), 0o777);
    expect(() => materializeSkills(skills, join(dir, "open"))).toThrow(/refusing skills root/);
    // A sticky shared root (like /tmp) is fine: nobody can replace our entries in it.
    mkdirSync(join(dir, "sticky"));
    chmodSync(join(dir, "sticky"), 0o1777);
    expect(existsSync(materializeSkills(skills, join(dir, "sticky")))).toBe(true);
  });
});
