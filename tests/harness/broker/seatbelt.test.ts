import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, realpathSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import {
  type FencePlaces,
  SANDBOX_EXEC,
  SeatbeltRunner,
  seatbeltProfile,
} from "../../../src/harness/broker/seatbelt.ts";

const mac = process.platform === "darwin";
const onMac = { skip: mac ? false : "Seatbelt is macOS's: these run on a Mac" };

/** Made-up places: the profile is text, so it can be checked on any machine. */
const PLACES: FencePlaces = {
  stateDir: "/rig/state",
  scratchDir: "/rig/scratch",
  secretDirs: ["/rig/secrets"],
  home: "/home/user",
  runtime: ["/opt/node"],
  agentRuntime: ["/opt/thermite"],
  metalCache: "/cache/com.apple.metal",
  userTemp: "/temp",
};
const RUN = {
  gpu: false,
  cwd: "/rig/scratch/runs/1/work",
  argv: ["/bin/true"],
  env: {},
  handed: [],
  others: [],
};

describe("the Seatbelt profile", () => {
  test("the judge's roles get no network; only the timing roles and a lane's device runs get the GPU", () => {
    const of = (role: string, gpu = false) =>
      seatbeltProfile({ ...RUN, role, gpu }, PLACES).profile;
    const noGpu = '(deny iokit-open (iokit-user-client-class "AGXDeviceUserClient"))';
    for (const role of ["builder", "run-2", "checker", "driver-1"])
      assert.match(of(role), /\(deny network\*\)/, role);
    for (const role of ["lane-1", "agent-3", "planner", "model-check"])
      assert.doesNotMatch(of(role), /network/, role);
    for (const role of ["run", "run-3", "lane-2-dev"]) assert.ok(!of(role).includes(noGpu), role);
    assert.ok(!of("lane-1", true).includes(noGpu), "a lane's run on the device");
    for (const role of ["builder", "checker", "driver", "lane-1", "agent-1", "planner"])
      assert.ok(of(role).includes(noGpu), role);
    // Paths are parameters, never text in the profile.
    assert.ok(!of("run").includes("/rig/"));
  });

  test("a checker's baseline folder: a job's in the runs folder, written; a sealed one, read only; also by a symlink; never elsewhere", () => {
    const top = realpathSync(mkdtempSync(join(tmpdir(), "thermite-sb-")));
    mkdirSync(join(top, "state", "baselines", "b1"), { recursive: true });
    mkdirSync(join(top, "scratch", "runs", "job-1", "baseline"), { recursive: true });
    symlinkSync(top, join(top, "link"));
    const places = { ...PLACES, stateDir: join(top, "state"), scratchDir: join(top, "scratch") };
    const as = (dir: string) => {
      const { profile, params } = seatbeltProfile(
        { ...RUN, role: "checker", env: { THERMITE_BASELINE_DIR: dir } },
        places,
      );
      const name = params.find((p) => p.endsWith(`=${realpathSync(dir)}`))?.split("=")[0];
      assert.ok(name, dir);
      const write = profile.split("\n").findLast((l) => l.startsWith("(allow file-write*"));
      return write?.includes(`(param "${name}")`) ? "write" : "read";
    };
    assert.equal(as(join(top, "scratch", "runs", "job-1", "baseline")), "write");
    assert.equal(as(join(top, "link", "scratch", "runs", "job-1", "baseline")), "write");
    assert.equal(as(join(top, "state", "baselines", "b1")), "read");
    assert.equal(as(join(top, "link", "state", "baselines", "b1")), "read");
    assert.throws(() => as(join(top, "state", "ledger")), /refuses the baseline folder/);
    assert.throws(() => as(join(top, "state", "baselines")), /refuses the baseline folder/);
    assert.throws(() => as(join(top, "scratch", "runs")), /refuses the baseline folder/);
  });

  test("refuses a path with a quote or a backslash", () => {
    assert.throws(
      () => seatbeltProfile({ ...RUN, role: "run", cwd: '/rig/scratch/a"b' }, PLACES),
      /refuses the path/,
    );
    assert.throws(
      () => seatbeltProfile({ ...RUN, role: "run", handed: ["/rig/scratch/a\\b"] }, PLACES),
      /refuses the path/,
    );
  });
});

describe("Seatbelt on this Mac", () => {
  test(
    "the last rule that matches wins: an allow inside an earlier denied folder opens it",
    onMac,
    () => {
      const root = realpathSync(mkdtempSync(join(tmpdir(), "thermite-sbpl-")));
      mkdirSync(join(root, "outer", "inner"), { recursive: true });
      writeFileSync(join(root, "outer", "secret.txt"), "secret\n");
      writeFileSync(join(root, "outer", "inner", "open.txt"), "open\n");
      const cat = (profile: string, file: string) =>
        spawnSync(
          SANDBOX_EXEC,
          [
            "-D",
            `OUTER=${root}/outer`,
            "-D",
            `INNER=${root}/outer/inner`,
            "-p",
            profile,
            "/bin/cat",
            file,
          ],
          { encoding: "utf8" },
        ).status;
      const denyThenAllow =
        '(version 1)(allow default)(deny file-read* (subpath (param "OUTER")))(allow file-read* (subpath (param "INNER")))';
      assert.notEqual(cat(denyThenAllow, join(root, "outer", "secret.txt")), 0);
      assert.equal(cat(denyThenAllow, join(root, "outer", "inner", "open.txt")), 0);
      const allowThenDeny =
        '(version 1)(allow default)(allow file-read* (subpath (param "INNER")))(deny file-read* (subpath (param "OUTER")))';
      assert.notEqual(cat(allowThenDeny, join(root, "outer", "inner", "open.txt")), 0);
    },
  );

  test(
    "a run reads its folder and a build named in its command, never the state or another role's folder",
    onMac,
    async () => {
      const root = realpathSync(mkdtempSync(join(tmpdir(), "thermite-fence-")));
      const stateDir = join(root, "state");
      const scratchDir = join(root, "scratch");
      const build = join(stateDir, "builds", "k1", "out");
      mkdirSync(build, { recursive: true });
      mkdirSync(scratchDir);
      writeFileSync(join(build, "data.txt"), "built\n");
      writeFileSync(join(stateDir, "ledger.txt"), "the ledger\n");
      const runner = new SeatbeltRunner({ stateDir, scratchDir, env: {} });
      assert.equal(runner.fence, "seatbelt");
      assert.equal(runner.separated, false);
      const mine = join(scratchDir, "mine");
      const theirs = join(scratchDir, "theirs");
      mkdirSync(mine);
      mkdirSync(theirs);
      writeFileSync(join(theirs, "x.txt"), "theirs\n");
      await runner.handOver(mine, "run");
      await runner.handOver(theirs, "lane-1");
      const cat = async (file: string) =>
        (await runner.run({ role: "run", argv: ["/bin/cat", file], cwd: mine, timeoutMs: 20_000 }))
          .exitCode;
      assert.equal(await cat(join(build, "data.txt")), 0, "a build named in its command");
      assert.notEqual(await cat(join(stateDir, "ledger.txt")), 0, "the state");
      assert.notEqual(await cat(join(theirs, "x.txt")), 0, "another role's folder, though named");
      // Taken back from the lane, the folder is no other role's: a run that names it may read it.
      await runner.reclaim(theirs);
      assert.equal(await cat(join(theirs, "x.txt")), 0);
      const write = await runner.run({
        role: "run",
        argv: ["/bin/sh", "-c", 'echo x > "$0"', join(scratchDir, "outside.txt")],
        cwd: mine,
        timeoutMs: 20_000,
      });
      assert.notEqual(write.exitCode, 0, "a write outside its folders");
      assert.match(execFileSync("/bin/ls", [scratchDir], { encoding: "utf8" }), /^mine\ntheirs\n$/);
    },
  );

  test("the fence is refused off macOS", { skip: mac ? "this is a Mac" : false }, () => {
    assert.throws(
      () => new SeatbeltRunner({ stateDir: "/s", scratchDir: "/t", env: {} }),
      /needs macOS/,
    );
  });
});
